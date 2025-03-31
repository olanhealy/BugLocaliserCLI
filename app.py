import json
import re
import torch
import logging
import torch.nn.functional as F
from flask import Flask, request, Response
from transformers import RobertaTokenizer, RobertaConfig
from model import CustomRobertaClassifier

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
logging.basicConfig(
    filename="bug_localisation_input.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

JAVA_KEYWORDS = set([
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while"
])

def mask_identifiers(code):
    tokens = re.split(r'(\W)', code)
    id_map = {}
    id_counter = 1
    masked_tokens = []
    for tok in tokens:
        if re.match(r'^[a-zA-Z_]\w*$', tok) and tok not in JAVA_KEYWORDS:
            if tok not in id_map:
                id_map[tok] = f"VAR{id_counter}"
                id_counter += 1
            masked_tokens.append(id_map[tok])
        else:
            masked_tokens.append(tok)
    return ''.join(masked_tokens)

# ---------------------------
# LOAD MODEL & TOKENISER
# ---------------------------
MODEL_DIR = "/app/FINAL_MODEL"

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
special_tokens = {
    "additional_special_tokens": ["[CONTEXT]", "[SNIPPET]", "[COMMIT]", "[PARENT]"]
}
tokenizer.add_special_tokens(special_tokens)

config = RobertaConfig.from_pretrained("microsoft/codebert-base", num_labels=2)
config.vocab_size = len(tokenizer)

model = CustomRobertaClassifier(config)
model.load_state_dict(torch.load(f"{MODEL_DIR}/pytorch_model.bin", map_location="cpu"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# FUNCTIONS
# ---------------------------
def extract_code_context(diff_text):
    lines = [line.rstrip() for line in diff_text.split("\n")]
    context_lines = []
    snippet_lines_added = []
    snippet_lines_removed = []
    found_change = False
    hunk_start_line = None
    file_name = "UnknownFile.java"

    for i, line in enumerate(lines):
        # Skip diff metadata
        if line.startswith(("diff ", "index ", "--- ", "+++ ")):
            if line.startswith("+++ "):
                file_name = line[4:].strip().replace("b/", "").replace("\"", "")
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            if match:
                hunk_start_line = int(match.group(1))
            continue

        # Detect changes from git diff
        if line.startswith("-") and not line.startswith("---"):
            snippet_lines_removed.append(line[1:].lstrip())
            found_change = True
        elif line.startswith("+") and not line.startswith("+++"):
            snippet_lines_added.append(line[1:].lstrip())
            found_change = True
        else:
            if not found_change:
                context_lines.append(line.lstrip())

    snippet_lines = snippet_lines_added if snippet_lines_added else snippet_lines_removed
    snippet_text = "\n".join(snippet_lines).strip()
    context_text = "\n".join(context_lines[-3:])  

    return context_text, snippet_text, snippet_lines, file_name, hunk_start_line

def preprocess_input(diff_text, commit_msg="No commit message", parent_commit=""):
    context_text, snippet_text, snippet_lines, file_name, hunk_start = extract_code_context(diff_text)

    context_text_masked = mask_identifiers(context_text)
    snippet_text_masked = mask_identifiers(snippet_text)

    input_text = (
        f"[CONTEXT] {context_text_masked}\n"
        f"[SNIPPET] {snippet_text_masked}\n"
        f"[COMMIT] {commit_msg}\n"
        f"[PARENT] {parent_commit}"
    )
    return input_text, snippet_lines, file_name, hunk_start

def tokenise_for_model(text):
    encoded = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}

def classify_change(input_text):
    inputs = tokenise_for_model(input_text)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item()
    return pred_class, confidence, logits, probs

# ---------------------------
# FLASK ROUTE
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    diff_text = data.get("diff", "").strip()
    commit_msg = data.get("commit_msg", "No commit message available").strip()
    parent_commit = data.get("parent_commit", "").strip()

    if not diff_text:
        return Response(json.dumps({"error": "No diff provided"}), status=400, mimetype="application/json")

    input_text, snippet_lines, file_name, hunk_start = preprocess_input(diff_text, commit_msg, parent_commit)
    logging.info(f"Model Input: {input_text}")

    pred, conf, logits, probs = classify_change(input_text)
    label = "Buggy" if pred == 1 else "Not Buggy"

    result = {
        "label": label,
        "buggy_lines": snippet_lines if label == "Buggy" else [],
        "confidence": round(conf, 4),
        "probs": [round(p, 4) for p in probs.tolist()[0]],
        "logits": [round(l, 4) for l in logits.tolist()[0]],
        "file": file_name,
        "hunk_start": hunk_start
    }

    logging.info(f"Prediction: {label} | Confidence: {conf:.4f}")
    logging.info(f"Logits: {result['logits']} | Probs: {result['probs']}")

    return Response(json.dumps(result, indent=4), status=200, mimetype="application/json")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
