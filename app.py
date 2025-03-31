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
# LOAD MODEL & TOKENIZER
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
def split_hunks(diff_text):
    hunks = []
    current_hunk = []
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = [line]
        elif current_hunk:
            current_hunk.append(line)
    if current_hunk:
        hunks.append(current_hunk)
    return ["\n".join(hunk) for hunk in hunks]

def extract_code_context(hunk):
    lines = hunk.split("\n")
    context_lines = []
    snippet_lines_added = []
    snippet_lines_removed = []
    found_change = False
    hunk_start_line = None
    file_name = "UnknownFile.java"

    for i, line in enumerate(lines):
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            if match:
                hunk_start_line = int(match.group(1))
            continue
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

def preprocess_input(context, snippet, commit_msg, parent_commit):
    context_masked = mask_identifiers(context)
    snippet_masked = mask_identifiers(snippet)
    return (
        f"[CONTEXT] {context_masked}\n"
        f"[SNIPPET] {snippet_masked}\n"
        f"[COMMIT] {commit_msg}\n"
        f"[PARENT] {parent_commit}"
    )

def tokenize_for_model(text):
    encoded = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}

def classify_change(input_text):
    inputs = tokenize_for_model(input_text)
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
    commit_msg = data.get("commit_msg", "No commit message").strip()
    parent_commit = data.get("parent_commit", "").strip()

    if not diff_text:
        return Response(json.dumps({"error": "No diff provided"}), status=400, mimetype="application/json")

    predictions = []
    for hunk in split_hunks(diff_text):
        context, snippet, snippet_lines, file_name, hunk_start = extract_code_context(hunk)
        input_text = preprocess_input(context, snippet, commit_msg, parent_commit)
        logging.info(f"Model Input: {input_text}")

        pred, conf, logits_tensor, probs_tensor = classify_change(input_text)
        label = "Buggy" if pred == 1 else "Not Buggy"

        result = {
            "label": label,
            "buggy_lines": snippet_lines if label == "Buggy" else [],
            "confidence": round(float(conf), 4),
            "probs": [round(float(x), 4) for x in probs_tensor.squeeze().tolist()],
            "logits": [round(float(x), 4) for x in logits_tensor.squeeze().tolist()],
            "file": file_name,
            "hunk_start": hunk_start
        }

        logging.info(f"Prediction: {label} | Confidence: {conf:.4f}")
        logging.info(f"Logits: {result['logits']} | Probs: {result['probs']}")
        predictions.append(result)

    return Response(json.dumps(predictions, indent=4), status=200, mimetype="application/json")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)