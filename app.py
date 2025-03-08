from flask import Flask, request, Response
import json
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True  

# 
MODEL_DIR = "/app/model_final_two"
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2, local_files_only=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_diff(diff_text):
    commit_message = "No commit message available"
    bug_type = "UNKNOWN"
    parent_commit = ""
    return (
        f"Context Before:\n<Diff Start>\n{diff_text}\n<Diff End>\n"
        f"Buggy Code: {diff_text[:200]}... \n"
        f"Commit Message: {commit_message}\n"
        f"Parent Commit: {parent_commit}\n"
        f"Bug Type: {bug_type}"
    )

def tokenize_for_model(text):
    encoded = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}

def classify_change(preprocessed_text):
    inputs = tokenize_for_model(preprocessed_text)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item()
    return pred_class, confidence

def parse_diff_for_bug_lines(diff_text):
    """
    Parse the diff text to extract hunk headers and changed lines.
    Returns a list of hunks, each with its starting line and the hunk text.
    """
    hunks = []
    current_hunk = None
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            m = re.match(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", line)
            if m:
                new_start = int(m.group(3))
                current_hunk = {
                    "new_start": new_start,
                    "lines": [line]  # 
                }
                hunks.append(current_hunk)
        elif current_hunk is not None:
            if line.startswith("+") or line.startswith("-"):
                current_hunk["lines"].append(line)
    return hunks

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'diff' not in data:
        return Response("No diff provided", status=400, mimetype="text/plain")
    diff_text = data["diff"]
    preprocessed = preprocess_diff(diff_text)
    pred, conf = classify_change(preprocessed)
    label = "Buggy" if pred == 1 else "Not Buggy"
    
    # Build the CLI-like output string
    cli_output = "=== Bug Localisation Prediction ===\n"
    cli_output += f"Label: {label} (Confidence: {conf:.4f})\n\n"
    if label == "Buggy":
        hunks = parse_diff_for_bug_lines(diff_text)
        if hunks:
            cli_output += "--- Changed Hunks (with line numbers) ---\n"
            for hunk in hunks:
                cli_output += f"\nBUGGY CODE DETECTED IN HUNK STARTING AT LINE {hunk['new_start']}:\n"
                cli_output += "\n".join(hunk["lines"]) + "\n"
        else:
            cli_output += "\nNo hunk information could be parsed from the diff.\n"
    cli_output += "\n--- Preprocessed Input (truncated) ---\n"
    cli_output += preprocessed[:500] + "...\n"
    cli_output += "--------------------------------------\n"
    
    return Response(cli_output, status=200, mimetype="text/plain")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
