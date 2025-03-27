import argparse
import subprocess
import os
import torch
import torch.nn.functional as F
from git import Repo
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ---------------------------
# STEP 1: LOAD MODEL & TOKENIZER
# ---------------------------
MODEL_DIR = "model_final"  
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# STEP 2: PREPROCESSING FUNCTION
# ---------------------------
def preprocess_diff(diff_text, commit_message="", parent_commit="", bug_type=""):
    """
    Convert raw diff text into the approximate format your model was trained on.
    Here, we do a simplistic approach by using the entire diff as 'buggy code'.
    """

    input_text = (
        f"Context Before:\n<Diff Start>\n"
        f"{diff_text}\n"
        f"<Diff End>\n"
        f"Buggy Code: {diff_text[:200]}... \n"  
        f"Commit Message: {commit_message}\n"
        f"Parent Commit: {parent_commit}\n"
        f"Bug Type: {bug_type}"
    )
    return input_text

def tokenize_for_model(text):
    encoded = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}

# ---------------------------
# STEP 3: INFERENCE FUNCTION
# ---------------------------
def classify_change(preprocessed_text):
    inputs = tokenize_for_model(preprocessed_text)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item()
    return pred_class, confidence

# ---------------------------
# STEP 4: GIT DIFF EXTRACTION
# ---------------------------
def get_diff(repo_path, commit_range=None, file_path=None):
    """
    Extracts diff text using GitPython.
    - If commit_range is provided (e.g. 'HEAD~1..HEAD'), we diff those commits.
    - If file_path is provided, we diff only that file.
    - If neither is provided, we diff staged changes.
    """
    repo = Repo(repo_path)
    
    if commit_range:
        if file_path:
            diff_text = repo.git.diff(commit_range, "--", file_path)
        else:
            diff_text = repo.git.diff(commit_range)
    else:
        if file_path:
            diff_text = repo.git.diff("--staged", "--", file_path)
        else:
            diff_text = repo.git.diff("--staged")
    return diff_text
# ---------------------------
# STEP 5: MAIN CLI LOGIC
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="FYP Bug Localisation CLI Tool")
    parser.add_argument("--repo", type=str, default=".", help="Path to local git repo (default: current dir)")
    parser.add_argument("--range", type=str, help="Commit range to diff, e.g. 'HEAD~1..HEAD'")
    parser.add_argument("--file", type=str, help="Specific file to diff")
    parser.add_argument("--commit_msg", type=str, default="", help="Simulated commit message")
    parser.add_argument("--parent_msg", type=str, default="", help="Parent commit message (if any)")
    parser.add_argument("--bug_type", type=str, default="UNKNOWN", help="Optional bug type label")
    args = parser.parse_args()
    
    diff_text = get_diff(args.repo, commit_range=args.range, file_path=args.file)
    if not diff_text.strip():
        print("No diff found. Please ensure you have staged changes or a valid commit range.")
        return
    
    preprocessed_text = preprocess_diff(
        diff_text,
        commit_message=args.commit_msg,
        parent_commit=args.parent_msg,
        bug_type=args.bug_type
    )
    
    pred_class, confidence = classify_change(preprocessed_text)
    label = "Buggy" if pred_class == 1 else "Not Buggy"
    print("\n=== Bug Localisation Prediction ===")
    print(f"Label: {label} (Confidence: {confidence:.4f})")
    print("\n--- Preprocessed Input (truncated) ---")
    print(preprocessed_text[:500] + "...")
    print("--------------------------------------")

if __name__ == "__main__":
    main()
