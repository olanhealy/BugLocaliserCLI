import argparse
import os
import torch
import torch.nn.functional as F
from git import Repo
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re
import sys

# ---------------------------
# STEP 1: LOAD MODEL & TOKENIZER
# ---------------------------
MODEL_DIR = "C:/Users/olan/Olan/[01] College/YR4/fyp cli/model_final_two"  # Absolute path of saved model
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# STEP 2: PREPROCESSING FUNCTION
# ---------------------------
def preprocess_diff(diff_text):
    """
    Convert raw diff text into the format the model was trained on.
    Uses default values for commit message and bug type.
    """
    commit_message = "No commit message available"
    bug_type = "UNKNOWN"
    parent_commit = ""
    input_text = (
        f"Context Before:\n<Diff Start>\n{diff_text}\n<Diff End>\n"
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
    - If commit_range is provided (e.g. 'HEAD~1..HEAD'), diffs those commits.
    - If file_path is provided, diffs only that file.
    - If neither is provided, it defaults to diffing staged changes.
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
# STEP 5: DIFF PARSING FUNCTION
# ---------------------------
def parse_diff_for_bug_lines(diff_text):
    """
    Parse the diff text to extract hunk headers and changed lines.
    Returns a list of dictionaries with hunk info, including the new starting line.
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
                    "lines": [line]  
                }
                hunks.append(current_hunk)
        elif current_hunk is not None:
            if line.startswith("+") or line.startswith("-"):
                current_hunk["lines"].append(line)
    return hunks
# ---------------------------
# STEP 6: MAIN CLI LOGIC
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Automatic Bug Localisation CLI Tool")
    parser.add_argument("--repo", type=str, default=".", help="Path to local git repo (default: current directory)")
    parser.add_argument("--range", type=str, help="Commit range to diff (e.g., 'HEAD~1..HEAD')")
    parser.add_argument("--file", type=str, help="Specific file to diff (optional)")
    args = parser.parse_args()
    
    diff_text = get_diff(args.repo, commit_range=args.range, file_path=args.file)
    if not diff_text.strip():
        print("No diff found. Please ensure you have staged changes or a valid commit range.")
        return
    
    preprocessed_text = preprocess_diff(diff_text)
    
    pred_class, confidence = classify_change(preprocessed_text)
    label = "Buggy" if pred_class == 1 else "Not Buggy"
    
    print("\n=== Bug Localisation Prediction ===")
    print(f"Label: {label} (Confidence: {confidence:.4f})")

    if label == "Buggy":
        hunks = parse_diff_for_bug_lines(diff_text)
        if hunks:
            print("\n=== Changed Hunks (with line numbers) ===")
            for hunk in hunks:
                print(f"\nBUGGY CODE DETECTED IN HUNK STARTING AT LINE {hunk['new_start']}:")
                print("\n".join(hunk["lines"]))
        else:
            print("\nNo hunk information could be parsed from the diff.")
    print("\n--- Preprocessed Input (truncated) ---")
    print(preprocessed_text[:500] + "...")
    print("--------------------------------------")

    if label == "Buggy":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
