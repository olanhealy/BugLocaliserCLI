#!/bin/sh

# This script is the one used to check for buggy lines in Java files staged for commit using a bug localisation model.
# It requires the buglocaliser Docker container to be running and the model to be loaded.
# The script reads the commit message and the staged files, and sends a request to the bug localisation API.
# It then parses the response and checks if any buggy lines were detected.
# If buggy lines are found, the commit is blocked; otherwise, it proceeds.
# Usage: ./commit-msg.sh <commit-message-file>, put in the .hooks directory of a git repository
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ "$SKIP_BUG_CHECK" = "1" ]; then
    echo "⚠️ Bug localisation check skipped due to SKIP_BUG_CHECK=1"
    exit 0
fi


if [ -z "$1" ]; then
    echo -e "${RED}No commit message file provided.${NC}"
    exit 1
fi

COMMIT_MSG=$(cat "$1" | head -c 500)
PARENT_COMMIT_MSG=$(git log -1 --pretty=%B | head -c 500)

STAGED_FILES=$(git diff --cached --name-only)
if ! echo "$STAGED_FILES" | grep -qE '\.java$'; then
    exit 0
fi

DIFF=$(git diff --staged | grep -vE '^\s*//' | grep -vE '^\s*$')
if [ -z "$DIFF" ]; then
    exit 0
fi

ESCAPED_COMMIT_MSG=$(printf '%s' "$COMMIT_MSG" | docker run --rm -i buglocaliser python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
ESCAPED_PARENT_COMMIT_MSG=$(printf '%s' "$PARENT_COMMIT_MSG" | docker run --rm -i buglocaliser python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
DIFF_JSON=$(printf '%s' "$DIFF" | docker run --rm -i buglocaliser python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

JSON_PAYLOAD=$(cat <<EOF
{
  "diff": $DIFF_JSON,
  "commit_msg": $ESCAPED_COMMIT_MSG,
  "parent_commit": $ESCAPED_PARENT_COMMIT_MSG
}
EOF
)

api_status=$(curl -s --head --request OPTIONS http://localhost:8080/predict | grep -E "200 OK|405 METHOD NOT ALLOWED")
if [ -z "$api_status" ]; then
    echo -e "${RED}Bug localisation API not running. Start the Docker container.${NC}"
    exit 1
fi

response=$(curl -s -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d "$JSON_PAYLOAD")
if [ -z "$response" ]; then
    echo -e "${RED}No response from API.${NC}"
    exit 1
fi

BUG_TMP=$(mktemp)
echo 0 > "$BUG_TMP"

# Parse predictions
echo "$response" | docker run --rm -i buglocaliser jq -c '.[]' | while read -r line; do
    label=$(echo "$line" | docker run --rm -i buglocaliser jq -r '.label')
    file=$(echo "$line" | docker run --rm -i buglocaliser jq -r '.file')
    hunk_start=$(echo "$line" | docker run --rm -i buglocaliser jq -r '.hunk_start')

    if [ "$label" = "Buggy" ]; then
        echo -e "${RED}⚠️  Bug localisation model flagged a buggy hunk.${NC}"
        echo -e "${YELLOW}🔍 Suspected bug in: ${file} (starting from line ${hunk_start})${NC}"
        echo 1 > "$BUG_TMP"
    fi
done

BUG_FOUND=$(cat "$BUG_TMP")
rm "$BUG_TMP"

if [ "$BUG_FOUND" = "1" ]; then
    echo -e "${RED}❌ Commit blocked due to detected bug(s).${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Bug localisation check passed. No buggy lines detected.${NC}"
    exit 0
fi
