#!/usr/bin/env bash
set -euo pipefail
REPO_URL="${1:-https://github.com/budhasantosh010/transmutor.git}"

if [ ! -d .git ]; then
  git init -b main
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

git add .
if ! git diff --cached --quiet; then
  git commit -m "Import preserved Transmutor research history"
fi
git push -u origin main
