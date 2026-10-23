#!/usr/bin/env bash
# Run ONCE from the repo root, BEFORE training. Removes the old synthetic-data
# artifacts (they remain in git history) so stale results can't be re-committed.
set -euo pipefail

git rm -r -f --ignore-unmatch \
    data/synthetic_transactions.csv \
    src/mule_fraud_detection.py \
    outputs
rm -rf legacy/ src/__pycache__ outputs
mkdir -p outputs

git add .gitignore requirements.txt README.md data/README.md data/.gitkeep \
        src/ tests/ scripts/

big=$(git diff --cached --name-only | while read -r f; do
        [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt $((50*1024*1024)) ] && echo "$f"
      done || true)
if [ -n "$big" ]; then echo "Aborting, large file staged: $big" >&2; exit 1; fi
echo "OK. Next: python src/train.py, then: git add outputs && git commit && git push"
