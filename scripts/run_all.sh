#!/usr/bin/env bash
set -euo pipefail

# cd to repo root (the directory where this script lives, then .. if placed in scripts/)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Activate conda env
# Assume the user has already activated the correct environment (conda or venv)
# eval "$(conda shell.bash hook)"
# conda activate quantum-dev

# Run
PYTHONPATH="$REPO_ROOT/src" python -m qmsl.cli run-all --seed 0 --out results

echo "Done. See docs/wechat_article.md and docs/figures/"
