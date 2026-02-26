#!/usr/bin/env bash
set -euo pipefail

DEST_REMOTE="https://github.com/BlazOrsos/crypto-shrev.git"

# temp folder
TMP_DIR="$(mktemp -d)"
echo "Temp dir: $TMP_DIR"

echo "Cloning quant-journey into temp dir (main branch)..."
git clone --no-local --branch main --single-branch . "$TMP_DIR"
cd "$TMP_DIR"

echo "Verifying paths exist (tracked files)..."
git ls-files --error-unmatch config/shock_reversion.json >/dev/null
git ls-files --error-unmatch src/utils/logger.py >/dev/null
git ls-files --error-unmatch src/pipelines/shock_reversion_pipeline.py >/dev/null
git ls-files --error-unmatch src/exchanges/binance_klines.py >/dev/null
git ls-files --error-unmatch src/exchanges/binance_websocket.py >/dev/null
git ls-files --error-unmatch src/strategies/shock_reversion.py >/dev/null
git ls-files --error-unmatch src/data/storage.py >/dev/null

echo "Filtering repo..."
git filter-repo \
  --path config/shock_reversion.json \
  --path src/utils/logger.py \
  --path src/pipelines/shock_reversion_pipeline.py \
  --path src/exchanges/binance_klines.py \
  --path src/exchanges/binance_websocket.py \
  --path src/strategies/shock_reversion.py \
  --path src/data/storage.py \
  --force

# Make sure filter didn't produce an empty repo
git rev-parse --verify HEAD >/dev/null

echo "Pushing to crypto-shrev..."
git remote add dest "$DEST_REMOTE"
# Push HEAD to main even if local branch name differs/detached
git push dest HEAD:refs/heads/main --force

echo "Done."