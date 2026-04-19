#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/third_party"

mkdir -p "$THIRD_PARTY_DIR"

clone_if_missing() {
    local repo_url="$1"
    local branch="$2"
    local target_dir="$3"

    if [ -d "$target_dir/.git" ]; then
        echo "Using existing checkout: $target_dir"
        return
    fi

    echo "Cloning $repo_url -> $target_dir"
    git clone --branch "$branch" --depth 1 "$repo_url" "$target_dir"
}

clone_if_missing "https://github.com/PufferAI/PufferTank.git" "4.0" "$THIRD_PARTY_DIR/puffertank"
clone_if_missing "https://github.com/PufferAI/PufferLib.git" "4.0" "$THIRD_PARTY_DIR/pufferlib"

echo
echo "Workspace ready."
echo "Open the repo in the devcontainer or use the local checkouts under third_party/."

