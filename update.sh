#!/usr/bin/env bash

set -e

# Change our working dir to the root of the project.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Pull the latest changes via Git. A cloud machine that runs the commit of the PC that started it checks that commit
# out itself and sets OT_UPDATE_SKIP_GIT_PULL=true (a pull would also fail on its detached HEAD).
if [[ "${OT_UPDATE_SKIP_GIT_PULL:-false}" != "true" ]]; then
    echo "[OneTrainer] Updating OneTrainer to latest version from Git repository..."
    git pull
fi

# Load the newest version of the function library.
source "lib.include.sh"

# Prepare runtime and upgrade all dependencies to latest compatible version.
prepare_runtime_environment upgrade
