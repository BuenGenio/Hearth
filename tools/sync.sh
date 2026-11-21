#!/bin/sh
# Copy the source tree to a host for building there:  sh tools/sync.sh server:hearth/src/
set -eu
cd "$(dirname "$0")/.."
rsync -a --delete --exclude /.venv --exclude /.venv-ha --exclude /models --exclude /data --exclude /.git \
    --exclude __pycache__ --exclude '*.egg-info' --exclude /.pytest_cache ./ "$1"
