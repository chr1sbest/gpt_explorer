#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
arch -arm64 python3 gpt_token_explorer.py "$@"
