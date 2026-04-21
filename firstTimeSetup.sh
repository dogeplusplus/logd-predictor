#!/bin/bash
set -xeuf -o pipefail

uv venv
uv sync
pre-commit install
