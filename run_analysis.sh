#!/usr/bin/env bash
set -euo pipefail

python analysis/compute_bias.py
python analysis/run_analysis.py
