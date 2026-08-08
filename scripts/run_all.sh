#!/usr/bin/env bash
# reproduce everything from a fresh clone; stages get uncommented as they land
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/check_env.py

# day 1: data + indexes
# python -m data.download
# python -m data.prepare_hotpotqa
# python -m data.prepare_2wiki

# day 2: single-hop baseline + gold harness + 2x2
# day 3: multi-hop + reranker ablation
# day 4: serving sweep -> results/pareto.png
# day 6: decoupling plot + ablation table
