#!/bin/bash
# A separate dependency overlay; never installs packages into the FAST runtime.
set -euo pipefail
DPLM_ROOT=$(cd "$(dirname "$0")/.." && pwd)
FAST_ROOT=${FAST_ROOT:-/hpc/mydata/kyle.hippe/github/fast-rewrite}
export DPLM_ROOT FAST_ROOT
if [[ ! -f "$DPLM_ROOT/.venv/pyvenv.cfg" ]]; then
    "$FAST_ROOT/.venv/bin/python" -m venv --without-pip "$DPLM_ROOT/.venv"
fi
"$DPLM_ROOT/.venv/bin/python" - <<'PY'
import os
import subprocess
import sysconfig
from pathlib import Path
fast = Path(os.environ['FAST_ROOT'])
base = subprocess.check_output([str(fast / '.venv/bin/python'), '-c',
    'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
Path(sysconfig.get_paths()['purelib'], 'fast_runtime.pth').write_text(base + '\n')
PY
uv pip install --python "$DPLM_ROOT/.venv/bin/python" --no-deps -r "$DPLM_ROOT/requirements-fast.txt"
"$DPLM_ROOT/.venv/bin/python" -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'
