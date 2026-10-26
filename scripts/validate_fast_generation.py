"""Inference-only generation validation on an explicit checkpoint and new frozen panel."""
import argparse
import json
from pathlib import Path

import torch

from fast_dplm.data import TokenCache, sha256
from fast_dplm.evaluation import evaluate_panel
from fast_dplm.recipe import model_from_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--val-cache', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    panel = json.loads(args.panel.read_text())
    if 'generation_validation' not in panel:
        raise ValueError('requires explicit generation-validation panel')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = model_from_spec(ck['model_spec'])
    model.load_state_dict(ck['model'], strict=True)
    model.cuda().eval()
    provenance = ck['fast_provenance'] | dict(checkpoint_sha256=sha256(args.checkpoint),
        panel_file_sha256=sha256(args.panel), checkpoint_step=ck['step'])
    evaluate_panel(model, TokenCache(args.val_cache), panel, args.out, ck['step'], provenance)


if __name__ == '__main__':
    main()
