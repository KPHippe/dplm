"""Exercise a frozen validation/generation panel without optimizer updates."""
import argparse
import json
from pathlib import Path

from fast_dplm.data import TokenCache, sha256
from fast_dplm.evaluation import evaluate_panel
from fast_dplm.recipe import model_from_spec
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'checkpoint', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads((a.bundle/'config.json').read_text())
    panel = json.loads((a.bundle/'panel.json').read_text())
    cache = TokenCache(cfg['val_cache'])
    if panel['cache_digest'] != cache.digest:
        raise ValueError('panel/cache mismatch')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = model_from_spec(ck['model_spec'])
    model.load_state_dict(ck['model'],strict=True)
    model.cuda().eval()
    provenance = ck['fast_provenance'] | dict(training_contract=ck['contract'] | dict(
        evaluation_checkpoint_sha256=sha256(a.checkpoint), evaluation_script_sha256=sha256(__file__)))
    evaluate_panel(model,cache,panel,a.out,ck['step'],provenance)
    print(f'Completed {len(panel["entries"])} validation entries and {len(panel["samples"])} generated samples',flush=True)


if __name__ == '__main__':
    main()
