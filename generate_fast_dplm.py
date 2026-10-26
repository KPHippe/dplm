"""Generate from a strictly loaded FAST-DPLM checkpoint; no coordinates needed."""
import argparse
from pathlib import Path

import torch

from fast_dplm.data import sha256
from fast_dplm.recipe import model_from_spec
from fast_dplm.sampling import generate, write_samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--count", type=int, default=2)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=919)
    a = p.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_spec(ck["model_spec"])
    model.load_state_dict(ck["model"], strict=True)
    model.cuda().eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        samples = generate(model, a.length, a.count, a.steps, a.seed)
    write_samples(a.out, samples, dict(fingerprint=ck["fast_provenance"]["fingerprint"], levels=model.levels,
        decoder_provenance=ck["fast_provenance"],
        model_sha256=sha256(a.checkpoint), model_step=ck["step"], seed=a.seed, sampling_steps=a.steps,
        generator_sha256=sha256(__file__),
        adapter_sha256={f.name: sha256(f) for f in
                       sorted((Path(__file__).parent / "src/fast_dplm").glob("*.py"))}))


if __name__ == "__main__":
    main()
