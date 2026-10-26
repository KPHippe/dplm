"""Compare real checkpoint tensors after interrupted/uninterrupted GPU canaries."""
import argparse
import json
from pathlib import Path

import torch


def compare(a, b, path="state"):
    if torch.is_tensor(a):
        if not torch.equal(a, b):
            raise AssertionError(f"tensor mismatch at {path}; max error {(a-b).abs().max()}")
        return 1
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        return sum(compare(a[k], b[k], f"{path}.{k}") for k in a)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        return sum(compare(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b, strict=True)))
    assert a == b, path
    return 0


def floating_drift(a, b):
    """Report GPU numeric drift separately from exact schedule/RNG invariants."""
    if torch.is_tensor(a):
        assert a.shape == b.shape and a.dtype == b.dtype
        if not a.is_floating_point() or a.ndim == 0:
            compare(a, b)
            return (0, 0, 0., 0.)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        delta = (a-b).float()
        return (a.numel(), int(torch.count_nonzero(delta)),
                float(delta.abs().max()), float(delta.double().square().sum()))
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        parts = [floating_drift(a[k], b[k]) for k in a]
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        parts = [floating_drift(x, y) for x, y in zip(a, b, strict=True)]
    else:
        compare(a, b)
        return (0, 0, 0., 0.)
    return (sum(p[0] for p in parts), sum(p[1] for p in parts),
            max((p[2] for p in parts), default=0.), sum(p[3] for p in parts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("resumed", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    a = torch.load(args.reference, map_location="cpu", weights_only=False, mmap=True)
    b = torch.load(args.resumed, map_location="cpu", weights_only=False, mmap=True)
    keys = ("step", "exposure", "rng", "model_spec", "fast_provenance")
    checked = {k: compare(a[k], b[k], k) for k in keys}
    # Output directories necessarily differ; all scientific settings must agree.
    checked["contract"] = compare({k: v for k, v in a["contract"].items() if k != "out"},
                                  {k: v for k, v in b["contract"].items() if k != "out"})
    drift = {}
    for key in ("model", "optimizer"):
        n, changed, maximum, squared = floating_drift(a[key], b[key])
        drift[key] = dict(elements=n, different_elements=changed, max_absolute_error=maximum,
                          rms_error=(squared / max(n, 1))**0.5)
    report = dict(exact_schedule_rng_exposure=True, checked_tensor_counts=checked,
                  numeric_drift=drift, bitwise_match=all(d["different_elements"] == 0 for d in drift.values()), step=a["step"],
                  reference=str(args.reference), resumed=str(args.resumed))
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
