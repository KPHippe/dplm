"""Run released DPLM-2 hard-token round trips on frozen paired native inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from native_tokenizer_loader import load_native_tokenizer


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--verify-roundtrip', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, provenance = load_native_tokenizer()
    model = model.cuda().eval().requires_grad_(False)
    inputs = json.loads((args.inputs / 'manifest.json').read_text())
    selected = inputs['rows'][:args.limit] if args.limit else inputs['rows']
    rows = []
    with torch.no_grad():
        for item in selected:
            start = time.monotonic()
            number = item['number']
            path = args.inputs / f'{number:04d}.npz'
            assert sha256(path) == item['npz_sha256']
            with np.load(path) as data:
                xyz = torch.as_tensor(data['native_xyz'], device='cuda')[None]
                assert data['native_mask'][:, :3].all()
                assert len(np.unique(data['chain_id'])) == 1
                assert (np.diff(data['residue_id']) == 1).all()
            length = xyz.shape[1]
            mask = torch.ones((1, length), device='cuda')
            tokens = model.tokenize(xyz, mask, seq_length=torch.tensor([length], device='cuda'))
            assert tokens.shape == (1, length)
            assert tokens.min() >= 0 and tokens.max() < 8192
            decoded = model.detokenize(tokens, res_mask=mask)
            verification = {}
            if args.verify_roundtrip:
                pre, _ = model.encode(xyz, mask, seq_length=torch.tensor([length], device='cuda'))
                quant, _, (_, _, direct_tokens) = model.quantize(pre, mask=mask.bool())
                looked_up = model.quantize.get_codebook_entry(tokens)
                assert torch.equal(direct_tokens, tokens)
                assert torch.allclose(quant, looked_up, atol=1e-6, rtol=0)
                direct = model.decode(looked_up, torch.zeros_like(tokens), mask)
                assert torch.equal(direct['final_atom_positions'], decoded['atom37_positions'])
                # Independent proper rotation and translation must preserve GVP features/tokens.
                matrix = torch.tensor([[.3, -.7, .1], [.4, .2, .8], [.9, .5, -.3]], device='cuda')
                rotation, _ = torch.linalg.qr(matrix)
                rotation[:, 0] *= torch.linalg.det(rotation)
                moved = xyz @ rotation + 7
                moved_pre, _ = model.encode(moved, mask, seq_length=torch.tensor([length], device='cuda'))
                moved_tokens = model.tokenize(moved, mask, seq_length=torch.tensor([length], device='cuda'))
                verification = dict(codebook_max_abs_error=float((quant-looked_up).abs().max()),
                    direct_decode_exact=True, rigid_pre_quant_max_abs_error=float((pre-moved_pre).abs().max()),
                    rigid_token_agreement=float((tokens==moved_tokens).float().mean()))
            positions = decoded['atom37_positions'][0]
            atom_mask = decoded['atom37_mask'][0].bool()
            assert torch.isfinite(positions[atom_mask]).all()
            assert atom_mask[:, [0, 1, 2, 4]].all()
            # The official PDB writer adjusts O from neighboring backbone atoms.
            # Save this separately: raw model output is the primary comparison.
            from native_tokenizer_loader import adjust_oxygen_pos
            adjusted = adjust_oxygen_pos(positions.cpu().clone(), atom_mask[:, 0].cpu())
            assert torch.equal(adjusted[:, :4], positions.cpu()[:, :4])
            outfile = args.out / f'{number:04d}.npz'
            np.savez_compressed(outfile, xyz=positions.cpu().numpy(), mask=atom_mask.cpu().numpy(),
                tokens=tokens[0].cpu().numpy(), oxygen_adjusted_xyz=adjusted.numpy())
            rows.append(dict(number=number, source=item['source'], entry_id=item['entry_id'],
                npz_sha256=sha256(outfile), seconds=time.monotonic()-start, verification=verification))
            print(f'Round trip {len(rows)}/{len(selected)} length={length} seconds={rows[-1]["seconds"]:.2f}', flush=True)
    manifest = dict(input_manifest_sha256=sha256(args.inputs / 'manifest.json'),
        provenance=provenance, precision='fp32; no autocast; TF32 disabled',
        decoder_recycles=int(model.decoder.cfg.trunk.max_recycles), gpu=torch.cuda.get_device_name(),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(), script_sha256=sha256(__file__), rows=rows)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
