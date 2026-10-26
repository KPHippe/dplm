"""Versioned descriptive sequence/fold diagnostics; these are not readiness gates."""
import hashlib

import numpy as np


def sequence_metrics(aa):
    aa = np.asarray(aa)
    if aa.ndim != 1 or not len(aa) or not np.issubdtype(aa.dtype, np.integer) or ((aa < 0) | (aa >= 20)).any():
        raise ValueError('nonempty canonical sequence required')
    count = np.bincount(aa.astype(int), minlength=20)
    p = count[count > 0] / len(aa)
    raw = aa.astype(np.uint8).tobytes()
    words = [raw[i:i+4] for i in range(max(len(aa)-3, 0))]
    longest = run = 1
    for before, after in zip(aa[:-1], aa[1:]):
        run = run+1 if before == after else 1
        longest = max(longest, run)
    return dict(length=len(aa), sequence_sha256=hashlib.sha256(raw).hexdigest(),
        aa_entropy_bits=float(-np.sum(p*np.log2(p))), max_aa_fraction=float(p.max()),
        repeat4_fraction=1-len(set(words))/len(words) if words else None,
        longest_identical_run=longest,
        max_lag_identity_1_40=max((float(np.mean(aa[k:] == aa[:-k]))
            for k in range(1, min(40, len(aa)//2)+1)), default=None))


def sequence_panel(rows, requested):
    if requested < len(rows) or requested < 1:
        raise ValueError('invalid requested denominator')
    ok = [r for r in rows if r.get('status') == 'ok']
    groups = {}
    for row in ok:
        groups.setdefault(row['length'], []).append(row)
    return dict(requested=requested, scored=len(ok), failed_or_missing=requested-len(ok),
        unique_sequences=len({r['sequence_sha256'] for r in ok}),
        unique_fraction_scored=len({r['sequence_sha256'] for r in ok})/len(ok) if ok else None,
        repeat4_over_half=sum((r['repeat4_fraction'] or 0) > .5 for r in ok),
        repeat4_flag_is_descriptive=True,
        by_length={str(n): dict(scored=len(rs), unique_sequences=len({r['sequence_sha256'] for r in rs}),
            mean_repeat4_fraction=float(np.mean([r['repeat4_fraction'] for r in rs
                if r['repeat4_fraction'] is not None])) if n >= 4 else None)
            for n, rs in groups.items()})


def ca_geometry(mobile, reference):
    """Proper-rotation RMSD and CA-only distance lDDT (<15 A, i<j, includes neighbors)."""
    mobile, reference = np.asarray(mobile, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if (mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3
            or len(mobile) < 3 or not np.isfinite(mobile).all() or not np.isfinite(reference).all()):
        raise ValueError('finite complete matched CA coordinates required')
    x, y = mobile-mobile.mean(0), reference-reference.mean(0)
    u, _, vh = np.linalg.svd(x.T@y)
    correction = np.diag([1., 1., np.linalg.det(u@vh)])
    rmsd = float(np.sqrt(np.mean(np.sum((x@(u@correction@vh)-y)**2, axis=1))))
    distances = np.linalg.norm(mobile[:, None]-mobile[None], axis=-1)
    target = np.linalg.norm(reference[:, None]-reference[None], axis=-1)
    pairs = np.triu(target < 15., 1)
    error = np.abs(distances[pairs]-target[pairs])
    return dict(ca_rmsd_a=rmsd,
        ca_lddt=float(np.mean([np.mean(error < t) for t in (.5, 1., 2., 4.)])) if len(error) else None,
        ca_lddt_pairs=int(len(error)), ca_lddt_definition='CA pairs i<j, reference distance<15A, includes adjacent residues; thresholds0.5/1/2/4A',
        backbone_below_2a=rmsd < 2.)


def summarize_scoring(rows, requested, fold_requested=True):
    if len(rows) != requested or len({r['id'] for r in rows}) != requested:
        raise ValueError('one explicit result per requested sample required')
    ok = [r for r in rows if r['status'] == 'ok']
    fold = [r for r in ok if r.get('ca_rmsd_a') is not None]
    identities = sum(r.get('exact_identity', False) for r in ok)
    backbone = sum(r.get('backbone_below_2a', False) for r in fold)
    combined = sum(r.get('exact_identity', False) and r.get('backbone_below_2a', False) for r in fold)
    def rate(n):
        return dict(count=n, requested_denominator=requested, fraction=n/requested if requested else None)
    atoms = sum(r['chemistry']['emitted_atoms'] for r in ok)
    bonds = sum(r['chemistry']['peptide_bonds'] for r in ok)
    return dict(requested=requested, scored=len(ok), failed=requested-len(ok), fold_scored=len(fold),
        fold_status='scored_or_failed' if fold_requested else 'not_requested',
        exact_identity=rate(identities), backbone_below_2a=rate(backbone) if fold_requested else None,
        identity_and_backbone=rate(combined) if fold_requested else None,
        median_ca_rmsd_a=float(np.median([r['ca_rmsd_a'] for r in fold])) if fold else None,
        mean_ca_lddt=float(np.mean([r['ca_lddt'] for r in fold if r['ca_lddt'] is not None]))
            if any(r['ca_lddt'] is not None for r in fold) else None,
        peptide_bonds_scored=bonds, emitted_atoms_scored=atoms,
        pooled_peptide_violation_fraction=sum(r['chemistry']['peptide_outside_1p1_1p6_a'] for r in ok)/bonds if bonds else None,
        pooled_severe_clashes_per_1000_atoms=1000*sum(r['chemistry']['severe_clash_pairs'] for r in ok)/atoms if atoms else None,
        chemistry_gate=None, full_atom_readiness=None,
        caveat='Identity/backbone successes exclude chemistry; raw chemistry is diagnostic, no new pass thresholds.')
