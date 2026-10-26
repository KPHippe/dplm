"""Shared DDP training loop with frozen production settings and bounded canaries."""
import argparse
import contextlib
import fcntl
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP

from fast_dplm.data import TokenCache, collate, sha256
from fast_dplm.model import FastDPLM, loss_sums, noise_plan
from fast_dplm.evaluation import evaluate_panel, make_panel
from fast_dplm.recipe import DPLM2Recipe, model_from_spec
from fast_dplm.provenance import decoder_provenance
from fast_dplm.objectives import loss_plan, released_mixup_input


def save_atomic(path, state):
    tmp = path.with_suffix(".partial")
    torch.save(state, tmp)
    os.replace(tmp, path)


def learning_rate(step, peak, warmup, horizon, warmup_init=0.):
    update = step + 1
    if not 1 <= update <= horizon or not 0 <= warmup < horizon:
        raise ValueError('invalid schedule address')
    if update <= warmup:
        return warmup_init + (peak-warmup_init)*update/warmup
    return peak * (1 - .9 * (update-warmup)/(horizon-warmup))


def repair_log(path, committed_step):
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    retained = [r for r in rows if r['step'] <= committed_step]
    if len(retained) != len(rows):
        recovery = path.parent / 'recovery'
        recovery.mkdir(exist_ok=True)
        (recovery / f'uncommitted_{time.time_ns()}.jsonl').write_text(
            ''.join(json.dumps(r)+'\n' for r in rows if r['step'] > committed_step))
        tmp = path.with_suffix('.partial')
        tmp.write_text(''.join(json.dumps(r)+'\n' for r in retained))
        os.replace(tmp, path)


def main(debug=True):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["index", "components"], default="components")
    p.add_argument("--architecture", choices=["fast_bridge", "dplm2_recipe"], default="fast_bridge")
    p.add_argument("--decoder-lineage", type=Path)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--warmup-init-lr", type=float, default=0.)
    p.add_argument("--schedule-steps", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--sample-length", type=int, default=100)
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument('--run-config', type=Path)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--validate-every', type=int, default=1000)
    p.add_argument('--milestone-every', type=int, default=1000)
    p.add_argument('--eval-panel', type=Path)
    p.add_argument('--self-mixup', choices=['off', 'released'], default='off')
    p.add_argument('--noise-weighting', choices=['linear', 'constant_mass'], default='linear')
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--run-config', type=Path)
    cfg_arg, _ = probe.parse_known_args()
    settings = None
    if cfg_arg.run_config:
        settings = json.loads(cfg_arg.run_config.read_text())
        allowed = {a.dest for a in p._actions} - {'help', 'run_config', 'resume', 'stop_after'}
        if set(settings) - allowed:
            raise ValueError(f'unknown run settings: {set(settings)-allowed}')
        p.set_defaults(**settings)
        for action in p._actions:
            if action.dest in settings:
                action.required = False
    args = p.parse_args()
    if not debug and settings is None:
        raise ValueError('production requires a frozen --run-config')
    if settings is not None:
        for key, value in settings.items():
            if str(getattr(args, key)) != str(value):
                raise ValueError(f'CLI changes frozen setting: {key}')
    endpoint = args.steps if args.stop_after is None else min(args.steps, args.stop_after)
    if args.steps < 1 or endpoint < 1 or (debug and endpoint > 500):
        raise ValueError('invalid endpoint or debug run exceeds 500 updates')
    if args.stop_after is not None and args.stop_after > 500:
        raise ValueError('canary stop-after must be at most 500 updates')
    if min(args.accumulate, args.batch_size, args.checkpoint_every, args.validate_every, args.milestone_every) < 1:
        raise ValueError('batch and interval settings must be positive')
    if not 0 <= args.warmup < args.schedule_steps or args.steps > args.schedule_steps:
        raise ValueError("invalid immutable schedule")
    if args.architecture != 'dplm2_recipe' and (args.self_mixup != 'off' or args.noise_weighting != 'linear'):
        raise ValueError('objective ablations require the DPLM-2 recipe architecture')
    rank, world, local = (int(os.environ.get(k, d)) for k, d in
                          (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = None
    if rank == 0:
        lock = open(args.out / ".writer.lock", "w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume and (args.out / "run.json").exists():
            raise FileExistsError("run exists; choose a fresh output or explicit --resume")
    if world > 1:
        dist.barrier()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cache, val = TokenCache(args.cache), TokenCache(args.val_cache)
    if cache.manifest["fingerprint"] != val.manifest["fingerprint"]:
        raise ValueError("train and validation FAST fingerprints differ")
    if any(e["split"] != "train" or e["cluster"] is None for e in cache.entries):
        raise ValueError("training cache must contain train entries with cluster IDs")
    if any(e["split"] != "val" for e in val.entries):
        raise ValueError("validation cache must be val only")
    if {(e["source"], e["entry_id"]) for e in cache.entries} & {(e["source"], e["entry_id"]) for e in val.entries}:
        raise ValueError("train/validation entry overlap")
    contract = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("resume", "stop_after", "run_config") and not (debug and k == 'steps')}
    contract.update(world_size=world, cache_digest=cache.digest, val_digest=val.digest)
    fast_provenance = decoder_provenance(cache.manifest, args.decoder_lineage)
    contract["decoder_provenance"] = fast_provenance
    contract['runtime'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                               numpy=np.__version__, transformers=transformers.__version__)
    panel = json.loads(args.eval_panel.read_text()) if args.eval_panel else make_panel(val, per_source=2,
        lengths=[args.sample_length], per_length=2, sampling_steps=args.sample_steps)
    if panel['cache_digest'] != val.digest:
        raise ValueError('evaluation panel belongs to another validation cache')
    contract['evaluation_panel'] = panel
    root = Path(__file__).resolve().parents[2]
    code_files = [*sorted((root / "src/fast_dplm").glob("*.py"))]
    contract["code_sha256"] = {str(f.relative_to(root)): sha256(f) for f in code_files}
    # Rank zero hashes the immutable downloaded weights once, avoiding NFS fanout.
    pretrained_hashes = [{f: sha256(args.pretrained / f) for f in
                         ("config.json", "vocab.txt", "pytorch_model.bin")} if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(pretrained_hashes, src=0)
    contract["pretrained_sha256"] = pretrained_hashes[0]
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.out / "last.ckpt", map_location="cpu", weights_only=False)
        if checkpoint["contract"] != contract:
            raise ValueError("resume changes the dataset/model/schedule/DDP contract")
        model = model_from_spec(checkpoint["model_spec"])
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model_class = DPLM2Recipe if args.architecture == "dplm2_recipe" else FastDPLM
        model = model_class.from_dplm(args.pretrained, mode=args.mode, levels=cache.manifest["levels"])
    if args.gradient_checkpointing:
        model.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    start, exposure = 0, {}
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, exposure = checkpoint["step"], checkpoint["exposure"]
        if start > endpoint:
            raise ValueError("resume endpoint precedes saved update")
        torch.set_rng_state(checkpoint["rng"][rank]["cpu"])
        torch.cuda.set_rng_state(checkpoint["rng"][rank]["cuda"], device)
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    if checkpoint is None:
        torch.manual_seed(args.seed + rank)
    if rank == 0:
        if checkpoint:
            repair_log(args.out / 'metrics.jsonl', start)
        (args.out / "run.json").write_text(json.dumps(dict(contract=contract,
            sampling_profile=cache.profile, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)), indent=2))
    global_batch = world * args.batch_size * args.accumulate
    def evaluate(step, sampling):
        if rank == 0:
            evaluate_panel(model, val, panel, args.out / 'evaluation' / f'step_{step:07d}',
                           step, fast_provenance | dict(training_contract=contract), sampling=sampling)
        if world > 1:
            dist.barrier()

    if checkpoint is None:
        rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device))
        all_rng = [None] * world
        if world > 1:
            dist.all_gather_object(all_rng, rng)
        else:
            all_rng = [rng]
        if rank == 0:
            save_atomic(args.out / 'last.ckpt', dict(model=model.state_dict(), model_spec=model.specification(),
                optimizer=optimizer.state_dict(), step=0, exposure=exposure, rng=all_rng,
                contract=contract, fast_provenance=fast_provenance))
    if rank == 0 and (start % args.milestone_every == 0 or start == args.steps):
        weights = args.out / 'milestones' / f'step_{start:07d}.pt'
        weights.parent.mkdir(exist_ok=True)
        if not weights.exists():
            save_atomic(weights, dict(model=model.state_dict(), model_spec=model.specification(),
                step=start, fast_provenance=fast_provenance, contract=contract))
    if start == 0:
        evaluate(0, sampling=True)
    elif start % args.validate_every == 0 or start % args.milestone_every == 0 or start == endpoint:
        evaluate(start, sampling=start % args.milestone_every == 0 or start == args.steps)
    started = time.monotonic()
    for step in range(start, endpoint):
        update_started = time.monotonic()
        model.train()
        ids = cache.draw_update(step, global_batch, 1234)
        batch_cpu = collate([cache[i] for i in ids], model.aa_map, model.special)
        generator = torch.Generator().manual_seed(args.seed * 1000003 + step)
        plan_cpu = noise_plan(batch_cpu, generator, balanced=args.architecture == "dplm2_recipe")
        loss_cpu = loss_plan(batch_cpu, plan_cpu, self_mixup=args.self_mixup,
                             noise_weighting=args.noise_weighting)
        denom = torch.tensor([loss_cpu["struct_mask"].sum(), loss_cpu["aa_mask"].sum()],
                              device=device).clamp_min(1)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.lr, args.warmup, args.schedule_steps, args.warmup_init_lr)
        optimizer.zero_grad(set_to_none=True)
        numer = torch.zeros(2, device=device)
        for micro in range(args.accumulate):
            lo = (micro * world + rank) * args.batch_size
            hi = lo + args.batch_size
            width = int((batch_cpu["state"][lo:hi] != 0).sum(-1).max())
            batch = {k: v[lo:hi, :width].to(device) for k, v in batch_cpu.items()}
            plan = {k: (v[lo:hi, :width] if v.ndim == 2 else v[lo:hi]).to(device)
                    for k, v in plan_cpu.items()}
            objective = {k: (v[lo:hi, :width] if v.ndim == 2 else v[lo:hi]).to(device)
                         for k, v in loss_cpu.items()}
            context = wrapped.no_sync() if world > 1 and micro < args.accumulate-1 else contextlib.nullcontext()
            with context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    extra = {}
                    if args.self_mixup == 'released':
                        extra['token_input_ids'] = released_mixup_input(model, batch, plan)
                    output = wrapped(batch, plan["struct_mask"], plan["aa_mask"], plan["independent"], **extra)
                    sums, _ = loss_sums(model, output, batch, objective)
                    loss = 0.5 * world * (sums / denom).sum()
                loss.backward()
            numer += sums.detach()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
        optimizer.step()
        for i in ids:
            e = cache.entries[i]
            exposure[e["source"]] = exposure.get(e["source"], 0) + e["length"]
        if world > 1:
            dist.all_reduce(numer)
        torch.cuda.synchronize()
        peak_memory = torch.tensor(torch.cuda.max_memory_allocated(), device=device)
        if world > 1:
            dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
        if rank == 0:
            row = dict(step=step+1, loss=float(0.5*(numer/denom).sum()),
                struct_loss=float(numer[0]/denom[0]), aa_loss=float(numer[1]/denom[1]),
                lr=optimizer.param_groups[0]["lr"], gradient_norm=float(grad), exposure=exposure.copy(),
                elapsed=time.monotonic()-started, peak_memory_gb=float(peak_memory)/1e9)
            row['valid_residues'] = sum(cache.entries[i]['length'] for i in ids)
            row['paired_tokens'] = 2 * sum(cache.entries[i]['length']+2 for i in ids)
            row['task_counts'] = torch.bincount(plan_cpu['task'], minlength=4).tolist()
            row['objective'] = dict(self_mixup=args.self_mixup, noise_weighting=args.noise_weighting,
                supervised_tokens=[int(loss_cpu[k].sum()) for k in ('struct_mask', 'aa_mask')],
                weight_mass=[float((loss_cpu[m].sum(-1)*loss_cpu[w]).sum())
                             for m,w in [('struct_mask','sw'),('aa_mask','aw')]])
            row["update_seconds"] = time.monotonic() - update_started
            with open(args.out / "metrics.jsonl", "a") as f:
                f.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
        milestone = (step+1) % args.milestone_every == 0 or step+1 == args.steps
        if (step+1) % args.checkpoint_every == 0 or step+1 == endpoint or milestone:
            rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device))
            all_rng = [None] * world
            if world > 1:
                dist.all_gather_object(all_rng, rng)
            else:
                all_rng = [rng]
            if rank == 0:
                save_atomic(args.out / "last.ckpt", dict(model=model.state_dict(),
                    model_spec=model.specification(), optimizer=optimizer.state_dict(),
                    step=step+1, exposure=exposure, rng=all_rng, contract=contract,
                    fast_provenance=fast_provenance))
                if milestone:
                    weights = args.out / 'milestones' / f'step_{step+1:07d}.pt'
                    weights.parent.mkdir(exist_ok=True)
                    if weights.exists():
                        raise FileExistsError(weights)
                    save_atomic(weights, dict(model=model.state_dict(), model_spec=model.specification(),
                        step=step+1, fast_provenance=fast_provenance, contract=contract))
            if world > 1:
                dist.barrier()
        if (step+1) % args.validate_every == 0 or step+1 == endpoint or milestone:
            evaluate(step+1, sampling=milestone or step+1 == endpoint)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if lock is not None:
        lock.close()


if __name__ == "__main__":
    main()
