"""Bind completed infrastructure checks to one immutable recovery bundle."""
import argparse
import math
import statistics
from pathlib import Path

from native_recovery_common import read,write,sha,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bundle-sha256',required=True);a=p.parse_args();root=a.root.resolve()
    cfg=verify(root,a.bundle_sha256);inputs={}
    def checked(name):
        path=root/name;inputs[name]=sha(path);return read(path)
    assert checked('data_audit.json')['passed']
    assert checked('reference_audit.json')['matched']==32
    assert checked('cpu_verification.json')['passed']
    panel=checked('panel.json');assert len(panel['conditional'])==32 and len(panel['joint'])==20
    render=checked('renderer_check/geometry.json');s=render['groups']['control']['summary']
    gate=cfg['renderer_gate']
    assert render['failures']==gate['failures']==0
    assert s['requested']==s['scored']==s['fold_scored']==gate['requested']==32
    assert s['median_ca_rmsd_a']<gate['median_ca_rmsd_below_a']
    assert s['mean_ca_lddt']>=gate['mean_ca_lddt_at_least']
    assert render['panel_sha256']==sha(root/'panel.json')
    if (root/'renderer_reuse.json').exists():checked('renderer_reuse.json')
    canary=checked('canary_verification.json')
    assert canary['passed'] and canary['updates']==20 and canary['split']==10
    assert canary['bundle_sha256']==a.bundle_sha256
    assert canary['initial_sha256']==checked('initial.json')['sha256']==sha(root/'initial.pt')
    peaks=[]
    for run in ('canary_reference','canary_split'):
        assert checked(run+'/complete.json')['step']==20
        assert checked(run+'/complete.json')['last_sha256']==sha(root/run/'last.pt')
        for rank in range(8):
            stress=checked(f'{run}/stress_rank_{rank}.json')
            assert stress['passed'] and stress['local_residues']==8192
            peaks.append(stress['peak_memory_gb'])
    folder='canary_reference/evaluation/step_0000000/'
    geometry=checked(folder+'geometry.json')
    assert geometry['failures']==0
    for kind in ('conditional','control','joint'):
        summary=geometry['groups'][kind]['summary']
        assert summary['requested']==summary['scored']==2
    assert checked(folder+'complete.json')['geometry_sha256']==sha(root/folder/'geometry.json')
    validation=checked(folder+'validation.json')
    assert validation['validation_entries']==128 and math.isfinite(validation['full_mask_structure_ce'])
    fold=checked(folder+'refold.json')
    assert fold['summary']['requested']==fold['summary']['scored']==2
    assert fold['geometry_sha256']==sha(root/folder/'geometry.json')
    import json
    metrics=[json.loads(line) for line in (root/'canary_reference/metrics.jsonl').read_text().splitlines()]
    assert len(metrics)==20
    assert all(math.isfinite(r[k]) for r in metrics for k in ('loss','gradient_norm','lr'))
    mean=statistics.mean(r['update_seconds'] for r in metrics[2:])
    result=dict(production_ready=True,bundle_sha256=a.bundle_sha256,verified_artifacts=inputs,
        verification_script_sha256=sha(__file__),renderer_control=s,
        maximum_stress_memory_gb=max(peaks),strict_resume=canary,
        canary_mean_update_seconds=mean,rough_training_node_hours=mean*cfg['steps']/3600,
        timing_limitation='20-update sample; excludes validation/checkpoint overhead and later batch-length variation',
        quality_claim='Infrastructure passed; baseline generation recovery is not yet established.')
    write(root/'verification.json',result);print(result)


if __name__=='__main__':main()
