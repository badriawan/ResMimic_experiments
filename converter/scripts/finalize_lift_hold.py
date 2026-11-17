"""Publish an immutable pair only after independent quality/loader reports pass."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import ROOT,DATA,install,sha,write_json
install()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--loader-report',type=Path,required=True)
    p.add_argument('--geometry-report',type=Path,required=True)
    args=p.parse_args()
    source=args.motion.resolve(strict=True)
    reports=[args.loader_report.resolve(strict=True),args.geometry_report.resolve(strict=True)]
    if DATA not in source.parents or any(ROOT not in r.parents for r in reports):
        raise ValueError('Input/report boundary mismatch')
    loader,geometry=[json.loads(r.read_text()) for r in reports]
    quality=ROOT/'config/quality_gates.json'
    for report in [loader,geometry]:
        if report['motion_sha256']!=sha(source) or report['quality_config_sha256']!=sha(quality):
            raise ValueError('Stale or mismatching independent validation')
    if loader['status']!='PASS_ACTUAL_LOADER' or not all(loader['gates'].values()) or not all(loader['contact_gates'].values()):
        raise ValueError('Actual loader/contact validation did not pass')
    if geometry['status']!='PASS_REFERENCE_KINEMATIC':raise ValueError('Independent reference quality failed')
    for group in ['array_gates','contact_gates','correction_gates','collision_gates']:
        if not geometry.get(group) or not all(geometry[group].values()):
            raise ValueError('Missing/failed quality group '+group)
    rate=geometry['collision']['sample_rate_hz']
    if (geometry['collision']['samples']!=loader['sample_count']
            or (rate<600 and not math.isclose(rate,600,rel_tol=0,abs_tol=1e-6))):
        raise ValueError('Collision audit did not cover the full dense loader grid')
    if not all(geometry['input_hashes_preserved'].values()):raise ValueError('Validation mutated inputs')
    final=DATA/'final'
    if final.exists():raise FileExistsError('Final output exists; immutable publication refused')
    final.mkdir()
    output=final/'lift_hold_001_g1.pkl'
    for original,destination in [(source,output),(source.with_suffix('.npz'),output.with_suffix('.npz'))]:
        with original.open('rb') as src,destination.open('xb') as dst:
            for block in iter(lambda:src.read(1024*1024),b''):dst.write(block)
        if sha(original)!=sha(destination):raise ValueError('Publication copy differs')
    manifest=dict(status='PASS_REFERENCE_AND_LOADER_WITH_DECLARED_LIMITATIONS',
        motion=str(output),object_motion=str(output.with_suffix('.npz')),mesh=str(DATA/'assets/floorlamp.obj'),
        hashes={str(x):sha(x) for x in [output,output.with_suffix('.npz'),DATA/'assets/floorlamp.obj']},
        validated_candidate=str(source),validated_candidate_sha256=sha(source),
        final_pair_byte_identical_to_validated_candidate=True,
        frames=loader['frames'],fps=loader['fps'],duration_s=loader['duration_s'],
        loader_report=str(reports[0]),loader_report_sha256=sha(reports[0]),
        reference_report=str(reports[1]),reference_report_sha256=sha(reports[1]),
        quality_configuration_sha256=sha(quality),
        required_interpolation_adapter=str(ROOT/'scripts/lift_hold_reference_loader.py'),
        ready_for_object_free_base_policy_trial=True,base_policy_tracking_verified=False,
        ready_for_residual_preflight_trial=True,residual_training_executed=False,
        policy_executed=False,render_executed=False,physical_grasp_verified=False,
        warnings=['Scale1 preserves source metric15cm lift relative to known original initial object position.',
            'Supported overlap begins after object has already risen8.262mm; lift from first exported pose is141.738mm.',
            'Last33.39ms of human capture is trimmed; complete withdraw/settle is not certified.',
            'Object reference geometry reaches7.407mm below nominalfloor; no physical floorlamp asset is certified.',
            'RootXY drift is preserved; source stay-in-place goal is not certified.',
            'Generated foot handoffs/rootZcorrection and small arm separation are declared kinematic corrections.',
            'Dense source shape collision proxies are not PhysX cooking/force/balance proof.',
            'Use the normalized shortest-path SLERP adapter; stock tiny-angle interpolation is not certified.'])
    write_json(ROOT/'reports/final_conversion_manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],motion=str(output),object_motion=str(output.with_suffix('.npz')))))


if __name__=='__main__':main()
