"""Regression: an unresolved farther constituent must invalidate link distance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import install,ROOT,write_json
install()
import numpy as np
from lift_hold_collision_oracle import StaticCollisionOracle


def entry(distance,valid=True,consistent=True):
    witness=np.array([0.,0.,0.,distance if consistent else distance+.004,0.,0.])
    return (distance,witness,'fixture',distance,witness.copy(),None,valid,None,False)


def sample(near,far):
    oracle=StaticCollisionOracle.__new__(StaticCollisionOracle)
    oracle.geom_ids={'A':{'capsule':[1,2]},'B':{'capsule':[3]}}
    oracle._cache={(1,3,10.):near,(2,3,10.):far}
    return oracle.pair_distance('A','B')


def main():
    bad=sample(entry(.001),entry(.002,valid=False,consistent=False))
    good=sample(entry(.001),entry(.002))
    near_bad=sample(entry(.001,valid=False),entry(.002))
    gates=dict(minimum_still_correct=bad['distance_m']==.001,
        selected_pair_was_certified=bad['selected_pair_signed_distance_valid'],
        unresolved_far_pair_not_hidden=not bad['signed_distance_valid'],
        inconsistent_far_witness_not_hidden=not bad['witness_gap_consistent'],
        unresolved_constituent_recorded=len(bad['unresolved_constituent_pairs'])==1,
        all_constituents_counted=bad['constituent_pair_count']==2,
        all_resolved_positive_control=good['signed_distance_valid'] and good['witness_gap_consistent'],
        unresolved_near_pair_also_fails=not near_bad['signed_distance_valid'])
    write_json(ROOT/'reports/collision_aggregation_fixture.json',dict(status='PASS' if all(gates.values()) else 'FAIL',gates=gates))
    assert all(gates.values()),gates
    print('Collision aggregation regression:8/8 PASS; no physics executed.')


if __name__=='__main__':main()
