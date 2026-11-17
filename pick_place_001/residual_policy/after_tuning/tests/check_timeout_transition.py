"""CPU regression: endpoint reset followed by ordinary transitions.

The actual G1HOI termination code and DaggerPPO reward ingestion are used.
Only the physics SDK step boundary is replaced with a deterministic fixture;
no simulator, gradient, backward call, or optimizer update is performed.
"""
import json,sys
from pathlib import Path
from types import SimpleNamespace as S
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT,REPO,install,write_json,relocate_path,relocate_values
install()
sys.path[:0]=[str(ROOT/'scripts')]+[str(REPO/p) for p in ['pose','legged_gym','rsl_rl']]
import isaacgym
import torch
from residual_env import ResidualFloorlampEnv
from physical_floorlamp_env import PhysicalLampEnv
from legged_gym.envs.g1.g1_hoi import G1HOI
from rsl_rl.algorithms.dagger_ppo import DaggerPPO

env=object.__new__(ResidualFloorlampEnv)
env.num_envs=1;env.num_actions=29;env.device='cpu'
env.cfg=S(env=S(send_timeouts=True),rewards=S(root_height_diff_threshold=.3,termination_roll=1.,termination_pitch=1.,termination_when_object_far=False))
env.contact_forces=torch.zeros(1,38,3);env.termination_contact_indices=torch.tensor([],dtype=torch.long)
env.all_root_states=torch.zeros(2,13);env.root_states=env.all_root_states[:1];env.object_root_states=env.all_root_states[1:]
env._ref_root_pos=torch.zeros(1,3);env._ref_object_root_pos=torch.zeros(1,3);env.dof_state=torch.zeros(1,29,2)
env.roll=torch.zeros(1);env.pitch=torch.zeros(1);env.dt=.02;env.max_episode_length=1000
env._motion_lib=S(get_motion_length=lambda ids:torch.tensor([5.416667]));env._motion_ids=torch.zeros(1,dtype=torch.long)
env.viewer=None;env._pose_termination=False;env.object_point_error=lambda:torch.zeros(1)
env.extras={};env.time_out_buf=torch.zeros(1,dtype=torch.bool)
fixture_index=[0];snapshots=[]

def fake_physics_step(self,actions):
    if fixture_index[0]==0:
        # Model a real endpoint/reset: parent stores the timeout tensor reference.
        self.episode_length_buf=torch.tensor([270])
        self.time_out_buf=torch.tensor([True]);self.reset_buf=torch.tensor([True])
        self.extras['time_outs']=self.time_out_buf
    else:
        self.episode_length_buf=torch.tensor([fixture_index[0]])
        # Actual code allocates a new False tensor; previous extras remains True
        # unless the local adapter publishes a fresh snapshot after this step.
        G1HOI.check_termination(self)
    fixture_index[0]+=1
    return torch.zeros(1,1764),torch.zeros(1,1741),torch.ones(1),self.reset_buf,self.extras

parent_step=PhysicalLampEnv.step
try:
    PhysicalLampEnv.step=fake_physics_step
    for i in range(3):
        obs,priv,reward,done,info=env.step(torch.zeros(1,29))
        saved={}
        transition=S(values=torch.tensor([[10.]]),clear=lambda:None)
        algorithm=S(transition=transition,gamma=.99,device='cpu',storage=S(add_transitions=lambda t:saved.update(reward=t.rewards.clone())),actor_critic=S(reset=lambda d:None))
        DaggerPPO.process_env_step(algorithm,reward,done,info)
        snapshots.append(dict(done=bool(done[0]),current_timeout=bool(env.time_out_buf[0]),reported_timeout=bool(info['time_outs'][0]),stored_reward=float(saved['reward'][0]),metadata_is_snapshot=info['time_outs'] is not env.time_out_buf))
finally:PhysicalLampEnv.step=parent_step

checks=dict(endpoint_bootstrap=snapshots[0]['reported_timeout'] and snapshots[0]['done'] and abs(snapshots[0]['stored_reward']-10.9)<1e-5,
            next_ordinary_no_bootstrap=not snapshots[1]['done'] and not snapshots[1]['reported_timeout'] and snapshots[1]['stored_reward']==1.,
            later_ordinary_no_bootstrap=not snapshots[2]['done'] and not snapshots[2]['reported_timeout'] and snapshots[2]['stored_reward']==1.,
            independent_snapshots=all(s['metadata_is_snapshot'] for s in snapshots),
            old_path_rebased=relocate_path('/old/server/ResMimic/InterPose/custom_data/pick_place_001/example.npz')==str(ROOT.parents[2]/'pick_place_001/example.npz'),
            nested_metadata_rebased=relocate_values({'asset':{'file':'/old/server/ResMimic/assets/g1/a.urdf'}})['asset']['file']==str(REPO/'assets/g1/a.urdf'))
env.cfg.env.send_timeouts=False
PhysicalLampEnv.step=fake_physics_step
try:
    env.step(torch.zeros(1,29));checks['disabled_timeouts_removed']='time_outs' not in env.extras
finally:PhysicalLampEnv.step=parent_step

report=dict(status='PASS_TIMEOUT_TRANSITION_REGRESSION' if all(checks.values()) else 'FAIL',checks=checks,transitions=snapshots,
            simulator_executed=False,optimizer_updates=0,training_executed=False,
            scope='Actual termination/reward-ingestion methods on CPU at a synthetic physics boundary; endpoint then ordinary steps. Does not validate physical carrying or PPO convergence.')
write_json(ROOT/'reports/publication_timeout_regression.json',report);print(json.dumps(report,indent=2));assert all(checks.values()),checks
