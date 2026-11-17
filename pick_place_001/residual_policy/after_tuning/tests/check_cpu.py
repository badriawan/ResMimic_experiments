"""CPU fixture checks only: no simulator, rollouts, backward or optimizer step."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT,REPO,install,write_json
install()
sys.path[:0]=[str(ROOT/'scripts')]+[str(REPO/x) for x in ['pose','legged_gym','rsl_rl']]
import isaacgym
import torch
from types import SimpleNamespace
from residual_env import build_configs,ResidualFloorlampEnv
from legged_gym.gym_utils.helpers import class_to_dict
from rsl_rl.runners import OnPolicyDaggerRunner
import physical_floorlamp_env

cfg,train=build_configs(2,2)
checks=dict(dimensions=cfg.env.num_observations==1764 and cfg.env.num_privileged_obs==1741,
            two_actors=cfg.env.num_actors==2,object_reward=cfg.rewards.scales.tracking_object_point_cloud==2.,
            delta_schema='tracking_root_pose_delta_local' not in class_to_dict(cfg.evaluations),
            positive_save_interval=train.runner.save_interval>0)
env=SimpleNamespace(cfg=cfg,num_obs=1764,num_privileged_obs=1741,num_envs=2,num_actions=29,obs_buf=torch.zeros(2,1764))
runner=OnPolicyDaggerRunner(env,class_to_dict(train),device='cpu',log_dir=None)
start=cfg.env.n_obs_single*(cfg.env.history_len+1)
raw=torch.randn(2,1764)*.01;raw[:,-360:]=0;raw[:,start:start+7]=torch.tensor([1.,2.,1.,0.,0.,0.,1.])
base=runner.full_obs_to_base_obs(raw)
changed=raw.clone();changed[:,start:start+7]+=10
checks['base_stripping_exact']=torch.equal(base,torch.cat((raw[:,:start],raw[:,start+7:]),dim=-1)) and torch.equal(base,runner.full_obs_to_base_obs(changed))
with torch.no_grad():
    norm,priv=runner._normalize_observations(raw,torch.zeros(2,1741))
    residual=runner.alg.actor_critic.act_inference(norm);value=runner.alg.actor_critic.evaluate(priv);base_action=runner.base_policy(base)
checks['actual_network_forward']=residual.shape==(2,29) and value.shape==(2,1) and base_action.shape==(2,29) and all(bool(torch.isfinite(x).all()) for x in [residual,value,base_action])
checks['base_frozen']=all(not p.requires_grad for p in runner.base_policy.parameters())
opt={id(p) for g in runner.alg.optimizer.param_groups for p in g['params']}
checks['optimizer_excludes_base_teacher']=not (opt&{id(p) for p in runner.base_policy.parameters()}) and not (opt&{id(p) for p in runner.teacher_actor_critic.parameters()})
checks['no_optimizer_updates']=not runner.alg.optimizer.state and runner.alg.counter==0

# Exercise actual reference translation using a two-environment synthetic loader.
m=object.__new__(ResidualFloorlampEnv);m.cfg=cfg;m.num_envs=2;m.device='cpu';m.num_dof=29;m.num_actions=29
m.episode_init_origin=torch.tensor([[10.,-2.,0.],[-4.,3.,0.]])
m.env_origins=m.episode_init_origin.clone();m._motion_ids=torch.zeros(2,dtype=torch.long)
m._get_motion_times=lambda:torch.zeros(2)
q=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1)
fixture=[torch.tensor([[0.,0.,.8]]).repeat(2,1),q,torch.zeros(2,3),torch.zeros(2,3),torch.ones(2,29)*.2,torch.zeros(2,29),torch.zeros(2,38,3),torch.zeros(2,3),q,torch.tensor([[1.,2.,1.]]).repeat(2,1),q]
m._motion_lib=SimpleNamespace(calc_hoi_motion_frame=lambda ids,t:[v.clone() for v in fixture])
for name,value in zip(['_ref_root_pos','_ref_root_rot','_ref_root_vel','_ref_root_ang_vel','_ref_dof_pos','_ref_dof_vel','_ref_body_pos','_ref_root_pos_delta_local','_ref_root_rot_delta_local','_ref_object_root_pos','_ref_object_root_rot'],fixture):setattr(m,name,torch.zeros_like(value))
m._update_ref_motion();first=m._ref_object_root_pos.clone();m._update_ref_motion()
checks['world_object_reference_once']=torch.allclose(first,fixture[9]+m.env_origins) and torch.equal(first,m._ref_object_root_pos)
m.all_root_states=torch.zeros(4,13);m.root_states=m.all_root_states.reshape(2,2,13)[:,0];m.object_root_states=m.all_root_states.reshape(2,2,13)[:,1]
m.root_states[:,3:7]=q;m.object_root_states[:,:3]=m._ref_object_root_pos;m.object_root_states[:,3:7]=q
m.object_points=torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.2,.3]])
checks['object_features_local']=torch.allclose(m._object_features(),torch.cat((fixture[9],q),-1))
m.object_root_states[1,0]+=.1
reward=m._reward_tracking_object_point_cloud()
checks['object_reward_origin_independent']=torch.allclose(reward,torch.tensor([1.,torch.exp(torch.tensor(-1.))]),atol=2e-6)

# Repeated observation assembly must append to a fresh parent critic buffer.
parent_compute=physical_floorlamp_env.PhysicalLampEnv.compute_observations
def fake_parent(self):
    self.obs_buf=torch.zeros(2,1764);self.privileged_obs_buf=torch.zeros(2,1734)
try:
    physical_floorlamp_env.PhysicalLampEnv.compute_observations=fake_parent
    m.compute_observations();m.compute_observations()
    checks['repeated_observation_dimensions']=m.obs_buf.shape==(2,1764) and m.privileged_obs_buf.shape==(2,1741)
    checks['actor_critic_object_features']=torch.equal(m.obs_buf[:,start:start+7],m._object_features()) and torch.equal(m.privileged_obs_buf[:,-7:],m._object_features())
    checks['future_slice_untouched']=bool((m.obs_buf[:,-360:]==0).all())
finally:physical_floorlamp_env.PhysicalLampEnv.compute_observations=parent_compute

# Actual reset methods against a fake SDK: verify selected actor IDs and origins.
records={}
class FakeGym:
    def set_actor_root_state_tensor_indexed(self,sim,states,ids,count):records['roots']=ids.clone()
    def set_dof_state_tensor_indexed(self,sim,states,ids,count):records['dofs']=ids.clone()
m.gym=FakeGym();m.sim=None;m.all_actor_ids=torch.arange(4,dtype=torch.int32).reshape(2,2)
m._inside_episode_reset=True;m.active=False;m.object_pose_setter_calls=0;m.asset_profile={'initialization':{'upward_floor_correction_m':0.}}
m.dof_state=torch.zeros(2,29,2);m.dof_pos=m.dof_state[:,:,0];m.dof_vel=m.dof_state[:,:,1]
before=m.all_root_states[:2].clone();unwrap=physical_floorlamp_env.gymtorch.unwrap_tensor
try:
    physical_floorlamp_env.gymtorch.unwrap_tensor=lambda tensor:tensor
    ids=torch.tensor([1]);m._reset_root_states(ids);m._reset_dofs(ids,None,None)
finally:physical_floorlamp_env.gymtorch.unwrap_tensor=unwrap
checks['subset_actor_indices']=torch.equal(records['roots'],torch.tensor([2,3],dtype=torch.int32)) and torch.equal(records['dofs'],torch.tensor([2],dtype=torch.int32))
# _reset_root_states expects LOCAL references, as supplied by _reset_ref_motion.
# Above fixture uses current world refs, so run a second reset with proper local data.
m._ref_root_pos[:]=fixture[0];m._ref_object_root_pos[:]=fixture[9]
try:
    physical_floorlamp_env.gymtorch.unwrap_tensor=lambda tensor:tensor
    m._reset_root_states(torch.tensor([1]))
finally:physical_floorlamp_env.gymtorch.unwrap_tensor=unwrap
checks['reset_world_translation']=torch.allclose(m.root_states[1,:3],fixture[0][1]+m.env_origins[1]) and torch.allclose(m.object_root_states[1,:3],fixture[9][1]+m.env_origins[1])
checks['subset_other_actor_untouched']=torch.equal(before,m.all_root_states[:2])
checks['consistent_zero_reset_velocities']=bool((m.root_states[1,7:]==0).all() and (m.object_root_states[1,7:]==0).all() and (m.dof_vel[1]==0).all())
m._inside_episode_reset=False
try:m._reset_root_states(torch.tensor([1]));checks['pose_setter_outside_reset_blocked']=False
except RuntimeError:checks['pose_setter_outside_reset_blocked']=True
checks={k:bool(v) for k,v in checks.items()}
report=dict(status='PASS_CPU_FIXTURES' if all(checks.values()) else 'FAIL',checks=checks,simulator_executed=False,training_executed=False,optimizer_updates=0,
            scope='Actual CPU runner/network forward and local tensor/reset methods with synthetic loader/SDK; no physical simulation or PPO backward/update verified.')
write_json(ROOT/'reports/cpu_validation.json',report);print(json.dumps(report,indent=2));assert all(checks.values())
