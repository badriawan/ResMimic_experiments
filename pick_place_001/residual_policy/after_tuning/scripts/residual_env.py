"""Local multi-environment floorlamp residual task; importing does not train."""
from copy import deepcopy
from types import SimpleNamespace
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate
from common import ROOT,ASSETS,MOTION,OBJECT,CHECKPOINT
from baseline_config import build_configs as original_configs,DELTA_METRICS
from legged_gym.gym_utils.helpers import class_to_dict
from physical_floorlamp_env import PhysicalLampEnv

def build_configs(num_envs,iterations,save_interval=100):
    cfg,train=original_configs(num_envs,iterations)
    cfg.env.num_actors=2
    cfg.env.nonblind=True
    cfg.env.num_observations=1764  # 1757 base layout + seven current object features.
    cfg.env.num_privileged_obs=1741  # Existing1734 + seven current object features.
    cfg.env.track_root=False
    cfg.env.pose_termination=True
    cfg.env.pose_termination_dist=.7
    cfg.env.record_video=False
    cfg.motion.motion_file=str(MOTION);cfg.motion.object_motion_file=str(OBJECT)
    cfg.env.object_asset_root=str(ASSETS);cfg.env.object_urdf_file='floorlamp_physical.urdf';cfg.env.object_obj_file='floorlamp_visual.obj'
    # Selected reference contains an exact hold. These are its actual derivatives,
    # verified again from the files before constructing this environment.
    cfg.env.object_initial_linear_velocity=[0.,0.,0.]
    cfg.env.object_initial_angular_velocity=[0.,0.,0.]
    cfg.asset.object_mass=4.;cfg.asset.object_density=12
    # Let tracking rewards continue through object drift rather than repeatedly
    # terminating before the later carrying segment. Physical robot stops remain.
    cfg.rewards.termination_when_object_far=False
    cfg.rewards.regularization_scale_curriculum=False
    scales=class_to_dict(cfg.rewards.scales)
    scales['tracking_object_point_cloud']=2.
    cfg.rewards.scales=SimpleNamespace(**scales)
    # Remove obsolete evaluation attributes before any constructor reset reports.
    cfg.evaluations=SimpleNamespace(**{k:v for k,v in class_to_dict(cfg.evaluations).items() if k not in DELTA_METRICS})
    for key in ['randomize_object','randomize_object_pos','randomize_object_scale','randomize_object_friction','randomize_object_mass','randomize_object_com']:
        setattr(cfg.domain_rand,key,False)
    train.env=deepcopy(cfg.env)
    train.policy.nonblind=True
    train.policy.num_object_observations=7
    train.policy.base_policy_path=str(CHECKPOINT)
    train.runner.experiment_name='pick_place_after_tuning_object_residual'
    train.runner.save_interval=save_interval
    return cfg,train

class ResidualFloorlampEnv(PhysicalLampEnv):
    """Automatic full/subset episode resets; object pose writes only at reset.

    PhysicalLampEnv.active deliberately stays False: that flag only serves the
    old single-episode probe, which suppressed automatic reset. Here real PPO
    resets and episode statistics remain enabled.
    """
    def __init__(self,*args,**kwargs):
        self.reset_calls=0;self.continuing_reset_integrations=0;self.continuing_max_displacement_m=0.
        self._inside_episode_reset=False
        super().__init__(*args,**kwargs)
        self.extras.pop('episode',None)
        assert self.num_obs==1764 and self.num_privileged_obs==1741

    def _reset_root_states(self,env_ids,**kwargs):
        if not self._inside_episode_reset:raise RuntimeError('Robot/object pose setters are allowed only within episode reset')
        super()._reset_root_states(env_ids,**kwargs)

    def reset_idx(self,ids,motion_ids=None):
        if not len(ids):return
        continuing=torch.ones(self.num_envs,device=self.device,dtype=torch.bool);continuing[ids]=False
        before=self.root_states[continuing,:3].clone()
        self._inside_episode_reset=True
        try:super().reset_idx(ids,motion_ids=motion_ids)
        finally:self._inside_episode_reset=False
        self.reset_calls+=1;self.continuing_reset_integrations+=int(continuing.sum())
        if before.numel():self.continuing_max_displacement_m=max(self.continuing_max_displacement_m,float(torch.linalg.vector_norm(self.root_states[continuing,:3]-before,dim=-1).max()))

    def _update_ref_motion(self):
        super()._update_ref_motion()
        # Stock G1HOI translates the robot reference, but leaves object local.
        # The loader supplies new local values on every call, so add exactly once.
        self._ref_object_root_pos+=self.episode_init_origin

    def _object_features(self):
        return torch.cat((self.object_root_states[:,:3]-self.episode_init_origin,self.object_root_states[:,3:7]),dim=-1)

    def compute_observations(self):
        super().compute_observations()
        start=self.cfg.env.n_obs_single*(self.cfg.env.history_len+1)
        features=self._object_features()
        self.obs_buf[:,start:start+7]=features
        # Parent replaces the critic buffer each call; append once, never to history.
        self.privileged_obs_buf=torch.cat((self.privileged_obs_buf,features),dim=-1)

    def object_point_error(self):
        n=self.num_envs;points=self.object_points[None].expand(n,-1,-1)
        flat=points.reshape(-1,3)
        actual_q=self.object_root_states[:,None,3:7].expand(-1,points.shape[1],-1).reshape(-1,4)
        ref_q=self._ref_object_root_rot[:,None,:].expand(-1,points.shape[1],-1).reshape(-1,4)
        actual=quat_rotate(actual_q,flat).reshape(n,-1,3)+self.object_root_states[:,None,:3]
        reference=quat_rotate(ref_q,flat).reshape(n,-1,3)+self._ref_object_root_pos[:,None,:]
        self.object_point_cloud_dist=torch.linalg.vector_norm(actual-reference,dim=-1).mean(-1)
        return self.object_point_cloud_dist

    def _reward_tracking_object_point_cloud(self):
        # Independent of termination_when_object_far; never stale/missing.
        return torch.exp(-10.*self.object_point_error())

    def check_termination(self):
        super().check_termination()
        lengths=self._motion_lib.get_motion_length(self._motion_ids)
        bound=torch.floor((lengths-self._motion_time_offsets-1e-5)/self.dt).long()
        endpoint=self.episode_length_buf>=bound
        self.time_out_buf|=endpoint;self.reset_buf|=endpoint
        self.endpoint_timeout_count+=int(endpoint.sum())
        nonfinite=~torch.isfinite(self.all_root_states.reshape(self.num_envs,-1)).all(-1)
        nonfinite|=~torch.isfinite(self.dof_state.reshape(self.num_envs,-1)).all(-1)
        self.reset_buf|=nonfinite
        self.reset_buf|=torch.linalg.vector_norm(self.object_root_states[:,7:10],dim=-1)>10.

    def step(self,actions):
        if actions.shape!=(self.num_envs,29) or not bool(torch.isfinite(actions).all()):raise FloatingPointError('Invalid summed residual/base actions')
        self.extras.pop('episode',None)
        result=super().step(actions)
        # The parent stores extras["time_outs"] only on reset but allocates a
        # new time_out_buf each subsequent step. Publish this transition's
        # snapshot every time, including ordinary nonterminal steps.
        if self.cfg.env.send_timeouts:
            self.extras['time_outs']=(self.time_out_buf.bool() & result[3].bool()).detach().clone()
        else:
            self.extras.pop('time_outs',None)
        for tensor in [result[0],result[1],result[2],self.all_root_states,self.dof_state]:
            if not bool(torch.isfinite(tensor).all()):raise FloatingPointError('Nonfinite state/reward/observation')
        # Always finite diagnostic values, including when no episode just ended.
        with torch.no_grad():
            self.extras['object_metrics']={'point_error_m':float(self.object_point_error().mean()),'position_error_m':float(torch.linalg.vector_norm(self.object_root_states[:,:3]-self._ref_object_root_pos,dim=-1).mean())}
        return result

    def training_diagnostics(self):
        return dict(reset_calls=self.reset_calls,object_pose_setter_calls_at_episode_reset=self.object_pose_setter_calls,
                    continuing_reset_integrations=self.continuing_reset_integrations,continuing_max_displacement_m=self.continuing_max_displacement_m,
                    endpoint_timeout_count=self.endpoint_timeout_count,reference_query_clamp_count=self.reference_query_clamp_count,
                    timeout_snapshot_per_transition=True,
                    robot_and_object_actors=self.num_actors,automatic_episode_resets=True,object_point_reward_weight=self.cfg.rewards.scales.tracking_object_point_cloud,
                    limitations=['Parent reset advances all environments by one2ms physics substep, including continuing environments.','No actual lift/carry success is implied by loss or episode reward.'])
