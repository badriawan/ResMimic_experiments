"""Pinned local residual configuration, adapted from the project's P1 adapter.

The caller imports Isaac Gym before this module. No simulator is constructed
here. Model and task classes are supplied by the pinned ResMimic checkout.
"""
from copy import deepcopy
from common import REPO,MOTION,CHECKPOINT
from legged_gym.envs.g1.g1_motion_config import G1MotionCfg,G1MotionCfgDAgger
DELTA_METRICS=('tracking_root_pose_delta_local','tracking_root_rotation_delta_local')

def build_configs(num_envs=32, iterations=50):
    """Build independent checkout config instances; do not construct a simulator."""
    if not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError('num_envs must be a positive integer')
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError('iterations must be a positive integer')
    cfg = G1MotionCfg()
    train_cfg = G1MotionCfgDAgger()
    cfg.seed = train_cfg.seed = 1
    cfg.env.num_envs = num_envs
    cfg.env.num_actors = 1
    cfg.env.nonblind = False
    cfg.env.debug_viz = False
    cfg.env.record_video = False
    cfg.env.rand_reset = False
    cfg.env.randomize_start_pos = False
    cfg.env.randomize_start_yaw = False
    cfg.env.enable_force_curriculum = False
    cfg.env.track_root = False
    cfg.env.pose_termination = True
    cfg.env.pose_termination_dist = .7
    cfg.env.evaluation_mode = True
    cfg.env.force_full_masking = True
    cfg.env.curriculum_masking = False
    cfg.env.temporal_dropout = False
    cfg.env.adaptive_masking = False
    cfg.env.use_masked_priv_info = False
    cfg.noise.add_noise = False
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand, key), bool):
            setattr(cfg.domain_rand, key, False)
    cfg.domain_rand.motor_strength_range = [1., 1.]
    cfg.motion.motion_file = str(MOTION)
    cfg.motion.motion_curriculum = False
    cfg.motion.motion_smooth = False
    cfg.motion.motion_decompose = False
    cfg.motion.height_offset = 0.
    cfg.asset.file = str(REPO / 'assets/g1/g1_custom_collision_29dof.urdf')
    cfg.terrain.mesh_type = 'plane'
    cfg.terrain.curriculum = False
    cfg.rewards.termination_roll = 1.
    cfg.rewards.termination_pitch = 1.
    cfg.rewards.root_height_diff_threshold = .3
    for name in DELTA_METRICS:
        setattr(cfg.evaluations, name, False)
        if float(getattr(cfg.rewards.scales, name, 0.)) != 0.:
            raise ValueError('Unreviewed FPS-dependent reward is active: ' + name)
    # The runner uses this section for history/observation bookkeeping.
    train_cfg.env = deepcopy(cfg.env)
    train_cfg.runner.experiment_name = 'priority1_residual_robot_only'
    train_cfg.runner.policy_class_name = 'ActorCriticFuture'
    train_cfg.runner.algorithm_class_name = 'DaggerPPO'
    train_cfg.runner.runner_class_name = 'OnPolicyDaggerRunner'
    train_cfg.runner.max_iterations = iterations
    train_cfg.runner.num_steps_per_env = 24
    train_cfg.runner.save_interval = 10
    train_cfg.runner.warm_iters = 0
    train_cfg.runner.eval_student = True
    train_cfg.runner.teacher_experiment_name = None
    train_cfg.runner.init_at_random_ep_len = False
    train_cfg.runner.save_to_wandb = False
    train_cfg.runner.resume = False
    train_cfg.policy.base_policy_path = str(CHECKPOINT)
    train_cfg.policy.nonblind = False
    train_cfg.policy.n_obs_single = cfg.env.n_obs_single
    train_cfg.policy.num_object_observations = 0
    train_cfg.policy.num_future_observations = cfg.env.n_future_obs
    train_cfg.policy.num_future_steps = len(cfg.env.tar_motion_steps_future)
    train_cfg.policy.final_layer_gain = .01
    train_cfg.policy.init_noise_std = .05
    train_cfg.policy.action_std = [.05] * cfg.env.num_actions
    train_cfg.policy.fix_action_std = False
    train_cfg.policy.future_dropout = 0.
    train_cfg.algorithm.dagger_coef = 0.
    train_cfg.algorithm.dagger_coef_min = 0.
    train_cfg.algorithm.num_learning_epochs = 1
    train_cfg.algorithm.num_mini_batches = 1
    train_cfg.algorithm.learning_rate = 1e-4
    train_cfg.algorithm.schedule = 'fixed'
    train_cfg.algorithm.entropy_coef = .001
    train_cfg.algorithm.normalizer_update_iterations = iterations
    if cfg.env.num_observations != 1757 or cfg.env.num_actions != 29:
        raise ValueError('P1 base checkpoint requires 1757 observations and 29 actions')
    return cfg, train_cfg

