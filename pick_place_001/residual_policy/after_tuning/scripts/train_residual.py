"""Residual PPO with the tuned Step3 reference, a free floorlamp and frozen base.

--check-only validates files/config on CPU; preflight constructs the GPU task
without optimizer updates; train performs real PPO optimization when requested.
"""
import argparse,json,math,signal,sys,time
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,MOTION,OBJECT,CHECKPOINT,ASSETS,install,sha,write_json

def positive(v):
    n=int(v)
    if n<1:raise argparse.ArgumentTypeError('must be a positive integer')
    return n

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',nargs='?',choices=['train','preflight'],default='preflight')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--num-envs',type=positive,default=32)
    p.add_argument('--iterations',type=positive,default=5000)
    p.add_argument('--num-mini-batches',type=positive)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--save-interval',type=positive,default=100)
    p.add_argument('--output',type=Path)
    return p.parse_args()

class StopRequested(Exception):pass

def main():
    args=parse_args();install()
    sys.path[:0]=[str(ROOT/'scripts')]+[str(REPO/x) for x in ['pose','legged_gym','rsl_rl']]
    import isaacgym  # SDK requires import before Torch.
    from isaacgym import gymapi,gymutil
    import torch,numpy as np,pickle
    from residual_env import build_configs,ResidualFloorlampEnv
    from foot_reference_loader import self_test,install_reference_interpolation
    from legged_gym.gym_utils.helpers import class_to_dict,set_seed
    from rsl_rl.runners import OnPolicyDaggerRunner
    assert self_test()['status']=='PASS';install_reference_interpolation()
    expected={MOTION:'4f5597f12544d8d93546c1f12848a5f87ad24638088c58a07aeb03b2a6683849',OBJECT:'04f42fe7f1876a7a7db4f4f22f199cade7a2eabac5b4bf38f5b411b04d967c58',CHECKPOINT:'fd98a2af49b8c1b3e39a57b503272db0dacc0d56c7e68d1c8dddfbd9cb9d931c'}
    for p,h in expected.items():
        if not p.is_file() or sha(p)!=h:raise ValueError('Missing or changed input: '+str(p))
    with MOTION.open('rb') as f:motion=pickle.load(f)
    with np.load(OBJECT,allow_pickle=False) as z:objects={k:z[k].copy() for k in z.files}
    fps=float(motion['fps']);duration=(len(motion['root_pos'])-1)/fps
    for array in [motion['root_pos'],motion['root_rot'],motion['dof_pos'],objects['trans'],objects['rot']]:
        if not np.isfinite(array).all() or not (array[:361]==array[0]).all():raise ValueError('Reference must contain exact synchronized0.75s stationary hold')
    if fps!=480 or len(motion['root_pos'])!=2601:raise ValueError('Unexpected selected reference timing')
    cfg,training=build_configs(args.num_envs,args.iterations,args.save_interval)
    cfg.seed=training.seed=args.seed
    batch=args.num_envs*training.runner.num_steps_per_env
    mini=args.num_mini_batches or math.ceil(batch/256)
    if args.num_mini_batches is None:
        while batch%mini:mini+=1
    if mini>batch or batch%mini:raise ValueError('--num-mini-batches must divide num_envs*24 without remainder')
    training.algorithm.num_mini_batches=mini
    config_summary=dict(status='PASS_INPUT_CONFIG_CHECK',motion=str(MOTION),object_motion=str(OBJECT),base_checkpoint=str(CHECKPOINT),
        input_sha256={str(p):h for p,h in expected.items()},frames=2601,fps=fps,duration_s=duration,num_envs=args.num_envs,iterations=args.iterations,
        rollout_batch=batch,num_mini_batches=mini,samples_per_minibatch=batch//mini,actor_observations=1764,base_observations=1757,critic_observations=1741,
        object_features=7,object_reward_weight=2.,actors_per_env=2,timeout_metadata='fresh_per_transition_and_terminal_masked',simulator_executed=False,optimizer_updates=0,training_executed=False)
    if args.check_only:
        write_json(ROOT/'reports/input_config_check.json',config_summary);print(json.dumps(config_summary,indent=2));return
    set_seed(args.seed)
    out=(args.output or ROOT/'runs'/('manual_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))).resolve()
    if ROOT/'runs' not in out.parents:raise ValueError('Output must be a new run under '+str(ROOT/'runs'))
    if out.exists() and any(p.name!='training.log' or not p.is_file() for p in out.iterdir()):raise FileExistsError('Refusing to overwrite existing run: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'env_config.json',class_to_dict(cfg));train_dict=class_to_dict(training);write_json(out/'train_config.json',train_dict)
    provenance=dict(config_summary,mode=args.mode,created_at_utc=datetime.now(timezone.utc).isoformat(),scripts_sha256={str(p):sha(p) for p in (ROOT/'scripts').glob('*.py')},
        asset_sha256={str(p):sha(p) for p in ASSETS.iterdir() if p.is_file()},python=sys.version,torch=torch.__version__,numpy=np.__version__,
        reference_has_pregrasped_fixed_hands=True,tracking_success_verified=False)
    write_json(out/'provenance.json',provenance)
    print('Output:',out,flush=True);print('PPO batch=%d, minibatches=%d, samples/minibatch=%d'%(batch,mini,batch//mini),flush=True)
    stop_requested=[False]
    def request_stop(*unused):
        stop_requested[0]=True;print('Stop requested; saving after current complete PPO iteration.',flush=True)
    for signum in [signal.SIGTERM,signal.SIGINT]:signal.signal(signum,request_stop)
    import wandb
    run=wandb.init(project='pick-place-after-tuning',name=out.name,mode='disabled',dir=str(out))
    env=None;runner=None;updates=0;started=time.monotonic()
    def digest(module):
        import hashlib
        h=hashlib.sha256()
        for name,v in sorted(module.state_dict().items()):h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    try:
        sim=gymapi.SimParams();gymutil.parse_sim_config(class_to_dict(cfg.sim),sim);sim.physx.use_gpu=True;sim.use_gpu_pipeline=True
        env=ResidualFloorlampEnv(cfg,sim,gymapi.SIM_PHYSX,'cuda:0',True)
        provenance['simulator_executed']=True;write_json(out/'provenance.json',provenance)
        runner=OnPolicyDaggerRunner(env,train_dict,log_dir=str(out),device='cuda:0')
        actor=runner.alg.actor_critic
        raw=env.get_observations();critic=env.get_privileged_observations();base_obs=runner.full_obs_to_base_obs(raw)
        start=cfg.env.n_obs_single*(cfg.env.history_len+1)
        expected_base=torch.cat((raw[:,:start],raw[:,start+7:]),dim=-1)
        opt={id(p) for g in runner.alg.optimizer.param_groups for p in g['params']}
        forbidden={id(p) for p in runner.base_policy.parameters()}|{id(p) for p in runner.teacher_actor_critic.parameters()}
        checks=dict(observation_layout=raw.shape==(args.num_envs,1764) and critic.shape==(args.num_envs,1741),base_layout=base_obs.shape==(args.num_envs,1757) and torch.equal(base_obs,expected_base),
                    frozen_base=all(not p.requires_grad for p in runner.base_policy.parameters()),optimizer_excludes_base_teacher=not bool(opt&forbidden),
                    residual_actor_optimized=all(id(p) in opt for p in actor.actor.parameters() if p.requires_grad),no_teacher=not runner.teacher_loaded and runner.alg.dagger_coef==0,
                    future_masked=not bool(env.future_mask.any()) and bool((raw[:,-360:]==0).all()),no_stale_episode_report='episode' not in env.extras,
                    initial_finite=bool(torch.isfinite(raw).all() and torch.isfinite(critic).all()),free_floorlamp=env.actual_physical_properties()['object_body_flags']==0)
        with torch.no_grad():
            norm,normcritic=runner._normalize_observations(raw,critic)
            action=runner.base_policy(base_obs)+actor.act_inference(norm)
            value=actor.evaluate(normcritic)
            checks['forward_finite']=bool(torch.isfinite(action).all() and torch.isfinite(value).all())
            reward=env._reward_tracking_object_point_cloud();checks['object_reward_finite']=bool(torch.isfinite(reward).all())
        checks={k:bool(v) for k,v in checks.items()};write_json(out/'initialization_checks.json',checks)
        if not all(checks.values()):raise RuntimeError('Initialization checks failed: '+str(checks))
        base_before=digest(runner.base_policy);actor_before=digest(actor.actor)
        runner.save(str(out/'initialization_untrained.pt'),infos={'untrained':True,'updates':0})
        if args.mode=='preflight':
            # Deliberately no learn/update/optimizer.step calls in this branch.
            with torch.no_grad():env.step(action)
            checks['one_step_finite']=bool(torch.isfinite(env.get_observations()).all())
            ids=torch.tensor([args.num_envs-1],device=env.device,dtype=torch.long);env.reset_idx(ids);env.compute_observations()
            checks['subset_reset_finite']=bool(torch.isfinite(env.get_observations()).all())
            checks['object_refs_translated']=bool(torch.allclose(env._ref_object_root_pos-env.episode_init_origin,env._motion_lib.calc_hoi_motion_frame(env._motion_ids.clone(),env._get_motion_times().clone())[9],atol=2e-6,rtol=0))
            checks['no_optimizer_state']=not bool(runner.alg.optimizer.state)
            checks['base_unchanged']=digest(runner.base_policy)==base_before
            write_json(out/'preflight.json',dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,training_executed=False,optimizer_updates=0,diagnostics=env.training_diagnostics()))
            if not all(checks.values()):raise RuntimeError('Preflight failed')
            return
        original_update=runner.alg.update;original_log=runner.log
        def checked_update():
            nonlocal updates
            losses=original_update()
            if not all(math.isfinite(float(v)) for v in losses):raise FloatingPointError('Nonfinite PPO loss')
            if not all(bool(torch.isfinite(p).all()) for p in actor.parameters()):raise FloatingPointError('Nonfinite residual parameters')
            if any(p.grad is not None for p in runner.base_policy.parameters()):raise RuntimeError('Frozen base received gradient')
            updates+=1
            with (out/'updates.jsonl').open('a') as stream:stream.write(json.dumps(dict(update=updates,losses=[float(v) for v in losses],peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))+'\n')
            return losses
        def checked_log(locs,*a,**kw):
            original_log(locs,*a,**kw)
            for k,v in env.extras.get('object_metrics',{}).items():runner.writer.add_scalar('Object/'+k,v,global_step=locs['it'])
            runner.writer.add_scalar('Memory/peak_cuda_allocated_MiB',torch.cuda.max_memory_allocated()/2**20,global_step=locs['it']);runner.writer.flush()
            write_json(out/'progress.json',dict(status='RUNNING',iteration=runner.current_learning_iteration,requested_iterations=args.iterations,optimizer_updates=updates,elapsed_s=time.monotonic()-started,object_metrics=env.extras.get('object_metrics',{})))
            if stop_requested[0]:raise StopRequested()
        runner.alg.update=checked_update;runner.log=checked_log
        provenance['training_executed']=True;provenance['simulator_executed']=True;write_json(out/'provenance.json',provenance)
        try:
            runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        except StopRequested:
            path=out/('model_%d_interrupted.pt'%runner.current_learning_iteration)
            runner.save(str(path),infos={'interrupted':True,'completed_iterations':runner.current_learning_iteration})
            write_json(out/'interrupted.json',dict(status='STOPPED_AT_ITERATION_BOUNDARY',checkpoint=str(path),optimizer_updates=updates,base_unchanged=digest(runner.base_policy)==base_before))
            write_json(out/'progress.json',dict(status='INTERRUPTED',iteration=runner.current_learning_iteration,requested_iterations=args.iterations,optimizer_updates=updates,checkpoint=str(path)))
            return
        final=out/('model_%d.pt'%runner.current_learning_iteration)
        final_checks=dict(updates_completed=updates==args.iterations,checkpoint_saved=final.is_file(),base_unchanged=digest(runner.base_policy)==base_before,
                          residual_actor_changed=digest(actor.actor)!=actor_before,inputs_preserved=all(sha(p)==h for p,h in expected.items()))
        write_json(out/'training_validation.json',dict(status='PASS_TRAINING_COMPLETION' if all(final_checks.values()) else 'FAIL',checks=final_checks,final_checkpoint=str(final),optimizer_updates=updates,tracking_success_verified=False,diagnostics=env.training_diagnostics()))
        if not all(final_checks.values()):raise RuntimeError('Training completion validation failed')
        write_json(out/'progress.json',dict(status='COMPLETED',iteration=runner.current_learning_iteration,requested_iterations=args.iterations,optimizer_updates=updates,checkpoint=str(final)))
        print('Training complete:',final,flush=True)
    except BaseException as exc:
        write_json(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),optimizer_updates=updates))
        write_json(out/'progress.json',dict(status='FAILED',iteration=runner.current_learning_iteration if runner is not None else 0,requested_iterations=args.iterations,optimizer_updates=updates,error=str(exc)))
        raise
    finally:
        if runner is not None and runner.writer is not None:runner.writer.close();runner.writer=None
        if env is not None:env.gym.destroy_sim(env.sim)
        run.finish()
        provenance['optimizer_updates']=updates
        provenance['elapsed_s']=time.monotonic()-started
        write_json(out/'provenance.json',provenance)

if __name__=='__main__':main()
