"""Render one bounded deterministic residual/base rollout; no optimization."""
import argparse,json,sys,time
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,CHECKPOINT,install,sha,write_json,relocate_path,relocate_values

def resolve_checkpoint(explicit=None):
    if explicit is not None:return explicit.expanduser().resolve(strict=True)
    candidates=[]
    state=ROOT/'state/latest_run.txt'
    if state.is_file():candidates.append(Path(relocate_path(state.read_text().strip()))/'progress.json')
    candidates.extend(sorted((ROOT/'runs').glob('*/progress.json'),reverse=True))
    for progress_path in candidates:
        if not progress_path.is_file():continue
        progress=json.loads(progress_path.read_text())
        if 'checkpoint' not in progress:continue
        checkpoint=Path(relocate_path(progress['checkpoint']))
        if ROOT/'runs' in checkpoint.parents and checkpoint.is_file():return checkpoint.resolve()
    raise FileNotFoundError('No completed checkpoint is available. Pass --checkpoint /absolute/path/to/model_N.pt under after_tuning/runs.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--base-only',action='store_true',help='Use the frozen base alone with the same saved task configuration')
    args=p.parse_args()
    if args.checkpoint is not None:args.checkpoint=args.checkpoint.expanduser().resolve()
    install()
    checkpoint=resolve_checkpoint(args.checkpoint);run=checkpoint.parent
    if ROOT/'runs' not in checkpoint.parents:raise ValueError('Checkpoint must be within after_tuning/runs')
    checkpoint=checkpoint.resolve(strict=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    label=('base_only_' if args.base_only else '')+checkpoint.stem
    out=ROOT/'evaluations'/(label+'_'+stamp);out.mkdir(parents=True,exist_ok=False)
    video_path=out/('base_policy_physx.mp4' if args.base_only else 'residual_policy_physx.mp4')
    print('Evaluation output:',out,flush=True)
    sys.path[:0]=[str(ROOT/'scripts')]+[str(REPO/x) for x in ['pose','legged_gym','rsl_rl']]
    import isaacgym
    from isaacgym import gymapi,gymutil
    import torch,numpy as np
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from residual_env import build_configs,ResidualFloorlampEnv
    from legged_gym.gym_utils.helpers import class_to_dict,set_seed
    from rsl_rl.runners import OnPolicyDaggerRunner
    provenance=json.loads((run/'provenance.json').read_text())
    for path,digest in provenance['input_sha256'].items():
        current=relocate_path(path)
        if sha(current)!=digest:raise ValueError('Training input changed: '+current)
    cfg,_=build_configs(1,1)
    def restore(target,values):
        if isinstance(target,dict):
            target.clear();target.update(values);return
        for key,value in values.items():
            if isinstance(value,dict):restore(getattr(target,key),value)
            else:setattr(target,key,value)
    restore(cfg,relocate_values(json.loads((run/'env_config.json').read_text())))
    cfg.env.num_envs=1;cfg.env.record_video=True
    train=relocate_values(json.loads((run/'train_config.json').read_text()));train['env']['num_envs']=1
    train['policy']['base_policy_path']=str(CHECKPOINT)
    set_seed(cfg.seed)
    motion=Path(cfg.motion.motion_file);object_file=Path(cfg.motion.object_motion_file)
    input_hashes={str(q):sha(q) for q in [checkpoint,run/'env_config.json',run/'train_config.json',motion,object_file,CHECKPOINT]}
    with np.load(motion.parent/'transition_clock.npz',allow_pickle=False) as z:clock={k:z[k].copy() for k in z.files}
    class EvaluationEnv(ResidualFloorlampEnv):
        evaluation_started=False
        def reset_idx(self,ids,motion_ids=None):
            if self.evaluation_started:
                self.terminal_reset_requests+=len(ids)
                return  # Retain terminal state for the final frame, then stop.
            super().reset_idx(ids,motion_ids=motion_ids)
    sim=gymapi.SimParams();gymutil.parse_sim_config(class_to_dict(cfg.sim),sim);sim.physx.use_gpu=True;sim.use_gpu_pipeline=True
    env=None;writer=None;started=time.monotonic()
    try:
        env=EvaluationEnv(cfg,sim,gymapi.SIM_PHYSX,'cuda:0',True)
        runner=OnPolicyDaggerRunner(env,train,log_dir=None,device='cuda:0')
        runner.load(str(checkpoint),load_optimizer=False)
        runner.alg.actor_critic.eval();runner.base_policy.eval()
        def forbidden(*a,**kw):raise RuntimeError('Optimization is forbidden in rendering')
        runner.learn=forbidden;runner.alg.update=forbidden;runner.alg.optimizer.step=forbidden
        env.terminal_reset_requests=0;env.evaluation_started=True
        initial_setters=env.object_pose_setter_calls
        maximum=int(np.floor((float(env._motion_lib._motion_lengths[0])-1e-5)/env.dt))
        obs=env.get_observations();samples=[];root=[];obj=[];dofs=[];acts=[];residuals=[];bases=[];observations=[]
        writer=imageio.get_writer(str(video_path),fps=1/env.dt,codec='libx264',quality=8,macro_block_size=16)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
        def arr(t):return t.detach().cpu().numpy().copy()
        def capture():
            assert obs.shape==(1,1764) and bool(torch.isfinite(obs).all()) and not bool(env.future_mask.any())
            elapsed=len(samples)*env.dt
            phase=float(np.interp(elapsed,clock['control_time_s'],clock['source_reference_time_s']))
            s=dict(time_s=elapsed,source_time_s=phase,root_xyz=env.root_states[0,:3].tolist(),root_xy_error_m=float(torch.linalg.vector_norm(env.root_states[0,:2]-env._ref_root_pos[0,:2])),
                   joint_tracking_rms_rad=float(torch.sqrt(((env.dof_pos-env._ref_dof_pos)**2).mean())),foot_force_z_N=env.contact_forces[0,env.feet_indices,2].tolist(),**env.object_diagnostics())
            samples.append(s);root.append(arr(env.root_states[0]));obj.append(arr(env.object_root_states[0]));dofs.append(arr(env.dof_pos[0]));observations.append(arr(obs[0]))
            picture=Image.fromarray(env.render_record()[0][:,:,:3].copy());draw=ImageDraw.Draw(picture)
            title='FROZEN BASE ONLY' if args.base_only else 'RESIDUAL %d + FROZEN BASE'%runner.current_learning_iteration
            draw.rectangle((12,12,1428,110),fill=(245,248,250));draw.text((26,20),title+' | FREE FLOORLAMP | LIVE PHYSX',font=font,fill=(25,38,53))
            draw.text((26,53),'elapsed %.2fs | source %.3fs | deterministic evaluation'%(elapsed,phase),font=font,fill=(50,65,80))
            draw.text((26,82),'Root error %.3fm | lamp tilt %.1fdeg'%(s['root_xy_error_m'],s['object_tilt_deg']),fill=(50,65,80))
            writer.append_data(np.asarray(picture))
            if len(samples)-1 in [0,maximum//2,maximum]:picture.save(out/('frame_%04d.jpg'%(len(samples)-1)),quality=94)
        capture();stop='bounded_clip_complete'
        with torch.no_grad():
            for index in range(maximum):
                critic=env.get_privileged_observations()
                normalized,_=runner._normalize_observations(obs,critic)
                base=runner.base_policy(runner.full_obs_to_base_obs(obs))
                residual=torch.zeros_like(base) if args.base_only else runner.alg.actor_critic.act_inference(normalized)
                action=base+residual
                if not bool(torch.isfinite(action).all()):raise FloatingPointError('Nonfinite action')
                bases.append(arr(base[0]));residuals.append(arr(residual[0]));acts.append(arr(action[0]))
                obs,_,_,done,_=env.step(action);capture()
                if bool(done[0]):
                    stop='bounded_clip_complete' if index+1==maximum else 'physical_or_pose_termination'
                    break
        writer.close();writer=None
        arrays=dict(root_states=np.asarray(root),object_root_states=np.asarray(obj),dof_pos=np.asarray(dofs),actions=np.asarray(acts),base_actions=np.asarray(bases),residual_actions=np.asarray(residuals),observations=np.asarray(observations))
        assert all(np.isfinite(v).all() for v in arrays.values())
        np.savez_compressed(out/'rollout_states.npz',**arrays);write_json(out/'samples.json',samples)
        reader=imageio.get_reader(str(video_path));meta=reader.get_meta_data();frames=sum(1 for _ in reader);reader.close();assert frames==len(samples)
        final=samples[-1];all_inputs_unchanged=all(sha(k)==v for k,v in input_hashes.items())
        result=dict(status='EVALUATION_COMPLETE',checkpoint=str(checkpoint),completed_training_iterations=runner.current_learning_iteration,stop_reason=stop,
                    simulated_time_s=(len(samples)-1)*env.dt,exported_reference_duration_s=float(env._motion_lib._motion_lengths[0]),steps=len(acts),initial=samples[0],final=final,
                    root_xy_error_mean_m=float(np.mean([s['root_xy_error_m'] for s in samples])),object_position_error_mean_m=float(np.mean([s['object_position_error_m'] for s in samples])),
                    max_object_floor_gap_m=max(s['object_floor_gap_m'] for s in samples),supported_airborne_proxy_samples=sum(s['robot_supported_airborne_evidence'] for s in samples),
                    frames=frames,fps=meta['fps'],video_sha256=sha(video_path),input_sha256=input_hashes,all_inputs_unchanged=all_inputs_unchanged,
                    object_pose_setters_during_rollout=env.object_pose_setter_calls-initial_setters,actual_physical_properties=env.actual_physical_properties(),
                    optimizer_updates=0,training_executed=False,deterministic_actions=True,base_only=args.base_only,residual_actions_enabled=not args.base_only,checkpoint_normalizers_loaded=True,wall_time_s=time.monotonic()-started,
                    limitations=['One-seed deterministic evaluation, not robust performance certification.','Bounded at the last complete control tick below the reference endpoint.','Supported-airborne is a heuristic, not direct proof of hand force closure.','Early termination retains the terminal frame and performs no automatic restart.'])
        assert all_inputs_unchanged and result['object_pose_setters_during_rollout']==0
        write_json(out/'result.json',result);write_json(ROOT/'reports/latest_evaluation.json',dict(output=str(out),video=str(video_path),result=str(out/'result.json')))
        print(json.dumps(result,indent=2),flush=True)
    finally:
        if writer is not None:writer.close()
        if env is not None:
            if env.viewer is not None:env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
