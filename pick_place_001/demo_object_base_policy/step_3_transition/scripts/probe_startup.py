"""Bounded real base-policy startup with one free dynamic floorlamp, no training."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,PICK,BASELINE,CHECKPOINT,install,sha,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=2.);p.add_argument('--video',action='store_true');args=p.parse_args();install()
    path=args.motion.resolve(strict=True);out=args.output.resolve()
    if path!=BASELINE.resolve() and ROOT not in path.parents:raise ValueError('Motion must be baseline or step3 candidate')
    if ROOT not in out.parents:raise ValueError('Output outside step3')
    out.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(ROOT/'scripts')]+[str(REPO/x) for x in ['pose','legged_gym','rsl_rl','InterPose/scripts']]
    import isaacgym
    from isaacgym import gymapi,gymtorch,gymutil
    import torch,numpy as np,pickle
    from scipy.spatial.transform import Rotation
    from baseline_config import build_configs
    from physical_floorlamp_env import PhysicalLampEnv
    from foot_reference_loader import install_reference_interpolation,self_test
    from legged_gym.gym_utils.helpers import class_to_dict,set_seed
    from types import SimpleNamespace
    set_seed(1);assert self_test()['status']=='PASS';install_reference_interpolation()
    with path.open('rb') as f:motion=pickle.load(f)
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:objects={k:z[k].copy() for k in z.files}
    inputs=[path,path.with_suffix('.npz'),CHECKPOINT,ROOT/'assets/asset_profile.json',ROOT/'assets/floorlamp_physical.urdf',REPO/'assets/g1/g1_custom_collision_29dof.urdf']
    clockpath=path.parent/'transition_clock.npz'
    if clockpath.exists():inputs.append(clockpath)
    hashes={str(p):sha(p) for p in inputs}
    fps=float(motion['fps']);duration=(len(motion['root_pos'])-1)/fps
    cfg,_=build_configs(1,1);cfg.motion.motion_file=str(path);cfg.motion.object_motion_file=str(path.with_suffix('.npz'))
    cfg.env.num_actors=2;cfg.env.nonblind=False;cfg.env.track_root=False;cfg.env.pose_termination=False;cfg.env.force_full_masking=True;cfg.env.record_video=args.video
    cfg.env.object_asset_root=str(ROOT/'assets');cfg.env.object_urdf_file='floorlamp_physical.urdf';cfg.env.object_obj_file='floorlamp_visual.obj'
    cfg.env.object_initial_linear_velocity=((objects['trans'][1]-objects['trans'][0])*fps).tolist()
    cfg.env.object_initial_angular_velocity=(Rotation.from_matrix(Rotation.from_quat(objects['rot'][1]).as_matrix()@Rotation.from_quat(objects['rot'][0]).as_matrix().T).as_rotvec()*fps).tolist()
    cfg.asset.object_mass=4.;cfg.asset.object_density=12;cfg.rewards.termination_when_object_far=False
    cfg.evaluations=SimpleNamespace(**{k:v for k,v in class_to_dict(cfg.evaluations).items() if k not in ['tracking_root_pose_delta_local','tracking_root_rotation_delta_local']})
    for key in ['randomize_object','randomize_object_pos','randomize_object_scale','randomize_object_friction','randomize_object_mass','randomize_object_com']:setattr(cfg.domain_rand,key,False)
    write_json(out/'env_config.json',class_to_dict(cfg))
    checkpoint_hash=json.loads((PICK/'baseline_manifest.json').read_text())['base_policy']['sha256'];assert sha(CHECKPOINT)==checkpoint_hash
    policy=torch.jit.load(str(CHECKPOINT),map_location='cuda:0').eval()
    for parameter in policy.parameters():parameter.requires_grad_(False)
    def scalar(t):return float(t.detach().cpu())
    def array(t):return t.detach().cpu().numpy().copy()
    class Startup(PhysicalLampEnv):
        def reset_idx(self,ids,motion_ids=None):
            if self.active:self.suppressed_resets+=len(ids);return
            super().reset_idx(ids,motion_ids=motion_ids)
        def check_termination(self):
            super().check_termination()
            if self.active and (not bool(torch.isfinite(self.root_states).all()) or not bool(torch.isfinite(self.dof_state).all())):self.reset_buf[:]=True
        def sample(self):
            value=dict(time_s=int(self.episode_length_buf[0])*self.dt,root_xyz=self.root_states[0,:3].tolist(),root_velocity=self.root_states[0,7:10].tolist(),root_speed_m_s=scalar(torch.linalg.vector_norm(self.root_states[0,7:10])),reference_root_xyz=self._ref_root_pos[0].tolist(),reference_root_velocity=self._ref_root_vel[0].tolist(),reference_joint_velocity_rms_rad_s=scalar(torch.sqrt(self._ref_dof_vel.square().mean())),joint_velocity_rms_rad_s=scalar(torch.sqrt(self.dof_vel.square().mean())),joint_tracking_rms_rad=scalar(torch.sqrt(((self.dof_pos-self._ref_dof_pos)**2).mean())),root_xy_tracking_error_m=scalar(torch.linalg.vector_norm(self.root_states[0,:2]-self._ref_root_pos[0,:2])),root_height_error_m=scalar(abs(self.root_states[0,2]-self._ref_root_pos[0,2])),roll_rad=scalar(self.roll[0]),pitch_rad=scalar(self.pitch[0]),foot_force_z_N=self.contact_forces[0,self.feet_indices,2].tolist(),foot_xy_speed_m_s=torch.linalg.vector_norm(self.rigid_body_states[0,self.feet_indices,7:9],dim=-1).tolist())
            value.update(self.object_diagnostics());return value
    sim=gymapi.SimParams();gymutil.parse_sim_config(class_to_dict(cfg.sim),sim);sim.physx.use_gpu=True;sim.use_gpu_pipeline=True
    env=None;writer=None;started=time.monotonic()
    try:
        env=Startup(cfg,sim,gymapi.SIM_PHYSX,'cuda:0',True);env.suppressed_resets=0;env.bound_steps=int(np.floor((float(env._motion_lib._motion_lengths[0])-1e-5)/env.dt));env.active=True;env.compute_observations();obs=env.get_observations()
        assert obs.shape==(1,1757) and bool(torch.isfinite(obs).all())
        props=env.actual_physical_properties();assert props['actor_count']==2 and props['object_body_flags']==0 and props['object_actor_dof_count']==0
        maximum=min(env.bound_steps,int(round(args.seconds/env.dt)));samples=[];states=[];dofs=[];velocities=[];objstates=[];forces=[];actions=[];observations=[];refroots=[];refdofs=[]
        if args.video:
            import imageio.v2 as imageio
            from PIL import Image,ImageDraw,ImageFont
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22);writer=imageio.get_writer(str(out/'startup_physx.mp4'),fps=1/env.dt,codec='libx264',quality=8,macro_block_size=16)
        def capture():
            assert bool(torch.isfinite(obs).all()) and not bool(env.future_mask.any()) and bool((obs[:,-360:]==0).all())
            samples.append(env.sample());states.append(array(env.root_states[0]));dofs.append(array(env.dof_pos[0]));velocities.append(array(env.dof_vel[0]));objstates.append(array(env.object_root_states[0]));forces.append(array(env.all_contact_forces[0,env.object_body_index]));observations.append(array(obs[0]));refroots.append(array(env._ref_root_pos[0]));refdofs.append(array(env._ref_dof_pos[0]))
            if writer is not None:
                pic=Image.fromarray(env.render_record()[0][:,:,:3].copy());draw=ImageDraw.Draw(pic);draw.rectangle((16,14,1424,92),fill=(245,248,250));draw.text((28,23),'STEP3 STARTUP | BASE POLICY | FREE FLOORLAMP | LIVE PHYSX',font=font,fill=(25,38,53));draw.text((28,56),'t=%.2fs | %s | no residual'%((len(samples)-1)*env.dt,path.parent.name),font=font,fill=(60,75,85));writer.append_data(np.asarray(pic))
        capture();stop='probe_time_complete'
        with torch.inference_mode():
            for index in range(maximum):
                action=policy(obs);assert action.shape==(1,29) and bool(torch.isfinite(action).all());actions.append(array(action[0]));obs,_,_,done,_=env.step(action);capture()
                if bool(done[0]):stop='source_clip_complete' if index+1==env.bound_steps else 'physical_termination';break
        if writer is not None:writer.close();writer=None
        arrays=dict(time_s=np.arange(len(samples))*env.dt,root_states=np.asarray(states),dof_pos=np.asarray(dofs),dof_vel=np.asarray(velocities),object_root_states=np.asarray(objstates),object_contact_forces=np.asarray(forces),actions=np.asarray(actions),observations=np.asarray(observations),reference_root_pos=np.asarray(refroots),reference_dof_pos=np.asarray(refdofs));assert all(np.isfinite(v).all() for v in arrays.values())
        np.savez_compressed(out/'rollout_states.npz',**arrays);write_json(out/'samples.json',samples)
        result=dict(status='COMPLETED_STARTUP_PROBE' if stop=='probe_time_complete' else 'STOPPED',stop_reason=stop,steps=len(actions),simulated_time_s=len(actions)*env.dt,control_dt_s=env.dt,physics_dt_s=cfg.sim.dt,reference_duration_s=duration,motion=str(path),checkpoint=str(CHECKPOINT),checkpoint_sha256=checkpoint_hash,initial=samples[0],final=samples[-1],pre_reset_physics=env.pre_reset_physics_state,adapter=env.local_adapter_diagnostics(),initial_object_floor_correction_m=env.asset_profile['initialization']['upward_floor_correction_m'],actual_physical_properties=props,post_initialization_object_pose_setters=env.object_post_initialization_setter_calls,all_arrays_finite=True,all_future_features_zero=True,physical_stops_retained=True,body_contact_stop_configured=bool(cfg.asset.terminate_after_contacts_on),automatic_resets_suppressed=env.suppressed_resets,input_sha256=hashes,all_inputs_preserved=all(sha(Path(p))==h for p,h in hashes.items()),rollout_sha256=sha(out/'rollout_states.npz'),wall_time_s=time.monotonic()-started,training_executed=False,residual_loaded=False,limitations=['Onefixedseedstartupdiagnostic,notwholecarrysuccess','Bothactorsresettoactualreferenceinitialvelocities; objectzeroonlywhenreferenceholdzero','Resetparentintegrationsretainedandmeasured;posttickvelocityisnotforcedzero','Noobjectforces/weld/supportactor; lampandrobotfreewithgravity','Motionclipendpointandphysicalstopsretained; no tracking-onlystop'])
        assert result['post_initialization_object_pose_setters']==0 and result['all_inputs_preserved'];write_json(out/'result.json',result)
        if args.video:
            reader=imageio.get_reader(str(out/'startup_physx.mp4'));meta=reader.get_meta_data();decoded=sum(1 for frame in reader);reader.close();assert decoded==len(samples)
            write_json(out/'video_validation.json',dict(status='PASS',frames=decoded,fps=meta['fps'],dimensions=meta['size'],sha256=sha(out/'startup_physx.mp4'),scope='LivefreeobjectPhysXbasepolicy,noreferenceforcing'))
        print(json.dumps(dict(status=result['status'],steps=result['steps'],stop_reason=stop,initial_root_speed=result['initial']['root_speed_m_s'],final_root_error=result['final']['root_xy_tracking_error_m'],final_object_tilt=result['final']['object_tilt_deg'])),flush=True)
    finally:
        if writer is not None:writer.close()
        if env is not None:
            if env.viewer is not None:env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
