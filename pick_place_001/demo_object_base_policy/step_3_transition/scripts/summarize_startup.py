"""Audit saved bounded PhysX runs and compare wall time and source progress."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,BASELINE,CHECKPOINT,install,sha,write_json

def main():
    install()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    runs={};checks={}
    metrics=['root_xy_tracking_error_m','joint_tracking_rms_rad','object_position_error_m','object_tilt_deg']
    def values(samples):
        return {k:np.asarray([s[k] for s in samples]) for k in metrics}
    for tag in ['probe_baseline','probe_A','probe_B','full_baseline','full_A','full_B']:
        folder=ROOT/'results'/tag
        result=json.loads((folder/'result.json').read_text());samples=json.loads((folder/'samples.json').read_text())
        with np.load(folder/'rollout_states.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        times=arrays['time_s'];motion=Path(result['motion']);source=times.copy()
        if (motion.parent/'transition_clock.npz').exists():
            with np.load(motion.parent/'transition_clock.npz',allow_pickle=False) as c:source=np.interp(times,c['control_time_s'],c['source_reference_time_s'])
        c=dict(finite=all(np.isfinite(a).all() for a in arrays.values()),samples_match=len(samples)==len(times)==len(arrays['actions'])+1,
               checkpoint_matches=sha(CHECKPOINT)==result['checkpoint_sha256'],inputs_preserved=all(sha(Path(p))==h for p,h in result['input_sha256'].items()),
               future_features_masked=bool((arrays['observations'][:,-360:]==0).all()),observations_shape=arrays['observations'].shape==(len(times),1757),
               free_object=result['actual_physical_properties']['object_body_flags']==0 and not result['actual_physical_properties']['asset_options']['fix_base_link'] and not result['actual_physical_properties']['asset_options']['disable_gravity'],
               two_actors=result['actual_physical_properties']['actor_count']==2,zero_post_init_pose_setters=result['post_initialization_object_pose_setters']==0,
               saved_root_matches_samples=bool(np.allclose(arrays['root_states'][:,:3],[s['root_xyz'] for s in samples])),
               physical_stops_retained=result['physical_stops_retained'],training_disabled=not result['training_executed'] and not result['residual_loaded'])
        if not tag.endswith('baseline'):
            pre=result['pre_reset_physics']
            c.update(pre_physics_robot_velocities_zero=bool((np.asarray(pre['robot_root'][7:13])==0).all()),pre_physics_object_velocities_zero=bool((np.asarray(pre['object_root'][7:13])==0).all()),
                     joint_reset_matches_reference=pre['joint_velocity_max_error_rad_s']==0 and pre['joint_position_max_error_rad']==0,
                     reference_hold_velocity_zero=all(s['reference_joint_velocity_rms_rad_s']==0 and np.linalg.norm(s['reference_root_velocity'])==0 for s in samples if s['time_s']<.75))
        reader=imageio.get_reader(str(folder/'startup_physx.mp4'));count=sum(1 for _ in reader);reader.close();c['video_frames_match']=count==len(times)
        checks[tag]=c;assert all(c.values()),(tag,c)
        rows=[]
        for target in ([.1,.5,.74,1.,1.74,2.] if tag.startswith('probe') else [.1,.74,1.74,2.,3.,4.,5.]):
            if target>times[-1]+1e-6:continue
            i=int(np.argmin(abs(times-target)));rows.append(dict(control_time_s=float(times[i]),source_reference_time_s=float(source[i]),**{k:float(samples[i][k]) for k in metrics}))
        summary=dict(status=result['status'],stop_reason=result['stop_reason'],steps=result['steps'],control_endpoint_s=float(times[-1]),exported_duration_s=result['reference_duration_s'],unqueried_tail_s=result['reference_duration_s']-float(times[-1]) if result['stop_reason']=='source_clip_complete' else None,
                     initial_root_speed_m_s=result['initial']['root_speed_m_s'],initial_joint_velocity_rms_rad_s=result['initial']['joint_velocity_rms_rad_s'],
                     initial_root_position_error_m=result['adapter']['last_reset_measurement']['robot_root_reference_error_m'],
                     initial_joint_position_rms_error_rad=result['initial']['joint_tracking_rms_rad'],final={k:result['final'][k] for k in metrics},
                     supported_airborne_proxy_samples=sum(s['robot_supported_airborne_evidence'] for s in samples),samples=len(samples),selected_wall_time_rows=rows,
                     input_result_sha256=sha(folder/'result.json'),rollout_sha256=sha(folder/'rollout_states.npz'))
        runs[tag]=dict(summary=summary,times=times,source=source,samples=samples,metrics=values(samples))
    equal_phase=[]
    for target in [.1,.25,.5,.75,1.,1.5,2.,3.,4.]:
        row=dict(requested_source_reference_time_s=target,runs={})
        for tag in ['full_baseline','full_A','full_B']:
            r=runs[tag];distance=abs(r['source']-target)
            # A repeated hold maps many physical samples to the same source phase.
            # Use the latest equal-distance tick, not the artificially exact reset.
            i=int(np.flatnonzero(distance<=distance.min()+1e-12)[-1])
            row['runs'][tag]=dict(actual_control_time_s=float(r['times'][i]),actual_source_reference_time_s=float(r['source'][i]),phase_error_s=float(r['source'][i]-target),**{k:float(r['metrics'][k][i]) for k in metrics})
        equal_phase.append(row)
    report=dict(status='PASS_ARTIFACT_AND_STARTUP_CONSISTENCY_CHECKS',checks=checks,runs={tag:r['summary'] for tag,r in runs.items()},equal_source_progress=equal_phase,
                selected_candidate='A_frame0000',selection_scope='Initial-transition improvement; physical carry remains unsuccessful',
                rationale=['Full original source interval retained; candidate B crops0.1s.','A has lower hold-phase joint error and object drift than B.','Both A and B pass dense geometry and actual loader validation.','A and B both fail full carrying despite improved startup.'],
                comparison_method='Wall-time results include different reference progress; equal-source rows use nearest actual 50Hz sample (usually up to 10ms phase discrepancy). Equal-distance repeated hold phases use the latest tick, not the initial reset. All actual phases recorded. One seed, not statistical robustness.',
                reset_interpretation='Zero reference/pre-physics robot and object velocities; post-integration physical states may have impulses. Two constructor reset integrations have state rewritten between them, not continuous4ms drift.',
                airborne_proxy='Floor gap>20mm, object upward net contact>=19.62N and tilt<20deg; this proxy does not identify individual grasp forces or prove wrench closure.',
                endpoint_interpretation='source_clip_complete means last complete control tick below endpoint, not reaching exact exported final pose. Raw STOPPED status also denotes this normal bounded endpoint; see stop_reason.',
                baseline_interpretation='Current baseline uses actual object reference initial linear/angular velocity, unlike historical step0 which zeroed object velocity. All current probes use the same reset rule.',
                limitations=['No successful free-object lift/carry.','Reference is pregrasped; no approach or active gripper.','Assumed4kg object and conservative convex colliders.','Full future features are masked, matching this base checkpoint usage.','Reference geometry validation does not imply physical grasp success.'])
    write_json(ROOT/'reports/startup_comparison.json',report)
    colors={'full_baseline':'#b94b43','full_A':'#2375a6','full_B':'#a57c16'}
    fig,axes=plt.subplots(4,2,figsize=(13,12),constrained_layout=True)
    labels=['Robot root XY error (m)','Joint tracking RMS (rad)','Lamp position error (m)','Lamp tilt (deg)']
    for row,(key,label) in enumerate(zip(metrics,labels)):
        for tag,color in colors.items():
            r=runs[tag]
            axes[row,0].plot(r['times'],r['metrics'][key],label=tag,color=color)
            axes[row,1].plot(r['source'],r['metrics'][key],label=tag,color=color)
        for ax in axes[row]:ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[0,0].set_title('Actual elapsed time: preparation delays source progress');axes[0,1].set_title('Mapped source progress: same path stage')
    axes[-1,0].set_xlabel('Control time (s)');axes[-1,1].set_xlabel('Original reference time (s)');axes[0,0].legend()
    fig.suptitle('Unchanged base policy + free floorlamp | seed1 | startup improves, carry fails')
    fig.savefig(ROOT/'reports/startup_comparison.png',dpi=160);fig.savefig(ROOT/'reports/startup_comparison.pdf');plt.close(fig)
    videos=ROOT/'videos';videos.mkdir(exist_ok=True)
    readers=[imageio.get_reader(str(ROOT/'results'/tag/'startup_physx.mp4')) for tag in ['probe_baseline','probe_A','probe_B']]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    writer=imageio.get_writer(str(videos/'startup_comparison.mp4'),fps=50,codec='libx264',quality=8,macro_block_size=16)
    names=['BEFORE | latest step2','A | full path + hold/ramp','B | crop0.1s + hold/ramp'];tags=['probe_baseline','probe_A','probe_B']
    try:
        for i in range(101):
            canvas=Image.new('RGB',(2160,560),(245,248,250));draw=ImageDraw.Draw(canvas)
            for j,(reader,tag,name) in enumerate(zip(readers,tags,names)):
                pic=Image.fromarray(reader.get_data(i)).resize((720,480));canvas.paste(pic,(720*j,80));r=runs[tag]
                draw.text((720*j+14,8),name,font=font,fill=(25,38,53));draw.text((720*j+14,38),'elapsed %.2fs | source %.3fs'%(r['times'][i],r['source'][i]),font=font,fill=(60,75,85))
            writer.append_data(np.asarray(canvas))
            if i in [0,37,100]:canvas.save(videos/('startup_comparison_%03d.jpg'%i),quality=94)
    finally:
        writer.close()
        for reader in readers:reader.close()
    check=imageio.get_reader(str(videos/'startup_comparison.mp4'));count=sum(1 for _ in check);check.close();assert count==101
    write_json(videos/'comparison_validation.json',dict(status='PASS',frames=count,fps=50,dimensions=[2160,560],sha256=sha(videos/'startup_comparison.mp4'),scope='Actual PhysX policy rollouts at equal elapsed time; source progress explicitly labeled, not matched across columns.'))
    print('PASS six saved rollout checks, source-phase comparison, plot and live comparison video',flush=True)

if __name__=='__main__':main()
