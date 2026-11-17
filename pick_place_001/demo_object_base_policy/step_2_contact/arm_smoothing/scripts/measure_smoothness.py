"""Independent sampled arm-motion roughness metrics; no dynamics/vibration claim."""
import argparse,json,pickle,sys,xml.etree.ElementTree as ET,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,BASELINE,URDF,install,sha,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path);p.add_argument('--tag',default=None);args=p.parse_args();install()
    import numpy as np
    from scipy.signal import detrend
    from scipy.spatial.transform import Rotation
    sys.path.insert(1,str(REPO/'legged_gym/legged_gym/scripts'))
    from convert_gmr_qpos import forward_kinematics
    with BASELINE.open('rb') as f:baseline=pickle.load(f)
    names=[j.get('name') for j in ET.parse(URDF).getroot().findall('joint') if j.get('type')=='revolute'];armids=np.arange(15,29);armnames=[names[i] for i in armids]
    dt=1/float(baseline['fps']);n=len(baseline['dof_pos']);duration=(n-1)*dt
    def summary(x):
        a=np.abs(np.asarray(x));return dict(count=int(a.size),rms=float(np.sqrt(np.mean(a*a))),p95_abs=float(np.percentile(a,95)),maximum_abs=float(a.max())) if a.size else None
    def spectrum(q):
        w=np.hanning(len(q));signal=detrend(q,axis=0,type='linear')*w[:,None];Y=np.fft.rfft(signal,axis=0);f=np.fft.rfftfreq(len(q),dt);weights=np.full(len(f),2.);weights[0]=1.
        if len(q)%2==0:weights[-1]=1.
        energy=np.abs(Y)**2*weights[:,None]/(len(q)**2*np.mean(w*w))
        return f,energy
    def measure(m):
        q=m['dof_pos'][:,armids];arrays={'position':q,'velocity':np.diff(q,axis=0)/dt,'acceleration':np.diff(q,n=2,axis=0)/dt**2,'jerk':np.diff(q,n=3,axis=0)/dt**3}
        times={'position':np.arange(n)*dt,'velocity':(np.arange(n-1)+.5)*dt,'acceleration':(np.arange(n-2)+1)*dt,'jerk':(np.arange(n-3)+1.5)*dt}
        frequencies,energy=spectrum(q);metrics=dict(aggregate={key:summary(value) for key,value in arrays.items()},frequency_energy=dict(method='Linear detrend, Hann taper, one-sided Parseval energy with mean-window-square normalization; angle units rad^2',total_rad2=float(energy.sum()),above10Hz_rad2=float(energy[frequencies>=10].sum()),above20Hz_rad2=float(energy[frequencies>=20].sum()),above40Hz_rad2=float(energy[frequencies>=40].sum())),per_joint=[],regions={})
        for index,name in enumerate(armnames):
            row=dict(joint=name)
            for key in ['velocity','acceleration','jerk']:
                v=arrays[key][:,index];peak=int(np.argmax(abs(v)));row[key]=summary(v);row[key]['peak_time_s']=float(times[key][peak])
            row['above10Hz_rad2']=float(energy[frequencies>=10,index].sum());metrics['per_joint'].append(row)
        for region,(start,stop) in dict(full=(0,duration),start=(0,.5),lift_transition=(1.5,2.5),walking_transition=(2,3),end=(3.7,duration)).items():
            metrics['regions'][region]={'bounds_s':[start,stop],**{key:summary(value[(times[key]>=start)&(times[key]<=stop)]) for key,value in arrays.items() if key!='position'}}
        local,_=forward_kinematics(ET.parse(URDF).getroot(),m['dof_pos'],names,m['link_body_list'])
        world=m['root_pos'][:,None,:]+np.einsum('tij,tkj->tki',Rotation.from_quat(m['root_rot']).as_matrix(),local)
        marker_metrics={}
        for link in ['left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link']:
            position=world[:,list(m['link_body_list']).index(link)]
            _,markerenergy=spectrum(position)
            marker_metrics[link]=dict(acceleration_m_s2=summary(np.linalg.norm(np.diff(position,n=2,axis=0)/dt**2,axis=1)),jerk_m_s3=summary(np.linalg.norm(np.diff(position,n=3,axis=0)/dt**3,axis=1)),position_energy_above10Hz_m2=float(markerenergy[frequencies>=10].sum()))
        metrics['world_link_marker_metrics']=marker_metrics
        return metrics,arrays,times,frequencies,energy
    before,barr,times,freq,benergy=measure(baseline);tag=args.tag or (args.candidate.parent.name if args.candidate else 'baseline')
    report=dict(status='BASELINE_MEASURED',baseline=str(BASELINE),fps=float(baseline['fps']),frames=n,duration_s=duration,arm_joint_names=armnames,baseline_metrics=before,metric_definitions=dict(velocity='Forward first difference q[i+1]-q[i], divided bydt; samples at intervalmidpoints',acceleration='Second difference q[i+2]-2q[i+1]+q[i], divided bydt²',jerk='Third difference divided bydt³; sampled roughness, not continuous jerk across the piecewise-linear loader knots',spectral='Hann-tapered detrended joint ANGLE energy; >=10Hz and additional>=20/40Hz bands, all14arms; no hard physiological threshold'),input_sha256={str(BASELINE):sha(BASELINE),str(BASELINE.with_suffix('.npz')):sha(BASELINE.with_suffix('.npz'))},limitations=['These metrics characterize stored reference roughness, not measured motor vibration or physical grasp success.','No hardware acceleration/jerk limits are asserted; URDF only provides velocity limits.','Derivative peaks near active constraints can change even if RMS improves; compare all aggregates and per-joint metrics.'])
    after=None
    if args.candidate:
        path=args.candidate.resolve(strict=True)
        if ROOT not in path.parents:raise ValueError('Candidate must be inside arm_smoothing')
        with path.open('rb') as f:new=pickle.load(f)
        assert len(new['dof_pos'])==n and float(new['fps'])==float(baseline['fps'])
        after,aarr,_,_,aenergy=measure(new);report.update(status='COMPARISON_MEASURED',candidate=str(path),candidate_metrics=after,preservation=dict(root_position_exact=np.array_equal(new['root_pos'],baseline['root_pos']),root_quaternion_exact=np.array_equal(new['root_rot'],baseline['root_rot']),inactive_joint_exact=np.array_equal(new['dof_pos'][:,:15],baseline['dof_pos'][:,:15]),object_npz_byte_identical=sha(path.with_suffix('.npz'))==sha(BASELINE.with_suffix('.npz'))),input_sha256={**report['input_sha256'],str(path):sha(path),str(path.with_suffix('.npz')):sha(path.with_suffix('.npz'))},maximum_arm_position_departure_rad=float(np.max(abs(new['dof_pos'][:,armids]-baseline['dof_pos'][:,armids]))))
        ratios={}
        for key in ['velocity','acceleration','jerk']:
            ratios[key]={measure:after['aggregate'][key][measure]/before['aggregate'][key][measure] if before['aggregate'][key][measure] else None for measure in ['rms','p95_abs','maximum_abs']}
        ratios['angle_energy_above10Hz']=after['frequency_energy']['above10Hz_rad2']/before['frequency_energy']['above10Hz_rad2']
        marker_ratios={};spatial_regressions=[]
        for link in before['world_link_marker_metrics']:
            b=before['world_link_marker_metrics'][link];a=after['world_link_marker_metrics'][link]
            marker_ratios[link]=dict(acceleration_rms=a['acceleration_m_s2']['rms']/b['acceleration_m_s2']['rms'],jerk_rms=a['jerk_m_s3']['rms']/b['jerk_m_s3']['rms'],above10Hz_position_energy=a['position_energy_above10Hz_m2']/b['position_energy_above10Hz_m2'])
            if marker_ratios[link]['above10Hz_position_energy']>1:spatial_regressions.append(dict(link=link,above10Hz_position_energy_ratio=marker_ratios[link]['above10Hz_position_energy'],before_band_position_rms_mm=1000*np.sqrt(b['position_energy_above10Hz_m2']),after_band_position_rms_mm=1000*np.sqrt(a['position_energy_above10Hz_m2'])))
        report['world_marker_candidate_to_baseline_ratios']=marker_ratios;report['spatial_frequency_regressions']=spatial_regressions
        report['spatial_frequency_status']='MIXED_SPATIAL_FREQUENCY_CHANGES' if spatial_regressions else 'ALL_QUERIED_MARKERS_HIGH_FREQUENCY_ENERGY_NOT_WORSE'
        report['candidate_to_baseline_ratios']=ratios
        departure=np.abs(new['dof_pos'][:,armids]-baseline['dof_pos'][:,armids])
        report['per_joint_position_departure']=[dict(joint=name,maximum_rad=float(departure[:,j].max()),rms_rad=float(np.sqrt(np.mean(departure[:,j]**2))),peak_time_s=float(np.argmax(departure[:,j])*dt)) for j,name in enumerate(armnames)]
        # Explicit measurement goals, rather than a hidden PASS derived from changed metrics.
        goals=dict(acceleration_rms_reduced=ratios['acceleration']['rms']<1,jerk_rms_reduced=ratios['jerk']['rms']<1,angle_high_frequency_energy_reduced=ratios['angle_energy_above10Hz']<1,acceleration_max_not_worse=ratios['acceleration']['maximum_abs']<=1,jerk_max_not_worse=ratios['jerk']['maximum_abs']<=1)
        report['comparative_goals']=goals;report['smoothness_goal_status']='PASS_ALL_COMPARATIVE_GOALS' if all(goals.values()) else 'MIXED_OR_NOT_IMPROVED'
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(3,2,figsize=(15,12),constrained_layout=True);axes=axes.ravel()
        for k,key in enumerate(['velocity','acceleration','jerk']):
            axes[k].plot(times[key],np.max(abs(barr[key]),axis=1),label='Before',color='#d76c25',alpha=.8);axes[k].plot(times[key],np.max(abs(aarr[key]),axis=1),label='After',color='#1768ac');axes[k].set(title='Maximum absolute arm '+key+' per sample',xlabel='Time [s]',ylabel={'velocity':'rad/s','acceleration':'rad/s²','jerk':'rad/s³'}[key]);axes[k].legend()
        axes[3].semilogy(freq,np.maximum(benergy.sum(axis=1),1e-30),label='Before',color='#d76c25');axes[3].semilogy(freq,np.maximum(aenergy.sum(axis=1),1e-30),label='After',color='#1768ac');axes[3].axvline(10,color='gray',ls='--');axes[3].set(title='Detrended Hann angle-energy spectrum: all14arms',xlabel='Frequency [Hz]',ylabel='Energy per bin [rad²]');axes[3].legend()
        x=np.arange(14);axes[4].bar(x-.18,[j['acceleration']['rms'] for j in before['per_joint']],width=.36,label='Before',color='#d76c25');axes[4].bar(x+.18,[j['acceleration']['rms'] for j in after['per_joint']],width=.36,label='After',color='#1768ac');axes[4].set_xticks(x,[name.replace('_joint','').replace('shoulder','sh').replace('left_','L_').replace('right_','R_') for name in armnames],rotation=65,ha='right');axes[4].set(title='Acceleration RMS per joint',ylabel='rad/s²');axes[4].legend()
        axes[5].bar(x-.18,[j['jerk']['rms'] for j in before['per_joint']],width=.36,label='Before',color='#d76c25');axes[5].bar(x+.18,[j['jerk']['rms'] for j in after['per_joint']],width=.36,label='After',color='#1768ac');axes[5].set_xticks(x,[name.replace('_joint','').replace('shoulder','sh').replace('left_','L_').replace('right_','R_') for name in armnames],rotation=65,ha='right');axes[5].set(title='Jerk RMS per joint',ylabel='rad/s³');axes[5].legend()
        for ax in axes:ax.grid(True,alpha=.2)
        fig.suptitle('Arm reference smoothness comparison | '+tag+'\nSampled kinematic metrics; no motor vibration or physical grasp certification',fontsize=14)
        fig.savefig(ROOT/('reports/smoothness_'+tag+'_comparison.png'),dpi=160);plt.close(fig)
    write_json(ROOT/('reports/smoothness_'+tag+'.json'),report)
    with (ROOT/('reports/smoothness_'+tag+'_per_joint.csv')).open('w',newline='') as handle:
        columns=['joint','before_accel_rms','before_accel_max','before_jerk_rms','before_jerk_max','after_accel_rms','after_accel_max','after_jerk_rms','after_jerk_max'];writer=csv.writer(handle);writer.writerow(columns)
        for i,row in enumerate(before['per_joint']):
            new=after['per_joint'][i] if after else None
            values=[row['joint'],row['acceleration']['rms'],row['acceleration']['maximum_abs'],row['jerk']['rms'],row['jerk']['maximum_abs']]
            values += [new[key][stat] if new else '' for key,stat in [('acceleration','rms'),('acceleration','maximum_abs'),('jerk','rms'),('jerk','maximum_abs')]]
            writer.writerow(values)
    print(json.dumps({k:report[k] for k in ['status','baseline_metrics'] if k in report},indent=2) if not after else json.dumps(dict(status=report['status'],ratios=report['candidate_to_baseline_ratios'],goals=report['comparative_goals']),indent=2))

if __name__=='__main__':main()
