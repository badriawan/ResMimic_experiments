"""Wall-clock transition diagnostics; finite differences are not a dynamics test."""
import argparse,pickle,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,BASELINE,install,sha,write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidates',type=Path,nargs='+',required=True);args=parser.parse_args();install()
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    def load(path):
        with path.open('rb') as f:m=pickle.load(f)
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:o={k:z[k] for k in z.files}
        return m,o
    def velocities(m,o):
        fps=float(m['fps']);time=(np.arange(len(m['root_pos'])-1)+.5)/fps
        def angular(q):
            r=Rotation.from_quat(q);return (r[:-1].inv()*r[1:]).as_rotvec()*fps
        arrays=dict(root_linear=np.diff(m['root_pos'],axis=0)*fps,joint=np.diff(m['dof_pos'],axis=0)*fps,root_angular=angular(m['root_rot']),object_linear=np.diff(o['trans'],axis=0)*fps,object_angular=angular(o['rot']))
        return time,arrays
    baseline,objects=load(BASELINE);bt,bv=velocities(baseline,objects)
    reports={};figure,axes=plt.subplots(5,2,figsize=(14,15),squeeze=False,constrained_layout=True)
    if len(args.candidates)>2:raise ValueError('At most two candidates per diagnostic figure')
    for col,path in enumerate(args.candidates):
        path=path.resolve(strict=True)
        if ROOT not in path.parents:raise ValueError('Candidate outside transition folder')
        tag=path.parent.name;m,o=load(path);time,vel=velocities(m,o);fps=float(m['fps']);native=np.arange(len(m['root_pos']))/fps
        with np.load(path.parent/'transition_clock.npz') as z:clock={k:z[k] for k in z.files}
        plan=json.loads((path.parent/'contact_plan.json').read_text())['transition'];hold=plan['hold_s'];join=hold+plan['ramp_s'];start=plan['source_start_time_s'];tau=clock['source_reference_time_s'];mid_tau=(tau[:-1]+tau[1:])/2
        metrics={};checks={}
        for key,value in vel.items():
            speed=np.linalg.norm(value,axis=1) if key!='joint' else np.sqrt(np.mean(value**2,axis=1));source_speed=np.linalg.norm(bv[key],axis=1) if key!='joint' else np.sqrt(np.mean(bv[key]**2,axis=1))
            metrics[key]=dict(initial=float(speed[0]),maximum=float(speed.max()),source_initial_at_start=float(source_speed[int(round(start*float(baseline['fps'])))]),source_maximum=float(source_speed.max()),hold_maximum=float(speed[time<hold].max()),suffix_maximum=float(speed[time>=join].max()))
            # A shared speed<=1 time mapping cannot exceed the baseline global
            # per-component velocity envelope; no acceleration/balance claim.
            checks[key+'_no_global_component_overspeed']=bool(np.all(np.max(abs(value),axis=0)<=np.max(abs(bv[key]),axis=0)+1e-8))
        ax=axes[0,col];ax.plot(native,tau,label='source phase τ');ax.plot(native,native+start,':',label='unwarped source phase');ax.set_ylabel('Source phase (s)');ax.legend(fontsize=8)
        ax2=ax.twinx();ax2.plot(native,clock['clock_rate'],color='tab:red',alpha=.5,label='clock speed');ax2.set_ylabel('Clock speed');ax2.set_ylim(-.05,1.1)
        for row,keys,ylabel in [(1,['root_linear','object_linear'],'Linear speed (m/s)'),(2,['root_angular','object_angular'],'Angular speed (rad/s)'),(3,['joint'],'Joint RMS speed (rad/s)')]:
            ax=axes[row,col]
            for key in keys:
                value=vel[key];speed=np.linalg.norm(value,axis=1) if key!='joint' else np.sqrt(np.mean(value**2,axis=1));reference=np.linalg.norm(bv[key],axis=1) if key!='joint' else np.sqrt(np.mean(bv[key]**2,axis=1))
                ax.plot(time,speed,label=key);ax.plot(time,np.interp(mid_tau,bt,reference),':',alpha=.6,label=key+' source speed at τ')
            ax.set_ylabel(ylabel);ax.legend(fontsize=7)
        axes[4,col].plot(time,np.diff(tau)*fps,label='FD source clock rate');axes[4,col].axhline(1,color='gray',ls=':');axes[4,col].set_ylabel('Δτ / Δt');axes[4,col].legend(fontsize=8)
        for ax in axes[:,col]:
            ax.axvline(hold,color='black',ls='--',lw=.7);ax.axvline(join,color='black',ls='--',lw=.7);ax.set_xlabel('Output wall time (s)');ax.grid(alpha=.2)
        axes[0,col].set_title(tag+' | original source start %.3fs'%start)
        report=dict(candidate=str(path),status='PASS_CLOCK_VELOCITY_DIAGNOSTICS' if all(checks.values()) else 'FAIL_CLOCK_VELOCITY_DIAGNOSTICS',checks=checks,metrics=metrics,duration_s=float(native[-1]),source_start_s=float(tau[0]),source_end_s=float(tau[-1]),hold_s=hold,ramp_s=plan['ramp_s'],suffix_join_output_s=join,suffix_join_source_s=plan['source_join_time_s'],scope='Finite differences on actual saved native480Hz reference. Clock C2 does not make the pre-existing piecewise pose interpolant globally C2. No acceleration, force balance, grasp or physical carry certificate.',input_sha256={str(p):sha(p) for p in [path,path.with_suffix('.npz'),path.parent/'transition_clock.npz',BASELINE,BASELINE.with_suffix('.npz')]})
        write_json(ROOT/('reports/transition_metrics_'+tag+'.json'),report);reports[tag]=report
        np.savez_compressed(ROOT/('reports/transition_metrics_'+tag+'.npz'),time_s=time,source_reference_time_s=mid_tau,**vel)
    figure.suptitle('Shared reference clock, exact rest prefix and 1× suffix | geometric/kinematic diagnostics',fontsize=13)
    figure.savefig(ROOT/'reports/transition_metrics_comparison.png',dpi=150);figure.savefig(ROOT/'reports/transition_metrics_comparison.pdf');plt.close(figure)
    print(json.dumps({k:dict(status=v['status'],metrics=v['metrics']) for k,v in reports.items()},indent=2))
    if not all(all(v['checks'].values()) for v in reports.values()):raise SystemExit(1)
if __name__=='__main__':main()
