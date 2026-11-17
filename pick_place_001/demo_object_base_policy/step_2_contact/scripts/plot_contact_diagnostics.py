"""Plot independently measured reference geometry; no dynamics or policy result."""
import argparse,json,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,MOTION,OBJECT,install

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--tag');args=parser.parse_args();install()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=args.candidate.resolve(strict=True);tag=args.tag or path.parent.name
    if ROOT not in path.parents:raise ValueError('Candidate must be in step2')
    report=json.loads((ROOT/('reports/independent_'+tag+'_validation.json')).read_text())
    with np.load(ROOT/('reports/independent_'+tag+'_dense_metrics.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    with np.load(ROOT/'reports/independent_baseline_contacts_metrics.npz',allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
    with path.open('rb') as f:m=pickle.load(f)
    with MOTION.open('rb') as f:source=pickle.load(f)
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
    with np.load(OBJECT,allow_pickle=False) as z:before={k:z[k].copy() for k in z.files}
    t=d['time_s'];bt=b['times_s'];colors=['#1768ac','#d76c25'];sides=['Left','Right']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,3,figsize=(18,13),constrained_layout=True);a=axes.ravel()
    for j,side in enumerate(sides):
        a[0].plot(bt,np.where(b['signed_distance_valid'][:,j],b['distances_m'][:,j]*1000,np.nan),color=colors[j],label=side+' hand–lamp')
    a[0].plot(bt,np.where(b['signed_distance_valid'][:,2],b['distances_m'][:,2]*1000,np.nan),color='#596b78',ls='--',label='Hand–hand')
    a[0].axhline(0,color='black',lw=1);a[0].set(title='Before: source convex-shape distance',xlabel='Time [s]',ylabel='Signed gap [mm]');a[0].legend(fontsize=8)
    for j,side in enumerate(sides):
        a[1].plot(t,d['pad_surface_gap_m'][:,j]*1000,color=colors[j],label=side+' pad normal gap')
        a[1].plot(t,d['support_plane_gap_m'][:,j]*1000,color=colors[j],ls=':',alpha=.8,label=side+' stem support certificate')
    for limit,label in [(0,'No penetration'),(2,'2 mm upper gate')]:a[1].axhline(limit,color='gray',ls='--',lw=1,label=label)
    a[1].set(title='After: gap to actual stem facet',xlabel='Time [s]',ylabel='Gap [mm]');a[1].legend(fontsize=7)
    for j,side in enumerate(sides):a[2].plot(t,d['pad_error_m'][:,j]*1000,color=colors[j],label=side)
    a[2].axhline(2,color='gray',ls='--',label='2 mm gate');a[2].set(title='After: material contact-center error',xlabel='Time [s]',ylabel='Error [mm]');a[2].legend(fontsize=8)
    for j,side in enumerate(sides):a[3].plot(t,d['normal_error_deg'][:,j],color=colors[j],label=side)
    a[3].axhline(5,color='gray',ls='--',label='5° gate');a[3].set(title='After: opposing-normal error',xlabel='Time [s]',ylabel='Angle [deg]');a[3].legend(fontsize=8)
    a[4].plot(t,d['self_distances_m'].min(axis=1)*1000,color=colors[0],label='Nonadjacent robot pairs')
    a[4].plot(t,d['object_nonhand_distances_m'].min(axis=1)*1000,color=colors[1],label='Lamp vs nonhand robot')
    a[4].axhline(1,color='gray',ls='--',label='1 mm gate');a[4].set(title='After: conservative clearance certificates',xlabel='Time [s]',ylabel='Reported lower bound / distance [mm]');a[4].legend(fontsize=8)
    a[5].plot(t,d['object_floor_gap_m'],color=colors[0]);a[5].axhline(0,color='gray',ls='--');a[5].set(title='After: lamp collider clearance from floor',xlabel='Time [s]',ylabel='Clearance [m]')
    for j,side in enumerate(sides):a[6].plot(t,d['material_twist_relative_initial_deg'][:,j],color=colors[j],label=side)
    a[6].set(title='After: material tangent drift vs initial',xlabel='Time [s]',ylabel='Twist [deg]');a[6].legend(fontsize=8)
    a[7].plot(source['root_pos'][:,0],source['root_pos'][:,1],color='#253747',label='Preserved robot root')
    a[7].plot(before['trans'][:,0],before['trans'][:,1],color='#8c9aa5',ls='--',label='Original lamp')
    a[7].plot(o['trans'][:,0],o['trans'][:,1],color=colors[1],label='Retargeted lamp')
    a[7].set(title='Coupled reference path in world XY',xlabel='World X [m]',ylabel='World Y [m]');a[7].axis('equal');a[7].legend(fontsize=8)
    checks=report['joint_velocity_checks'];limits=np.asarray([item['limit_rad_s'] for item in checks]);ratio=np.max(abs(np.diff(m['dof_pos'],axis=0))*float(m['fps'])/limits,axis=1)
    a[8].plot((np.arange(len(ratio))+.5)/float(m['fps']),ratio,color=colors[0],label='Max across29joints')
    a[8].axhline(1,color='gray',ls='--',label='URDF speed limit');a[8].set(title='After: joint speed / actual URDF limit',xlabel='Time [s]',ylabel='Ratio');a[8].legend(fontsize=8)
    for axis in a:axis.grid(True,alpha=.2)
    fig.suptitle('Step2 — coupled hand–floorlamp reference correction | '+tag+'\nIndependent static source-hull/capsule checks; no policy, force, balance, or free-object lifting proof. '+report['status'],fontsize=14)
    output=ROOT/'reports/contact_diagnostics.png';fig.savefig(output,dpi=160);plt.close(fig);print(output)

if __name__=='__main__':main()
