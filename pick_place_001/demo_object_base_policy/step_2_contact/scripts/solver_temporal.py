"""C2 lamp SE3 temporal model and exact P1-continuous high-rate reprojection inputs."""
import argparse,copy,json,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,MOTION,OBJECT,ASSETS,URDF,install,write_json,sha
install()
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation,Slerp
import xml.etree.ElementTree as ET
sys.path.insert(0,str(REPO/'legged_gym/legged_gym/scripts'))
from convert_gmr_qpos import forward_kinematics

def linear(values,times,newtimes):
    values=np.asarray(values);flat=values.reshape(len(values),-1);out=np.stack([np.interp(newtimes,times,flat[:,j]) for j in range(flat.shape[1])],axis=1)
    return out.reshape((len(newtimes),)+values.shape[1:])

def prepare(rate=480,sigma_s=.1,tag='attempt12'):
    source=ROOT/'candidates/attempt11/interpose_floorlamp_g1_hand_contact.pkl';old=pickle.load(source.open('rb'));p1=pickle.load(MOTION.open('rb'));obj=np.load(source.with_suffix('.npz'));p1obj=np.load(OBJECT)
    fps=float(p1['fps']);n=len(p1['root_pos']);times=np.arange(n)/fps;newtimes=np.arange((n-1)*int(rate/fps)+1)/rate;ratio=int(rate/fps)
    assert rate==480 and ratio==4
    signals=np.column_stack([obj['trans'],np.unwrap(Rotation.from_quat(obj['rot']).as_euler('xyz')[:,2])]);filtered=gaussian_filter1d(signals,sigma_s*fps,axis=0,mode='nearest')
    spline=CubicSpline(times,filtered,axis=0,bc_type='natural');dense=np.linspace(times[0],times[-1],10001);densevalues=spline(dense)
    floor=float(json.loads((ASSETS/'asset_profile.json').read_text())['source_bounds_m'][0][2]);floor_correction=max(0.,.002-float((densevalues[:,2]+floor).min()));filtered[:,2]+=floor_correction
    spline=CubicSpline(times,filtered,axis=0,bc_type='natural');values=spline(newtimes);velocity=spline(newtimes,1);acceleration=spline(newtimes,2);denseacc=spline(dense,2)
    base=copy.deepcopy(p1);base['fps']=rate;base['root_pos']=linear(p1['root_pos'],times,newtimes);base['root_pos'][::ratio]=p1['root_pos'];base['root_rot']=Slerp(times,Rotation.from_quat(p1['root_rot']))(newtimes).as_quat();base['root_rot'][::ratio]=p1['root_rot'];base['dof_pos']=linear(p1['dof_pos'],times,newtimes);base['dof_pos'][::ratio]=p1['dof_pos'];base['dof_pos'][:,15:]=linear(old['dof_pos'][:,15:],times,newtimes)
    # Inputs' local body positions are recomputed, not linearly interpolated.
    u=ET.parse(URDF).getroot();names=[j.get('name') for j in u.findall('joint') if j.get('type')!='fixed'];base['local_body_pos'],_=forward_kinematics(u,base['dof_pos'],names,base['link_body_list'])
    out=ROOT/'reports';path=out/('solver_temporal_'+tag+'_input.pkl')
    with path.open('wb') as f:pickle.dump(base,f,protocol=4)
    rotation=Rotation.from_euler('z',values[:,3:4]).as_quat();source_times=np.interp(newtimes,times,p1obj['source_time_s']);source_times[::ratio]=p1obj['source_time_s']
    np.savez_compressed(out/('solver_temporal_'+tag+'_target.npz'),trans=values[:,:3],rot=rotation,source_time_s=source_times)
    np.savez_compressed(out/('solver_temporal_'+tag+'_model.npz'),knot_times_s=times,filtered_signals=filtered,cubic_coefficients=spline.c,target_times_s=newtimes,analytic_velocity_xyz_yaw=velocity,analytic_acceleration_xyz_yaw=acceleration)
    originalacc=np.gradient(np.gradient(signals,1/fps,axis=0),1/fps,axis=0);report=dict(status='PREPARED_FROZEN_C2_LAMP_PATH',source=str(source),source_sha256=sha(source),source_object_sha256=sha(source.with_suffix('.npz')),frames=len(newtimes),fps=rate,duration_s=float(newtimes[-1]),filter=dict(method='Gaussian lowpass on XYZ and unwrapped upright yaw, followed by natural cubic C2 spline; coefficients saved',sigma_seconds=sigma_s,sigma_native_frames=sigma_s*fps,endpoint='nearest Gaussian boundary and natural spline second derivative zero',channels=['x_m','y_m','z_m','yaw_rad'],global_floor_correction_m=floor_correction),maximum_deviation_from_attempt11_native=dict(position_m=float(np.linalg.norm(spline(times)[:,:3]-signals[:,:3],axis=1).max()),yaw_rad=float(np.max(abs(spline(times)[:,3]-signals[:,3])))),maximum_linear_acceleration_m_s2=dict(before_native_finite_difference=float(np.linalg.norm(originalacc[:,:3],axis=1).max()),after_analytic_dense=float(np.linalg.norm(denseacc[:,:3],axis=1).max())),maximum_yaw_acceleration_rad_s2=dict(before_native_finite_difference=float(abs(originalacc[:,3]).max()),after_analytic_dense=float(abs(denseacc[:,3]).max())),p1_knots_bit_exact=bool(np.array_equal(base['root_pos'][::4],p1['root_pos']) and np.array_equal(base['root_rot'][::4],p1['root_rot']) and np.array_equal(base['dof_pos'][::4,:15],p1['dof_pos'][:,:15]) and np.array_equal(source_times[::4],p1obj['source_time_s'])),object_path_frozen_for_reprojection=True)
    write_json(out/('solver_temporal_'+tag+'.json'),report);print(json.dumps(report,indent=2));return path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',default='attempt12');p.add_argument('--sigma-seconds',type=float,default=.1);a=p.parse_args();prepare(sigma_s=a.sigma_seconds,tag=a.tag)
