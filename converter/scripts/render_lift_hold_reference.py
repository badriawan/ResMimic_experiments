"""Render the accepted robot/object reference; no integration or learned policy."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import ROOT,REPO,DATA,install,sha,write_json,motion_root
install()
sys.path.insert(0,str(REPO/'legged_gym/legged_gym/scripts'))
from render_motion import MotionUnpickler,build_scene,rotate_xyzw
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
import trimesh


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fps',type=int,default=30)
    parser.add_argument('--azimuth',type=float,default=315)
    parser.add_argument('--motion',type=Path,default=DATA/'final/lift_hold_001_g1.pkl')
    parser.add_argument('--report',type=Path,default=ROOT/'reports/reference_video.json')
    parser.add_argument('--output',type=Path,default=ROOT/'videos/lift_hold_001_reference.mp4')
    args=parser.parse_args()
    output=args.output.resolve()
    if ROOT not in output.parents:raise ValueError('Reference video must be inside converter')
    if output.exists():raise FileExistsError('Refusing to overwrite an existing reference video')
    output.parent.mkdir(parents=True,exist_ok=True)
    motion_path=args.motion.resolve(strict=True)
    motion_root(motion_path)
    if ROOT not in args.report.resolve().parents:raise ValueError('Report must be inside converter')
    object_path=motion_path.with_suffix('.npz')
    mesh_path=DATA/'assets/floorlamp.obj'
    robot_path=REPO/'assets/g1/g1_custom_collision_29dof.urdf'
    inputs={str(p):sha(p) for p in [motion_path,object_path,mesh_path,robot_path]}
    with motion_path.open('rb') as f:motion=MotionUnpickler(f).load()
    with np.load(object_path,allow_pickle=False) as z:
        obj={k:z[k].copy() for k in ['trans','rot','source_time_s']}
    native_fps=float(motion['fps'])
    step=int(round(native_fps/args.fps))
    if step<1 or abs(native_fps/step-args.fps)>1e-9:raise ValueError('Video FPS must divide the stored reference FPS')
    frames=np.arange(0,len(motion['root_pos']),step)
    if frames[-1]!=len(motion['root_pos'])-1:raise ValueError('Video sampling must include the final reference pose')
    width,height=1280,960
    model,addresses,root_address,object_id=build_scene(robot_path,mesh_path,width,height)
    data=mujoco.MjData(model)

    def pose(frame):
        data.qpos[root_address:root_address+3]=motion['root_pos'][frame]
        data.qpos[root_address+3:root_address+7]=motion['root_rot'][frame][[3,0,1,2]]
        data.qpos[addresses]=motion['dof_pos'][frame]
        data.mocap_pos[object_id]=obj['trans'][frame]
        data.mocap_quat[object_id]=obj['rot'][frame][[3,0,1,2]]
        mujoco.mj_kinematics(model,data)
        mujoco.mj_camlight(model,data)

    checked=[i for i,b in enumerate(motion['link_body_list']) if b not in ['left_toe_link','right_toe_link']]
    body_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,motion['link_body_list'][i]) for i in checked]
    if min(body_ids)<0:raise ValueError('Robot body absent after renderer URDF import')
    maximum_fk=0.
    for frame in [0,len(motion['root_pos'])//2,len(motion['root_pos'])-1]:
        pose(frame)
        expected=rotate_xyzw(motion['root_rot'][frame],motion['local_body_pos'][frame,checked])+motion['root_pos'][frame]
        maximum_fk=max(maximum_fk,float(np.linalg.norm(data.xpos[body_ids]-expected,axis=1).max()))
    if maximum_fk>1e-5:raise ValueError('Renderer joint ordering/FK differs from reference')

    world=rotate_xyzw(motion['root_rot'][frames,None],motion['local_body_pos'][frames])+motion['root_pos'][frames,None]
    bounds=trimesh.load_mesh(mesh_path,process=False).bounds
    corners=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    rot=Rotation.from_quat(obj['rot'][frames]).as_matrix()
    object_corners=np.einsum('tij,cj->tci',rot,corners)+obj['trans'][frames,None]
    points=np.concatenate([world.reshape(-1,3),object_corners.reshape(-1,3)])
    low,high=points.min(axis=0)-.12,points.max(axis=0)+.12
    camera=mujoco.MjvCamera()
    camera.lookat[:]=(low+high)/2
    camera.distance=max(3.,float(np.linalg.norm(high-low))*1.5)
    camera.azimuth=args.azimuth
    camera.elevation=-12
    options=mujoco.MjvOption();options.geomgroup[0]=0
    font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),24) if font_path.exists() else ImageFont.load_default()
    font_small=ImageFont.truetype(str(font_path),20) if font_path.exists() else ImageFont.load_default()
    renderer=mujoco.Renderer(model,width=width,height=height)
    selected=sorted(set([0,int(round(.7*args.fps)),int(round(5.7*args.fps)),len(frames)//2,len(frames)-1]))
    pictures=[];started=time.monotonic()
    try:
        with imageio.get_writer(str(output),fps=args.fps,codec='libx264',quality=8,macro_block_size=1) as writer:
            for index,frame in enumerate(frames):
                pose(int(frame))
                renderer.update_scene(data,camera=camera,scene_option=options)
                pic=Image.fromarray(renderer.render())
                draw=ImageDraw.Draw(pic)
                draw.rounded_rectangle((18,16,width-18,100),radius=10,fill=(246,248,250))
                draw.text((34,25),'KINEMATIC REFERENCE  |  G1 + FLOORLAMP  |  lift_hold_001',font=font,fill=(25,38,53))
                draw.text((34,61),'Pose time %.2f / %.2f s  |  reference %g Hz  |  video %d FPS'%
                    (frame/native_fps,(len(motion['root_pos'])-1)/native_fps,native_fps,args.fps),font=font_small,fill=(65,78,90))
                writer.append_data(np.asarray(pic))
                if index in selected:
                    image_path=output.with_name(output.stem+'_frame_%04d.png'%index)
                    pic.save(image_path);pictures.append(str(image_path))
                if index%60==0:print('Reference video frame %d/%d'%(index+1,len(frames)),flush=True)
    finally:
        renderer.close()
    reader=imageio.get_reader(str(output));metadata=reader.get_meta_data()
    count=0;shape=None
    try:
        for picture in reader:count+=1;shape=list(picture.shape)
    finally:reader.close()
    checks=dict(all_frames_decode=count==len(frames),fps_matches=abs(metadata['fps']-args.fps)<1e-6,
                dimensions_match=shape==[height,width,3],
                duration_matches=abs(metadata['duration']-len(frames)/args.fps)<.03,
                final_reference_pose_included=int(frames[-1])==len(motion['root_pos'])-1,
                input_hashes_unchanged=all(sha(Path(p))==v for p,v in inputs.items()),
                renderer_FK_10um=maximum_fk<1e-5)
    report=dict(status='PASS_REFERENCE_VIDEO' if all(checks.values()) else 'FAIL',checks=checks,
        video=str(output),video_sha256=sha(output),native_frames=len(motion['root_pos']),native_fps=native_fps,
        pose_endpoint_duration_s=(len(motion['root_pos'])-1)/native_fps,
        video_frames=count,video_fps=args.fps,video_duration_s=metadata['duration'],dimensions=shape,
        sampled_native_frame_indices=frames.tolist(),review_images=pictures,
        maximum_renderer_FK_error_m=maximum_fk,excluded_geometry_free_toe_markers=['left_toe_link','right_toe_link'],
        camera=dict(lookat=list(camera.lookat),distance=camera.distance,azimuth=camera.azimuth,elevation=camera.elevation),
        source_files=inputs,elapsed_s=time.monotonic()-started,
        renderer_environment=dict(engine='MuJoCo',version=mujoco.__version__,backend='EGL',python=sys.executable),
        render_executed=True,physics_integration_executed=False,policy_executed=False,training_executed=False,
        scope='Stored final robot/object poses via mj_kinematics + mj_camlight; no mj_step or learned policy.',
        limitations=['Reference visual replay does not demonstrate physical grasp or policy tracking.',
                    '30FPS presentation samples every fourth120Hz knot; endpoint pose retained.',
                    'Source trimming, nominal object floor penetration, and source root drift remain as documented.'])
    write_json(args.report,report)
    if not all(checks.values()):raise RuntimeError('Video validation failed')
    print('Video: '+str(output),flush=True)


if __name__=='__main__':main()
