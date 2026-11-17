"""Render stored corrected robot/object poses: geometry replay, not a policy rollout."""
import argparse,json,sys,pickle,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,ASSETS,URDF,install,sha,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();install()
    source=args.candidate.resolve();out=args.output.resolve()
    if ROOT not in out.parents or ROOT not in source.parents:raise ValueError('Paths must be inside step_2')
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np,mujoco,imageio.v2 as imageio
    from scipy.spatial.transform import Rotation
    from PIL import Image,ImageDraw,ImageFont
    sys.path.insert(1,str(REPO/'legged_gym/legged_gym/scripts'))
    from render_motion import build_scene,set_frame,verify_fk
    with source.open('rb') as f:motion=pickle.load(f)
    with np.load(source.with_suffix('.npz'),allow_pickle=False) as z:obj={k:z[k].copy() for k in z.files}
    plan=json.loads((source.parent/'contact_plan.json').read_text());fps=float(motion['fps']);indices=np.arange(0,len(motion['root_pos']),int(round(fps/30)))
    assert indices[-1]==len(motion['root_pos'])-1
    model,addresses,rootadr,objid=build_scene(URDF,ASSETS/'floorlamp_visual.obj',960,720)
    # Show the collider whose surface is used for contact, alongside the thinner visual shaft.
    intermediate=out/'reference_scene_base.xml';mujoco.mj_saveLastXML(str(intermediate),model)
    tree=ET.parse(intermediate);scene=tree.getroot();asset=scene.find('asset');body=scene.find("worldbody/body[@name='reference_object']")
    for component in ['base','stem','shade']:
        name='proxy_'+component
        ET.SubElement(asset,'mesh',name=name,file=str(ASSETS/('floorlamp_'+component+'_collision.obj')))
        ET.SubElement(body,'geom',type='mesh',mesh=name,rgba='.2 .8 .4 .22',group='3',contype='0',conaffinity='0')
    model=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'));data=mujoco.MjData(model);verify_fk(model,data,motion,obj,addresses,rootadr,objid)
    # Persist render model locally, with absolute source mesh paths retained as provenance.
    mujoco.mj_saveLastXML(str(out/'reference_scene.xml'),model)
    renderer=mujoco.Renderer(model,height=720,width=960);option=mujoco.MjvOption();option.geomgroup[0]=0
    overview=mujoco.MjvCamera();overview.distance=3.2;overview.elevation=-12
    detail=mujoco.MjvCamera();detail.distance=.85;detail.elevation=-10
    handids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,side+'_rubber_hand') for side in ['left','right']]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    writer=imageio.get_writer(str(out/'reference_corrected.mp4'),fps=30,codec='libx264',quality=8,macro_block_size=16);selected={0,len(indices)//2,len(indices)-1};stills=[]
    try:
        for ordinal,frame in enumerate(indices):
            set_frame(model,data,motion,obj,int(frame),addresses,rootadr,objid)
            world_rotation=Rotation.from_quat(obj['rot'][frame]).as_matrix()
            anchors=np.asarray([world_rotation@np.asarray(plan['hands'][side]['target_point_object'])+obj['trans'][frame] for side in ['left','right']])
            target=anchors.mean(axis=0)
            bodyforward=Rotation.from_quat(motion['root_rot'][frame]).as_matrix()[:,0]
            bearing=np.degrees(np.arctan2(bodyforward[1],bodyforward[0]));overview.azimuth=float(bearing+140);detail.azimuth=float(bearing+135)
            overview.lookat[:]=(motion['root_pos'][frame]+obj['trans'][frame])/2;overview.lookat[2]=1.0;detail.lookat[:]=target
            canvas=Image.new('RGB',(1920,720))
            for column,camera in enumerate([overview,detail]):
                renderer.update_scene(data,camera=camera,scene_option=option)
                for anchor,color in zip(anchors,[(.15,.8,.35,1),(.85,.25,.15,1)]):
                    if renderer.scene.ngeom<renderer.scene.maxgeom:
                        geom=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.004,.004,.004]),anchor,np.eye(3).ravel(),np.array(color));renderer.scene.ngeom+=1
                pic=Image.fromarray(renderer.render());draw=ImageDraw.Draw(pic);draw.rounded_rectangle((12,12,948,90),radius=8,fill=(245,248,250));draw.text((24,20),'STEP2 | CORRECTED REFERENCE | no policy / dynamics',font=font,fill=(25,35,45));draw.text((24,50),'t=%.3fs | %s | pre-grasped fixed hands'%(frame/fps,'full scene' if column==0 else 'hand detail'),font=font,fill=(60,75,85));draw.text((24,690),'Translucent green = collision proxy; markers = target anchors',fill=(25,45,40));canvas.paste(pic,(column*960,0))
            writer.append_data(np.asarray(canvas))
            if ordinal in selected:
                name='reference_frame_%04d.png'%frame;canvas.save(out/name);stills.append(name)
    finally:
        writer.close();renderer.close()
    reader=imageio.get_reader(str(out/'reference_corrected.mp4'));meta=reader.get_meta_data();decoded=sum(1 for _ in reader);reader.close();assert decoded==len(indices) and meta['fps']==30
    # Full-resolution selected frames retained; sheet uses complete frames at half scale.
    sheet=Image.new('RGB',(960,1080),'white')
    for row,name in enumerate(stills):sheet.paste(Image.open(out/name).resize((960,360)),(0,row*360))
    sheet.save(out/'contact_sheet.jpg',quality=94)
    write_json(out/'video_validation.json',dict(status='PASS',frames=decoded,fps=30,dimensions=[720,1920,3],native_fps=fps,native_frames=len(motion['root_pos']),native_endpoint_s=(len(motion['root_pos'])-1)/fps,media_duration_s=decoded/30,sampled_native_frame_indices=indices.tolist(),sha256=sha(out/'reference_corrected.mp4'),source_sha256={str(p):sha(p) for p in [source,source.with_suffix('.npz'),source.parent/'contact_plan.json']},scope='Stored pose replay via MuJoCo renderer; no base policy, no dynamics, no physical-lift claim',contact_markers='Fixed target anchors on object collider, not measured force contacts',translucent_green_geometry='Actual three collision proxy meshes; visual lamp is thinner'))
    print('PASS reference video',decoded,'frames',flush=True)

if __name__=='__main__':main()
