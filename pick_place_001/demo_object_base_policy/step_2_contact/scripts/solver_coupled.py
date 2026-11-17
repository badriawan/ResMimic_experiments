"""Per-frame coupled arm/object IK with source-collider clearance hinges."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from solver_kinematics import Chain
from geometry_oracle import Oracle
from hand_contact_ik import ArmContactKinematics

def refine_coupled(base,qseed,op0,oq0,hands,names,urdf,lower,upper,velocity,fps,tangent_weight=0.,lock_initial_tangent=False,freeze_object=False):
    g=Oracle();ids=np.arange(15,29);mapping={int(j):i for i,j in enumerate(ids)};n=len(qseed);rr=Rotation.from_quat(base['root_rot']).as_matrix();r0=Rotation.from_quat(oq0).as_matrix();zaxis=np.array([0.,0.,1.]);chains={name:Chain(urdf,names,name) for name in base['link_body_list'] if name!='pelvis'}
    if lock_initial_tangent:
        for side in ['left','right']:
            hand=hands[side];chain=Chain(urdf,names,hand['link']);_,handr=chain.fk(qseed[0],base['root_pos'][0],rr[0]);local_tangent=r0[0].T@handr@np.asarray(hand['tangent_local']);normal=np.asarray(hand['outward_normal_object']);local_tangent-=normal*np.dot(local_tangent,normal);hand['target_tangent_object']=(local_tangent/np.linalg.norm(local_tangent)).tolist()
    dimension=14 if freeze_object else 17
    pairs=g.self_pairs+[(link,'floorlamp') for link in g.object_nonhand_links]
    qout=qseed.copy();op=op0.copy();oq=oq0.copy();traces=[];previous=qseed[0,ids].copy();prevdelta=np.zeros(3)
    for frame in range(n):
        models=[];tmodels=[]
        for side in ['left','right']:
            h=hands[side];chain=chains[h['link']]
            models.append(ArmContactKinematics(chain,base['dof_pos'][frame:frame+1],base['root_pos'][frame:frame+1],rr[frame:frame+1],h['pad_local'],h['pad_normal_local'],ids))
            tmodels.append(ArmContactKinematics(chain,base['dof_pos'][frame:frame+1],base['root_pos'][frame:frame+1],rr[frame:frame+1],h['pad_local'],h['tangent_local'],ids))
        low=lower[ids].copy() if freeze_object else np.r_[lower[ids],[-.12,-.12,-.5]];high=upper[ids].copy() if freeze_object else np.r_[upper[ids],[.12,.12,.5]]
        if frame:
            low[:14]=np.maximum(low[:14],previous-velocity[ids]/fps*.95);high[:14]=np.minimum(high[:14],previous+velocity[ids]/fps*.95)
        cache={}
        def calc(x):
            if 'x' in cache and np.array_equal(cache['x'],x):return cache
            full=base['dof_pos'][frame].copy();full[ids]=x[:14]
            position=op0[frame].copy();rot=r0[frame].copy()
            if not freeze_object:position[:2]+=x[14:16];rot=Rotation.from_rotvec(zaxis*x[16]).as_matrix()@r0[frame]
            pquat=Rotation.from_matrix(rot).as_quat();g.set_frame(full,base['root_pos'][frame],base['root_rot'][frame],position,pquat)
            body_twists={}
            def point_jac(link,point):
                jac=np.zeros((3,dimension))
                if link=='floorlamp':
                    if not freeze_object:jac[:,14:16]=np.eye(3)[:,:2];jac[:,16]=np.cross(zaxis,point-position)
                    return jac
                if link=='pelvis' or not any(index in mapping for index in chains[link].indices):return jac
                if link not in body_twists:
                    p=base['root_pos'][frame].copy();r=rr[frame].copy();axes=np.zeros((14,3));origins=np.zeros((14,3))
                    for xyz,rotation,index,axis in chains[link].steps:
                        p=p+r@xyz;r=r@rotation
                        if index is not None:
                            if index in mapping:axes[mapping[index]]=r@axis;origins[mapping[index]]=p
                            r=r@Rotation.from_rotvec(full[index]*axis).as_matrix()
                    body_twists[link]=(axes,origins)
                axes,origins=body_twists[link];jac[:,:14]=np.cross(axes,point-origins).T
                return jac
            residual=[];jacobians=[];contact=[]
            for hi,side in enumerate(['left','right']):
                h=hands[side];p,nn,jp,jn=models[hi].evaluate(x[:14][None]);p,nn,jp,jn=p[0],nn[0],jp[0],jn[0]
                local=np.array(h['target_point_object'])+np.array(h['outward_normal_object'])*h['target_gap_m'];outnormal=rot@h['outward_normal_object'];target=position+rot@local
                jpos=np.zeros((3,dimension));jpos[:,:14]=jp
                if not freeze_object:jpos[:,14:16]=-np.eye(3)[:,:2];jpos[:,16]=-np.cross(zaxis,rot@local)
                jnorm=np.zeros((3,dimension));jnorm[:,:14]=jn
                if not freeze_object:jnorm[:,16]=np.cross(zaxis,outnormal)
                if tangent_weight:
                    _,tv,_,tj=tmodels[hi].evaluate(x[:14][None]);goal=rot@h['target_tangent_object'];jtv=np.zeros((3,dimension));jtv[:,:14]=tj[0]
                    if not freeze_object:jtv[:,16]=-np.cross(zaxis,goal)
                    residual.append((tv[0]-goal)*tangent_weight);jacobians.append(jtv*tangent_weight)
                residual.extend([(p-target)*3000,(nn+outnormal)*500]);jacobians.extend([jpos*3000,jnorm*500]);contact.append((np.linalg.norm(p-target),np.degrees(np.arccos(np.clip(nn@(-outnormal),-1,1)))))
            jreg=np.zeros((14,dimension));jreg[:,:14]=np.eye(14)
            residual.extend([(x[:14]-qseed[frame,ids])*.05,(x[:14]-previous)*.2]);jacobians.extend([jreg*.05,jreg*.2])
            if not freeze_object:
                objweights=np.array([20.,20.,2.]);jobj=np.zeros((3,dimension));jobj[:,14:]=np.diag(objweights)
                residual.append(x[14:]*objweights);jacobians.append(jobj)
                smooth=np.array([100.,100.,10.]);jobj2=np.zeros((3,dimension));jobj2[:,14:]=np.diag(smooth)
                residual.append((x[14:]-prevdelta)*smooth);jacobians.append(jobj2)
            collision=np.zeros(len(pairs));jc=np.zeros((len(pairs),dimension));worst=1.
            for pi,(a,b) in enumerate(pairs):
                distance=g.fast_pair_distance(a,b,threshold=.003);d=distance['distance_m'];worst=min(worst,d)
                if d<.002:
                    collision[pi]=(.002-d)*10000
                    if distance['normal_valid']:
                        grad=distance['normal_a_to_b']@(point_jac(b,distance['point_b_world'])-point_jac(a,distance['point_a_world']));jc[pi]=-grad*10000
            residual.append(collision);jacobians.append(jc)
            cache.update(x=x.copy(),fun=np.concatenate(residual),jac=np.vstack(jacobians),contact=contact,worst=worst,objpos=position,objquat=pquat)
            return cache
        seed=qseed[frame,ids].copy() if freeze_object else np.r_[qseed[frame,ids],prevdelta];seed[:14]=np.clip(seed[:14],low[:14]+1e-10,high[:14]-1e-10)
        result=least_squares(lambda x:calc(x)['fun'],seed,jac=lambda x:calc(x)['jac'],bounds=(low,high),max_nfev=70,ftol=1e-8,xtol=1e-8,gtol=1e-7,x_scale='jac')
        value=calc(result.x);qout[frame,ids]=result.x[:14];op[frame]=value['objpos'];oq[frame]=value['objquat'];previous=result.x[:14].copy();prevdelta=np.zeros(3) if freeze_object else result.x[14:].copy()
        for hi,side in enumerate(['left','right']):traces.append(dict(frame=frame,side=side,point_error_m=float(value['contact'][hi][0]),normal_error_deg=float(value['contact'][hi][1]),tangent_error_deg=0.,nfev=int(result.nfev),collision_minimum_m=float(value['worst']),object_delta=prevdelta.tolist(),object_path_frozen=bool(freeze_object)))
        if frame%25==0:print('Coupled',frame,'contact',value['contact'],'mincollision',value['worst'],'objdelta',result.x[14:],flush=True)
    if freeze_object:op=op0.copy();oq=oq0.copy()
    return qout,op,oq,traces
