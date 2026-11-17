"""Pure URDF chain helpers and established analytic contact Jacobians."""
import numpy as np
from scipy.spatial.transform import Rotation

def origin(element):
    e=element.find('origin')
    xyz=np.zeros(3) if e is None else np.fromstring(e.get('xyz','0 0 0'),sep=' ')
    rpy=np.zeros(3) if e is None else np.fromstring(e.get('rpy','0 0 0'),sep=' ')
    return xyz,Rotation.from_euler('xyz',rpy).as_matrix()

class Chain:
    def __init__(self,urdf,names,tip):
        self.indices=[];by_child={j.find('child').get('link'):j for j in urdf.findall('joint')};joints=[]
        while tip!='pelvis':
            j=by_child[tip];joints.append(j);tip=j.find('parent').get('link')
        self.steps=[]
        for j in joints[::-1]:
            xyz,r=origin(j);idx=names.index(j.get('name')) if j.get('type')!='fixed' else None
            axis=np.fromstring(j.find('axis').get('xyz'),sep=' ') if idx is not None else None
            if idx is not None:self.indices.append(idx)
            self.steps.append((xyz,r,idx,axis))
    def fk(self,q,rp,rr):
        p,r=rp.copy(),rr.copy()
        for xyz,rot,idx,axis in self.steps:
            p=p+r@xyz;r=r@rot
            if idx is not None:r=r@Rotation.from_rotvec(q[idx]*axis).as_matrix()
        return p,r
