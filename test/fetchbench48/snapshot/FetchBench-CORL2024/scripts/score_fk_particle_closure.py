"""Execution-aware geometric contact scoring on reconstructed object surfaces."""
import concurrent.futures
import importlib.util
import inspect
from pathlib import Path
import os
import sys
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
spec=importlib.util.spec_from_file_location('environment_score',Path(__file__).with_name('score_fk_particle_environment.py'))
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
CACHE={}


def closure_score(gripper,r,allq,trans,rotation,robot_surface,manifest):
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    from scipy.special import expit
    from utils.point_cloud_geometry import estimate_point_cloud_geometry
    from utils.multigripper_fk import graspqp_friction_cone_metrics
    view=r['view']
    if view not in CACHE:
        p=next(p for p in manifest['inputs'][view] if p.endswith('sam3d_fused_robot_base.npy'))
        obj=np.load(p)
        if len(obj)>4096:obj=obj[np.random.default_rng(20260915).choice(len(obj),4096,replace=False)]
        geo=estimate_point_cloud_geometry(obj,k_neighbors=30)
        CACHE[view]=(obj,cKDTree(obj),geo['normals'],geo['confidence'])
    obj,tree,normals,confidence=CACHE[view]
    T=np.asarray(r['robot_from_object']);fingers=len(gripper.tip_links)
    with torch.no_grad():
        pts,ns=gripper.tip_link_surface_geometry(torch.tensor(allq.reshape(20,-1)),torch.tensor(trans),torch.tensor(rotation))
    pts=pts.numpy().reshape(4,5,fingers,-1,3)@T[:3,:3].T+T[:3,3]
    ns=ns.numpy().reshape(pts.shape)@T[:3,:3].T
    gap,idx=tree.query(pts.reshape(-1,3),workers=1);gap=gap.reshape(pts.shape[:-1]);idx=idx.reshape(gap.shape)
    closest=gap.argmin(-1)
    a,b,f=np.indices(closest.shape)
    gaps=gap[a,b,f,closest];oi=idx[a,b,f,closest]
    contact=obj[oi];normal=normals[oi];opposition=-np.sum(ns[a,b,f,closest]*normal,axis=-1)
    activation=np.exp(-.5*(gaps/.005)**2)*(.25+.75*confidence[oi])*expit(3*opposition)
    center=obj.mean(0);scale=max(float(np.linalg.norm(obj-center,axis=1).max()),1e-4)
    with torch.no_grad():
        qp,residual,sv=graspqp_friction_cone_metrics(
            torch.tensor(contact.reshape(20,fingers,3),dtype=torch.float32),
            torch.tensor(normal.reshape(20,fingers,3),dtype=torch.float32),
            torch.tensor(center,dtype=torch.float32),torch.tensor(scale,dtype=torch.float32),
            contact_activation=torch.tensor(activation.reshape(20,fingers),dtype=torch.float32),
            iterations=80,include_torque_disturbances=True)
    qp=qp.numpy().reshape(4,5);residual=residual.numpy().reshape(4,5);sv=sv.numpy().reshape(4,5)
    _,idx=tree.query(robot_surface.reshape(-1,3),workers=1)
    signed=np.sum((robot_surface.reshape(-1,3)-obj[idx])*normals[idx],axis=1).reshape(robot_surface.shape[:-1])
    penetration=np.maximum(-signed,0)
    results=[]
    for i in range(4):
        feat={}
        for tag,j in [('nominal',0),('outer',1),('inner',4)]:
            feat.update({f'closure_{tag}_gap_min':float(gaps[i,j].min()),f'closure_{tag}_gap_max':float(gaps[i,j].max()),
                f'closure_{tag}_gap_mean':float(gaps[i,j].mean()),f'closure_{tag}_within5mm':float((gaps[i,j]<.005).mean()),
                f'closure_{tag}_within10mm':float((gaps[i,j]<.010).mean()),f'closure_{tag}_activation':float(activation[i,j].mean()),
                f'closure_{tag}_normal_opposition':float(opposition[i,j].mean()),f'closure_{tag}_qp':float(qp[i,j]),
                f'closure_{tag}_wrench_residual':float(residual[i,j]),f'closure_{tag}_wrench_sv':float(sv[i,j]),
                f'closure_{tag}_object_pen_mean':float(penetration[i,j].mean()),f'closure_{tag}_object_pen_max':float(penetration[i,j].max())})
        feat['closure_gap_improvement']=float((gaps[i,1]-gaps[i,4]).mean())
        feat['closure_sweep_pen_max']=float(penetration[i,1:].max())
        feat['closure_sweep_activation_min']=float(activation[i,1:].mean(-1).min())
        assert np.isfinite(list(feat.values())).all()
        results.append(feat)
    return results


source=inspect.getsource(base.worker)
anchor='        for i,c in enumerate(cs):'
assert source.count(anchor)==1
source=source.replace(anchor,'        closure_rows = _closure_score(gripper,r,allq,trans,rotation,robot,manifest)\n'+anchor)
source=source.replace("            feat={}\n","            feat=dict(closure_rows[i])\n")
source=source.replace("environment_features_","closure_features_")
base.__dict__['_closure_score']=closure_score
exec(compile(source,'<fixed-pose-closure-features>','exec'),base.__dict__)


if __name__=='__main__':
    with concurrent.futures.ProcessPoolExecutor(4) as pool:
        for result in pool.map(base.worker,[('A','barrett'),('A','shadowhand'),('B','barrett'),('B','shadowhand')]):print(result,flush=True)
