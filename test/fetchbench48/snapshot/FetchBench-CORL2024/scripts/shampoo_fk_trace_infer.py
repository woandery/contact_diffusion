"""Isolated FK trace and continuous environment-energy adapter."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch

META=json.loads(Path(os.environ['FK_TRACE_META']).read_text())
DEST=Path(META['directory']);DEST.mkdir(parents=True,exist_ok=True)
C=Path(META['contact_root'])
if META.get('deterministic'):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


UP=load(C/'scripts/infer_local_contactdiffusion_grasp.py','fk_trace_original')
sample_contact_targets=UP.sample_contact_targets
ACTIVE=None
TRACES=[]
REF=None
SCENE=None
INFO=None
T=None


def digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        if isinstance(a,torch.Tensor):a=a.detach().cpu().contiguous().numpy()
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def environment(surface):
    rotation=surface.new_tensor(T[:3,:3]);translation=surface.new_tensor(T[:3,3])
    robot=surface@rotation.T+translation
    scene=surface.new_tensor(SCENE)
    nearest=torch.cat([torch.cdist(x,scene[None]).min(2).values for x in robot.split(128,dim=1)],dim=1)
    proximity=torch.relu(.005-nearest)
    lo=surface.new_tensor(INFO['table_xy_min']);hi=surface.new_tensor(INFO['table_xy_max'])
    x,y,z=robot.unbind(2)
    gate=(torch.sigmoid((x-lo[0])/.01)*torch.sigmoid((hi[0]-x)/.01)
          *torch.sigmoid((y-lo[1])/.01)*torch.sigmoid((hi[1]-y)/.01))
    plane=torch.relu(float(INFO['support_plane_z_m'])+.005-z)*gate
    values=torch.cat((proximity,plane),1)
    k=max(1,int(np.ceil(.1*values.shape[1])))
    return values.mean(1)+values.topk(k,dim=1).values.mean(1),values.max(1).values


def before_backward(step,gripper,joints,translation,rotation,q_raw,total):
    weight=float(META.get('environment_weight',0))
    env_energy=total.detach()*0;env_max=env_energy
    if SCENE is not None:
        with torch.set_grad_enabled(weight>0):
            env_energy,env_max=environment(gripper.surface_points(joints,translation,rotation))
    ACTIVE['steps'].append(dict(step=int(step),state_sha256=digest(translation,rotation,q_raw),
        total_sha256=digest(total),total=total.detach().cpu().tolist(),
        environment_max=env_max.detach().cpu().tolist(),environment_energy=env_energy.detach().cpu().tolist()))
    if step in (0,49,99,199):
        ACTIVE['snapshots'].append(dict(step=int(step),translation=translation.detach().cpu().tolist(),
            rotation_6d=rotation.detach().cpu().tolist(),q_raw=q_raw.detach().cpu().tolist()))
    return total+weight*env_energy if weight else total


def backward(total,translation,rotation,q_raw):
    parameters=[translation,rotation,q_raw]
    if META.get('repeat_backward') and ACTIVE['steps'][-1]['step']==0:
        total.mean().backward(retain_graph=True)
        first=[p.grad.detach().clone() for p in parameters]
        for p in parameters:p.grad=None
        total.mean().backward()
        ACTIVE['same_graph_backward']=dict(first=digest(*first),second=digest(*[p.grad for p in parameters]),
            max_abs=[float((x-p.grad).abs().max()) for x,p in zip(first,parameters)])
    else:total.mean().backward()
    gradients=[p.grad for p in parameters]
    ACTIVE['steps'][-1]['gradient_sha256']=digest(*gradients)
    if ACTIVE['steps'][-1]['step']==0:
        ACTIVE['first_gradients']=[p.detach().cpu().tolist() for p in gradients]


def main():
    global ACTIVE,SCENE,INFO,T,REF
    UP.sample_contact_targets=globals()['sample_contact_targets']
    if META.get('scene'):
        REF=load(C/'scripts/refine_contactdiffusion_environment_visible_scene.py','fk_trace_environment')
        SCENE,INFO=REF.prepare_visible_environment(np.load(META['scene']),np.load(META['object_robot']),
            crop_margin=.35,voxel=.008,maximum=4096)
        T=np.asarray(META['robot_from_object'],dtype=np.float32)
        assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5)
    original=UP.optimize_gripper_to_contacts
    source=inspect.getsource(original)
    anchor='        optimizer.zero_grad(set_to_none=True)\n        total.mean().backward()\n        optimizer.step()'
    assert source.count(anchor)==1
    source=source.replace(anchor,
        '        total = _before_backward(step_index, gripper, joints, translation, rotation_6d, q_raw, total)\n'
        '        optimizer.zero_grad(set_to_none=True)\n'
        '        _trace_backward(total, translation, rotation_6d, q_raw)\n'
        '        optimizer.step()')
    scope=dict(original.__globals__);scope.update(_before_backward=before_backward,_trace_backward=backward)
    exec(compile(source,'<isolated-fk-trace>','exec'),scope)
    implementation=scope['optimize_gripper_to_contacts']
    def optimize(*args,**kwargs):
        global ACTIVE
        gripper,contacts,obj,q0=args[:4]
        links=sorted(gripper.surface_points_local)
        ACTIVE=dict(seed=kwargs['seed'],input_sha256=digest(contacts,obj,q0),
            normals_sha256=digest(kwargs['object_normals']),confidence_sha256=digest(kwargs['object_normal_confidence']),
            hand_surface_sha256=digest(*[gripper.surface_points_local[k] for k in links]),surface_links=links,
            deterministic=torch.are_deterministic_algorithms_enabled(),steps=[],snapshots=[])
        TRACES.append(ACTIVE)
        try:
            solved=implementation(*args,**kwargs)
            ACTIVE['initialization_state_sha256']=solved['initialization_state_sha256']
            ACTIVE['final_candidates']=[{k:c[k] for k in ('particle','root_pose','joint_positions','contact_chamfer_m')} for c in solved['candidates']]
            if SCENE is not None:
                candidates=solved['candidates']
                poses=torch.tensor([c['root_pose'] for c in candidates],device=contacts.device,dtype=contacts.dtype)
                joints=torch.tensor([c['joint_positions'] for c in candidates],device=contacts.device,dtype=contacts.dtype)
                rot=torch.cat((poses[:,:3,0],poses[:,:3,1]),dim=1)
                with torch.no_grad():energy,maximum=environment(gripper.surface_points(joints,poses[:,:3,3],rot))
                ACTIVE['final_environment']=[dict(particle=c['particle'],maximum=float(m),energy=float(e)) for c,m,e in zip(candidates,maximum,energy)]
            return solved
        except Exception as error:
            ACTIVE['error']=repr(error);raise
        finally:
            (DEST/f'trace_{len(TRACES)-1}.json').write_text(json.dumps(ACTIVE,indent=2))
    UP.optimize_gripper_to_contacts=optimize
    UP.main()


if __name__=='__main__':main()
