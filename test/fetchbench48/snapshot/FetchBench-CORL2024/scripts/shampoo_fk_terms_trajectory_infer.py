"""Separated FK terms and true 0..200 optimizer-state recording.

No independent ENV stage. ENV_PROBE_META selects baseline/environment/obstacles.
Trajectory arrays contain initialization and EVERY completed Adam update, not
interpolations of the old four energy snapshots. Designed for paired replay.
"""
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch

import importlib.util


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


META=json.loads(Path(os.environ['ENV_PROBE_META']).read_text())
JOINT=load(Path(__file__).with_name('shampoo_joint_fk_infer.py'),'frozen_joint_helpers')
E3=JOINT.E3;UP=JOINT.UP
sample_contact_targets=E3.sample_contact_targets
MODE=META['separated_mode']
assert MODE in ('baseline','environment','obstacles')
WEIGHT=float(META['joint_environment_weight'])
assert (MODE=='environment' and WEIGHT in (1.,3.,10.)) or (MODE!='environment' and WEIGHT==0.)
ACTIVE=None
STATES=[]
SERIAL=0


def capture(completed_updates,gripper,translation,rotation,q_raw):
    with torch.no_grad():
        joints=gripper.constrain_joints(q_raw)
        STATES.append(dict(step=int(completed_updates),
            translation=translation.detach().cpu().numpy().copy(),
            rotation_6d=rotation.detach().cpu().numpy().copy(),
            q_raw=q_raw.detach().cpu().numpy().copy(),
            joints=joints.detach().cpu().numpy().copy(),
            root_pose=gripper.root_pose_matrix(translation,rotation).detach().cpu().numpy().copy()))


def separated_term(step,steps,gripper,joints,translation,rotation,total,penetration,surface_match):
    zero=torch.zeros_like(total)
    environment=zero;obstacle=zero
    ramp=max(0.,min(1.,(step/max(steps-1,1)-.25)/.75))
    if MODE=='environment':
        from utils.multigripper_fk import ordered_joint_values
        spec=JOINT.CONFIG['grippers'][META['gripper']]
        close=joints.new_tensor(ordered_joint_values(spec['close_dir'],gripper.joint_names,label='close_dir'))
        opening=torch.where(close>0,gripper.lower,gripper.upper)
        closing=torch.where(close>0,gripper.upper,gripper.lower)
        outer=torch.where(close[None]!=0,joints+.10*(opening[None]-joints),joints)
        inner=torch.where(close[None]!=0,joints+.20*(closing[None]-joints),joints)
        alpha=torch.linspace(0,1,4,device=joints.device,dtype=joints.dtype)
        sweep=outer[:,None]+alpha[None,:,None]*(inner-outer)[:,None]
        b=len(joints)
        surface=gripper.surface_points(sweep.reshape(b*4,-1),
            translation[:,None].expand(-1,4,-1).reshape(b*4,-1),
            rotation[:,None].expand(-1,4,-1).reshape(b*4,-1)).reshape(b,-1,3)
        environment=WEIGHT*E3.environment_metrics(E3.robot_points(surface))['energy']
    elif MODE=='obstacles':
        assert surface_match is not None
        gap=surface_match['assigned_finger_distances'].max(1).values
        obstacle=ramp*(4*(torch.relu(penetration['max']-.007)/.007).square()
                       +10*(torch.relu(gap-.010)/.010).square())
    extra=environment+obstacle
    assert bool(torch.isfinite(extra).all()), 'Nonfinite separated energy'
    ACTIVE['energy_before_update'].append(dict(step=int(step),ramp=ramp,
        original=total.detach().cpu().tolist(),environment=environment.detach().cpu().tolist(),
        obstacles=obstacle.detach().cpu().tolist(),extra=extra.detach().cpu().tolist()))
    return total if MODE=='baseline' else total+extra


def main():
    UP.sample_contact_targets=globals()['sample_contact_targets']
    original=UP.optimize_gripper_to_contacts
    source=inspect.getsource(original)
    seed_anchor='    initial_state_digest = hashlib.sha256()'
    backward_anchor='        optimizer.zero_grad(set_to_none=True)\n        total.mean().backward()\n        optimizer.step()'
    assert source.count(seed_anchor)==source.count(backward_anchor)==1
    source=source.replace(seed_anchor,
        '    translation_init = _seed(gripper, translation_init, rotation_init, q_raw_init)\n'+seed_anchor)
    source=source.replace(backward_anchor,
        '        if step_index == 0:\n'
        '            _capture(0, gripper, translation, rotation_6d, q_raw)\n'
        '        total = _term(step_index, int(steps), gripper, joints, translation, rotation_6d, total, penetration_terms, surface_match)\n'
        +backward_anchor+'\n'
        '        _capture(step_index + 1, gripper, translation, rotation_6d, q_raw)')
    scope=dict(original.__globals__)
    scope.update(_seed=E3.improve_initialization,_capture=capture,_term=separated_term)
    exec(compile(source,'<isolated-fk-terms-full-trajectory>','exec'),scope)
    implementation=scope['optimize_gripper_to_contacts']

    def optimize(*args,**kwargs):
        global ACTIVE,STATES,SERIAL
        gripper=args[0];slot=SERIAL;SERIAL+=1;STATES=[]
        ACTIVE=dict(mode=MODE,weight=WEIGHT,seed=kwargs['seed'],energy_before_update=[],
            input_sha256=JOINT.digest(*args[1:4]),joint_names=gripper.joint_names,
            frame='object_sam3d_surface_centered',root_pose_includes_base_alignment=True)
        result=implementation(*args,**kwargs)
        assert [s['step'] for s in STATES]==list(range(201))
        ACTIVE['initialization_state_sha256']=result['initialization_state_sha256']
        ACTIVE['frames']=201
        dest=Path(META['audit']).parent
        data={k:np.stack([s[k] for s in STATES]) for k in ('translation','rotation_6d','q_raw','joints','root_pose')}
        data['step']=np.arange(201,dtype=np.int32)
        assert all(np.isfinite(v).all() for v in data.values())
        final={c['particle']:c for c in result['candidates']}
        for particle,c in final.items():
            np.testing.assert_allclose(data['root_pose'][-1,particle],c['root_pose'],atol=1e-5)
            np.testing.assert_allclose(data['joints'][-1,particle],c['joint_positions'],atol=1e-6)
        np.savez_compressed(dest/f'trajectory_{slot}.npz',**data)
        (dest/f'trajectory_{slot}.json').write_text(json.dumps(ACTIVE,indent=2))
        result['separated_protocol']=dict(mode=MODE,environment_weight=WEIGHT,env_steps=0,trajectory_frames=201)
        return result

    UP.optimize_gripper_to_contacts=optimize
    try:UP.main()
    finally:E3.save_audit()


if __name__=='__main__':main()
