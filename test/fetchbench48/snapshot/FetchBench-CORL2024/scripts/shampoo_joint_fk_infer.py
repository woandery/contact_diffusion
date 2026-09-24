"""Isolated joint FK + observed-environment objective; never runs ENV refinement.

Loaded through the frozen A/B condition adapter. E3 contact screening and seed
repair are reused. No source/config/checkpoint is edited in ContactDiffusion.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
import importlib.util


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


META = json.loads(Path(os.environ['ENV_PROBE_META']).read_text())
E3 = load(Path(__file__).with_name('shampoo_environment_probe_infer_20260914.py'), 'joint_e3_seed')
UP = E3.UP
sample_contact_targets = E3.sample_contact_targets
WEIGHT = float(META['joint_environment_weight'])
CONFIG = yaml.safe_load(Path(META['config']).read_text())
ACTIVE = None
TRACES = []


def digest(*values):
    h = hashlib.sha256()
    for value in values:
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def joint_term(step, steps, gripper, joints, translation, rotation, total,
               penetration, surface_match):
    if WEIGHT == 0:
        if step in (0, 49, 99, 199):
            ACTIVE['steps'].append(dict(step=step, original_total=total.detach().cpu().tolist()))
        return total
    spec = CONFIG['grippers'][META['gripper']]
    from utils.multigripper_fk import ordered_joint_values
    close = joints.new_tensor(ordered_joint_values(spec['close_dir'], gripper.joint_names, label='close_dir'))
    opening = torch.where(close > 0, gripper.lower, gripper.upper)
    closing = torch.where(close > 0, gripper.upper, gripper.lower)
    outer = torch.where(close[None] != 0, joints + .10 * (opening[None] - joints), joints)
    inner = torch.where(close[None] != 0, joints + .20 * (closing[None] - joints), joints)
    alpha = torch.linspace(0, 1, 4, device=joints.device, dtype=joints.dtype)
    swept_joints = outer[:, None] + alpha[None, :, None] * (inner - outer)[:, None]
    batch = len(joints)
    surface = gripper.surface_points(swept_joints.reshape(batch * 4, -1),
        translation[:, None].expand(-1, 4, -1).reshape(batch * 4, -1),
        rotation[:, None].expand(-1, 4, -1).reshape(batch * 4, -1))
    env = E3.environment_metrics(E3.robot_points(surface.reshape(batch, -1, 3)))
    assert surface_match is not None, 'Joint barriers require distal-surface matching'
    contact_max = surface_match['assigned_finger_distances'].max(1).values
    object_barrier = (torch.relu(penetration['max'] - .007) / .007).square()
    contact_barrier = (torch.relu(contact_max - .010) / .010).square()
    environment_barrier = (torch.relu(env['maximum'] - .0001) / .005).square()
    # Same normalized thresholds and 25% warmup as the old ENV barriers.
    # Environment mean+CVaR acts at every FK step, including the warmup.
    ramp = max(0., min(1., (step / max(steps - 1, 1) - .25) / .75))
    extra = WEIGHT * env['energy'] + ramp * (
        4 * object_barrier + 10 * contact_barrier + (4 * WEIGHT / 10) * environment_barrier)
    if not bool(torch.isfinite(extra).all()):
        raise FloatingPointError('Nonfinite joint objective')
    if step in (0, 49, 99, 199):
        row = dict(step=step, ramp=ramp)
        for key, value in dict(original_total=total, extra=extra, environment_energy=env['energy'],
            environment_max_m=env['maximum'], contact_max_m=contact_max,
            object_max_m=penetration['max'], object_barrier=object_barrier,
            contact_barrier=contact_barrier, environment_barrier=environment_barrier).items():
            row[key] = value.detach().cpu().tolist()
        ACTIVE['steps'].append(row)
    return total + extra


def main():
    UP.sample_contact_targets = globals()['sample_contact_targets']
    original = UP.optimize_gripper_to_contacts
    source = inspect.getsource(original)
    seed_anchor = '    initial_state_digest = hashlib.sha256()'
    backward_anchor = '        optimizer.zero_grad(set_to_none=True)\n        total.mean().backward()\n        optimizer.step()'
    assert source.count(seed_anchor) == source.count(backward_anchor) == 1
    source = source.replace(seed_anchor,
        '    translation_init = _seed_hook(gripper, translation_init, rotation_init, q_raw_init)\n' + seed_anchor)
    source = source.replace(backward_anchor,
        '        total = _joint_hook(step_index, int(steps), gripper, joints, translation, rotation_6d, total, penetration_terms, surface_match)\n'
        + backward_anchor)
    scope = dict(original.__globals__)
    scope.update(_seed_hook=E3.improve_initialization, _joint_hook=joint_term)
    exec(compile(source, '<joint-fk-no-env-stage>', 'exec'), scope)
    implementation = scope['optimize_gripper_to_contacts']

    def optimize(*args, **kwargs):
        global ACTIVE
        ACTIVE = dict(weight=WEIGHT, steps=[], input_sha256=digest(*args[1:4]),
            normals_sha256=digest(kwargs['object_normals']), seed=kwargs['seed'])
        TRACES.append(ACTIVE)
        try:
            result = implementation(*args, **kwargs)
            ACTIVE['initialization_state_sha256'] = result['initialization_state_sha256']
            result['joint_environment_protocol'] = dict(weight=WEIGHT, independent_env_steps=0,
                closure_sweep_samples=4, closure_fractions=[.10,.20],
                geometry='selected_view_partial_scene', barriers='obj7mm/contact10mm/env0.1mm',
                ranking='unchanged FK scores exclude new energy; all particles are executed')
            return result
        finally:
            Path(META['audit']).with_name(f'joint_trace_{len(TRACES)-1}.json').write_text(json.dumps(ACTIVE, indent=2))

    UP.optimize_gripper_to_contacts = optimize
    try:
        UP.main()
    finally:
        E3.save_audit()


if __name__ == '__main__':
    main()
