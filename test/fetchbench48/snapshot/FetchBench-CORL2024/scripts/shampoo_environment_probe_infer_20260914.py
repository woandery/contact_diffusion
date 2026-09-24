"""Process-local inference adapter: bounded contact refill and safe translation seeds.

Loaded by the unchanged A/B condition adapters using CONTACTDIFF_INFER_SOURCE.
No production module or frozen configuration is edited on disk.
"""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


META = json.loads(Path(os.environ['ENV_PROBE_META']).read_text())
C = Path(META['contact_root'])
UP = load(C/'scripts/infer_local_contactdiffusion_grasp.py', 'environment_probe_original')
REF = load(C/'scripts/refine_contactdiffusion_environment_visible_scene.py', 'environment_probe_visible')
T = np.asarray(META['robot_from_object'], dtype=np.float32)
assert T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1])
assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
SCENE, INFO = REF.prepare_visible_environment(
    np.load(META['scene']), np.load(META['object_robot']),
    crop_margin=.35, voxel=.008, maximum=4096)
AUDIT = dict(schema='environment-probe-v1', arm=META['arm'],
             coordinate_frame='robot_base', environment=INFO,
             contact_clearance_m=.005, initialization_translation_limit_m=.04,
             draw_budget=32, draws=[], initializations=[], budget_exhausted=False)
AUDIT_PATH = Path(META['audit'])
SLOT = 0


def save_audit():
    AUDIT_PATH.write_text(json.dumps(AUDIT, indent=2)+'\n')


def robot_points(points):
    return points @ torch.as_tensor(T[:3, :3].T, device=points.device) + torch.as_tensor(T[:3, 3], device=points.device)


def environment_metrics(surface):
    """Same mean+CVaR energy as ENV refinement, with bounded cdist memory."""
    scene = torch.as_tensor(SCENE, device=surface.device, dtype=surface.dtype)
    nearest = torch.cat([torch.cdist(chunk, scene[None]).min(2).values
                         for chunk in surface.split(128, dim=1)], dim=1)
    proximity = torch.relu(.005-nearest)
    lower = torch.as_tensor(INFO['table_xy_min'], device=surface.device)
    upper = torch.as_tensor(INFO['table_xy_max'], device=surface.device)
    x, y, z = surface.unbind(2)
    gate = (torch.sigmoid((x-lower[0])/.01)*torch.sigmoid((upper[0]-x)/.01)
            *torch.sigmoid((y-lower[1])/.01)*torch.sigmoid((upper[1]-y)/.01))
    plane = torch.relu(float(INFO['support_plane_z_m'])+.005-z)*gate
    violations = torch.cat((proximity, plane), 1)
    k = max(1, int(np.ceil(.1*violations.shape[1])))
    return dict(energy=violations.mean(1)+violations.topk(k, dim=1).values.mean(1),
                maximum=violations.max(1).values, nearest=nearest.min(1).values)


class ContactBudgetExhausted(RuntimeError):
    pass


def sample_contact_targets(*args, **kwargs):
    global SLOT
    slot = SLOT
    SLOT += 1
    base_seed = torch.initial_seed()
    attempt = 0
    while len(AUDIT['draws']) < 32:
        draw_seed = base_seed if attempt == 0 else base_seed+100000+attempt*1000
        if attempt:
            torch.manual_seed(draw_seed)
            torch.cuda.manual_seed_all(draw_seed)
        result = UPSTREAM_SAMPLE(*args, **kwargs)
        targets, raw = result[0], result[1]
        with torch.no_grad():
            world = robot_points(targets)
            nearest = torch.cdist(world[None], torch.as_tensor(SCENE, device=world.device)[None]).min(2).values[0]
            # Keep the plane local to its observed footprint. No hidden geometry.
            xy = world[:, :2]
            in_footprint = ((xy >= torch.as_tensor(INFO['table_xy_min'], device=world.device)) &
                            (xy <= torch.as_tensor(INFO['table_xy_max'], device=world.device))).all(1)
            below = in_footprint & (world[:, 2] < float(INFO['support_plane_z_m'])+.005)
            safe = bool(torch.isfinite(targets).all() and (nearest >= .005).all() and not below.any())
        accepted = safe or META['arm'] not in ('E1', 'E3')
        AUDIT['draws'].append(dict(slot=slot, attempt=attempt, seed=draw_seed, safe=safe,
            accepted=accepted, nearest_scene_m=nearest.cpu().tolist(), below_support=below.cpu().tolist(),
            contacts_object=targets.detach().cpu().tolist(), raw_object=raw.detach().cpu().tolist(),
            contacts_robot=world.detach().cpu().tolist()))
        save_audit()
        if accepted:
            return result
        attempt += 1
    AUDIT['budget_exhausted'] = True
    save_audit()
    raise ContactBudgetExhausted('No safe contact set within 32 total diffusion draws')


UPSTREAM_SAMPLE = UP.sample_contact_targets


def improve_initialization(gripper, translation, rotation6d, q_raw):
    """Keep 4 particles, rotations, joints fixed; search <=4 cm translations.

    This is a bounded geometric seed search, not additional FK optimization.
    Select the smallest feasible displacement, otherwise lowest ENV energy.
    """
    with torch.no_grad():
        surface_object = gripper.surface_points(gripper.constrain_joints(q_raw), translation, rotation6d)
        surface_robot = robot_points(surface_object)
        shifts = [np.zeros(3, dtype=np.float32)]
        for distance in (.01, .02, .04):
            for axis in range(3):
                for sign in (-1, 1):
                    shift = np.zeros(3, dtype=np.float32)
                    shift[axis] = sign*distance
                    shifts.append(shift)
        metrics = [environment_metrics(surface_robot+torch.as_tensor(s, device=translation.device)) for s in shifts]
        chosen = []
        for p in range(len(translation)):
            feasible = [i for i,m in enumerate(metrics) if float(m['maximum'][p]) <= .0001]
            if feasible:
                idx = min(feasible, key=lambda i:(float(np.linalg.norm(shifts[i])), float(metrics[i]['energy'][p])))
            else:
                idx = min(range(len(shifts)), key=lambda i:(float(metrics[i]['energy'][p]), float(np.linalg.norm(shifts[i]))))
            chosen.append(idx)
        selected = np.stack([shifts[i] for i in chosen])
        improved = translation + torch.as_tensor(selected @ T[:3, :3], device=translation.device)
        before = metrics[0]
        after = environment_metrics(robot_points(gripper.surface_points(gripper.constrain_joints(q_raw), improved, rotation6d)))
        assert torch.all(after['energy'] <= before['energy']+1e-6), 'Initialization environment energy increased'
        idx = len(AUDIT['initializations'])
        np.savez_compressed(AUDIT_PATH.parent/f'initialization_{idx}.npz',
            object_robot=np.load(META['object_robot']), scene_robot=SCENE,
            hand_before_robot=surface_robot.cpu().numpy(), hand_after_robot=(surface_robot+torch.as_tensor(selected, device=translation.device)[:,None]).cpu().numpy())
        AUDIT['initializations'].append(dict(slot=idx, shifts_robot_m=selected.tolist(),
            before_energy=before['energy'].cpu().tolist(), after_energy=after['energy'].cpu().tolist(),
            before_max_m=before['maximum'].cpu().tolist(), after_max_m=after['maximum'].cpu().tolist(),
            low_collision_achieved=(after['maximum'] <= .0001).cpu().tolist(),
            fixed_rotations_and_joints=True))
        save_audit()
        return improved


def main():
    # The unchanged A/B adapter replaces this module's sample function first.
    UP.sample_contact_targets = globals()['sample_contact_targets']
    if META['arm'] in ('E2', 'E3'):
        original = UP.optimize_gripper_to_contacts
        source = inspect.getsource(original)
        anchor = '    initial_state_digest = hashlib.sha256()'
        assert source.count(anchor) == 1
        source = source.replace(anchor,
            '    translation_init = _environment_seed_hook(gripper, translation_init, rotation_init, q_raw_init)\n'+anchor)
        namespace = dict(original.__globals__)
        namespace['_environment_seed_hook'] = improve_initialization
        exec(compile(source, '<isolated-environment-seed-hook>', 'exec'), namespace)
        UP.optimize_gripper_to_contacts = namespace['optimize_gripper_to_contacts']
    try:
        UP.main()
    except ContactBudgetExhausted:
        # Preserve completed slots; no unsafe fallback and no invented candidates.
        path = Path(META['output'])
        if not path.exists():
            payload = json.loads(Path(META['baseline_raw']).read_text())
            payload['records'] = []
            path.write_text(json.dumps(payload))
    finally:
        save_audit()
