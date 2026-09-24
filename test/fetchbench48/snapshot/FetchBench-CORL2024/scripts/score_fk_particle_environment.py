"""CPU-only observed-scene scoring of fixed candidates, no optimization/physics."""
import concurrent.futures
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
P=Path('/inspire/qb-ilm2/project/zhanghanbo/public/mck');C=P/'ContactDiffusion'
ROOT=C/'outputs/fk_only_particle_ranking_20260915'
SOURCE=C/'outputs/shampoo_drawer_e3_all48views_s4p4_fk200_env200_20260914'


def worker(key):
    import numpy as np
    import torch
    import yaml
    from scipy.spatial import cKDTree
    from scipy.special import expit
    sys.path.insert(0,str(C));os.chdir(C);torch.set_num_threads(1)
    from utils.multigripper_fk import load_gripper_from_calibration,ordered_joint_values
    spec=importlib.util.spec_from_file_location('visible',C/'scripts/refine_contactdiffusion_environment_visible_scene.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    with gzip.open(ROOT/'source_dataset.json.gz','rt') as f:data=json.load(f)
    manifest=json.loads((SOURCE/'manifest.json').read_text());condition,hand=key
    config=yaml.safe_load(Path(manifest['config_paths']['E3'][hand]).read_text())
    name='Barrett' if hand=='barrett' else 'shadow_hand'
    torch.manual_seed(20260808);np.random.seed(20260808)
    gripper=load_gripper_from_calibration(name,config,config['paths']['tip_offset_calibration'],device='cpu')
    close=np.array(ordered_joint_values(config['grippers'][name]['close_dir'],gripper.joint_names,label='close_dir'))
    output=[];cache={};start=time.time()
    for r in data['rows']:
        if (r['condition'],r['hand'])!=key:continue
        view=r['view']
        if view not in cache:
            paths=manifest['inputs'][view]
            scene_path=next(p for p in paths if p.endswith('scene_partial_robot_base.npy'))
            object_path=next(p for p in paths if p.endswith('sam3d_fused_robot_base.npy'))
            scene,info=ref.prepare_visible_environment(np.load(scene_path),np.load(object_path),crop_margin=.35,voxel=.008,maximum=4096)
            cache[view]=(cKDTree(scene),info)
        tree,info=cache[view]
        cs=r['candidates'];q=np.array([c['joint_positions'] for c in cs],dtype=np.float32)
        lower=gripper.lower.numpy();upper=gripper.upper.numpy()
        opened=np.where(close>0,lower,upper);closed=np.where(close>0,upper,lower)
        outer=np.where(close[None]!=0,q+.10*(opened-q),q)
        inner=np.where(close[None]!=0,q+.20*(closed-q),q)
        sweep=outer[:,None]+np.linspace(0,1,4,dtype=np.float32)[None,:,None]*(inner-outer)[:,None]
        # Nominal, followed by the four execution-closure sweep poses.
        allq=np.concatenate((q[:,None],sweep),axis=1)
        poses=np.asarray([c['root_pose'] for c in cs],dtype=np.float32)
        rawposes=poses@np.linalg.inv(gripper.base_alignment.numpy())
        trans=np.repeat(rawposes[:,None,:3,3],5,axis=1).reshape(20,3)
        rotation=np.concatenate((rawposes[:,:3,0],rawposes[:,:3,1]),axis=1)
        rotation=np.repeat(rotation[:,None],5,axis=1).reshape(20,6)
        with torch.no_grad():
            surface=gripper.surface_points(torch.tensor(allq.reshape(20,-1)),torch.tensor(trans),torch.tensor(rotation)).numpy().reshape(4,5,-1,3)
        T=np.asarray(r['robot_from_object']);robot=surface@T[:3,:3].T+T[:3,3]
        nearest=tree.query(robot.reshape(-1,3),workers=1)[0].reshape(robot.shape[:-1])
        proximity=np.maximum(.005-nearest,0)
        lo=np.array(info['table_xy_min']);hi=np.array(info['table_xy_max'])
        plane=np.maximum(info['support_plane_z_m']+.005-robot[...,2],0)
        xy=robot[...,:2]
        plane*=np.prod(expit((xy-lo)/.01)*expit((hi-xy)/.01),axis=-1)
        violations=np.concatenate((proximity,plane),axis=2)
        for i,c in enumerate(cs):
            feat={}
            for name,vals in [('nominal',violations[i,0]),('outer',violations[i,1]),('inner',violations[i,-1]),('sweep',violations[i,1:].reshape(-1))]:
                k=max(1,int(np.ceil(.1*len(vals))))
                feat.update({f'env_{name}_mean':float(vals.mean()),f'env_{name}_max':float(vals.max()),
                    f'env_{name}_cvar':float(np.partition(vals,-k)[-k:].mean()),f'env_{name}_fraction':float((vals>0).mean())})
            feat['env_sweep_nearest']=float(nearest[i,1:].min())
            feat['support_plane_observed']=float(info['support_plane_observed'])
            assert np.isfinite(list(feat.values())).all()
            output.append(dict(view=view,condition=condition,hand=hand,sample=r['sample'],particle=c['particle'],features=feat))
    path=ROOT/f'environment_features_{condition}_{hand}.json'
    path.write_text(json.dumps(dict(rows=output,seconds=time.time()-start,geometry='selected_view_partial_scene',no_pose_changes=True),indent=2)+'\n')
    return str(path),len(output),time.time()-start


if __name__=='__main__':
    with concurrent.futures.ProcessPoolExecutor(4) as pool:
        for result in pool.map(worker,[('A','barrett'),('A','shadowhand'),('B','barrett'),('B','shadowhand')]):print(result,flush=True)
