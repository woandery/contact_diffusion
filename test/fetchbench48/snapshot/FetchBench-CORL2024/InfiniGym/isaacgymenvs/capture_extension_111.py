"""111 independent RGB-D observations; no video, plots or interactive viewer.

Visibility mathematics reuse the archived HANDONLY benchmark implementation.
All geometry saved here is WORLD; only offline coverage reads the true mesh.
"""
import os, sys, json, hashlib
from pathlib import Path
import isaacgym
from isaacgym import gymapi
import hydra
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import isaacgymenvs
from isaacgymenvs.utils.utils import set_seed
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/reference111'))
from analyze_mesh_visibility import _camera_directions,_camera_position,_sample_mesh_surface,_camera_projection,_visible_surface_mask
from render_camera_sweep import _render_camera,_quaternion_xyzw_to_matrix

def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

@hydra.main(version_base='1.1',config_name='config',config_path='./config')
def main(cfg):
    out=Path(os.environ['CAPTURE_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    cfg.seed=set_seed(cfg.seed,torch_deterministic=cfg.torch_deterministic,rank=0)
    cfg.task.task.scene_config_path=cfg.scene.scene_list
    cfg.task.experiment_name='extension111_capture'
    task_index=int(cfg.task.solution.task_index);scene_ref=str(cfg.scene.scene_list[0])
    asset=json.loads((ROOT/'Task'/scene_ref/'asset_config.json').read_text())
    data=np.load(ROOT/'Task'/scene_ref/'task_config.npz')
    obj=int(data['task_obj_index'][task_index]);label=str(data['task_obj_label'][task_index]).removeprefix('rigid_obj_') if sys.version_info>=(3,9) else str(data['task_obj_label'][task_index])[10:]
    scene_dir=ROOT/'benchmark_scenes'/asset['scene_config']['asset_root'].split('/benchmark_scenes/')[1]
    supports=[s for s in json.loads((scene_dir/'support.json').read_text()) if s['label']==label]
    if not supports:raise ValueError('No matching support region for '+label)
    xy=np.concatenate([np.asarray(json.loads(s['polygon'])['coordinates'][0])+np.asarray(s['translation'])[:2] for s in supports])
    local_center=np.r_[(xy.min(0)+xy.max(0))/2,np.mean([s['translation'][2] for s in supports])]
    task=isaacgymenvs.make(cfg.seed,cfg.task_name,cfg.task.env.numEnvs,cfg.sim_device,cfg.rl_device,cfg.graphics_device_id,cfg.headless,cfg.multi_gpu,cfg.capture_video,cfg.force_render,cfg)
    task.reset_task(task_index)
    target_rgb=[float(value) for value in os.environ.get('CAPTURE_TARGET_RGB','1,0,0').split(',')]
    if len(target_rgb)!=3 or not all(0.0<=value<=1.0 for value in target_rgb):
        raise ValueError('CAPTURE_TARGET_RGB must contain three comma-separated values in [0,1]')
    # Segmentation IDs identify the target.  Keep visual recoloring configurable
    # because the legacy red marker can erase object appearance used by SAM3D.
    task.gym.set_rigid_body_color(task.envs[0],task.objects[0][obj],0,gymapi.MESH_VISUAL,gymapi.Vec3(*target_rgb))
    task.gym.refresh_actor_root_state_tensor(task.sim)
    # support.json is in generator coordinates, not URDF actor coordinates.
    # Same mapping as archived render_camera_sweep._load_support_geometry.
    center=local_center*np.asarray([-1.,-1.,1.])
    pose=task._obj_state[0,obj,:7].cpu().numpy();rot=_quaternion_xyzw_to_matrix(pose[3:])
    objdir=ROOT/'benchmark_objects'/asset['object_config'][obj]['asset_root'].split('/benchmark_objects/')[1]
    sample=_sample_mesh_surface(objdir/'mesh.obj',60000,int(cfg.seed)+obj)
    points=sample['points_local']@rot.T+pose[:3];normals=sample['normals_local']@rot.T
    env=task.envs[0];cam=task.cameras[0][0];segid=obj+4
    rows=[];union=np.zeros(60000,dtype=bool)
    for r in [1.,1.5,2.]:
      for el,az in _camera_directions():
        view_id=f'r{r:.1f}_el{el:+.0f}_az{az:+.0f}';p=out/'rgbd_views'/view_id
        position=_camera_position(center,r,el,az)
        # Gym camera local axes: +X forward, +Y left, +Z up. Deterministic poles.
        forward=(center-position)/np.linalg.norm(center-position)
        up=np.array([0.,0.,1.]) if abs(forward[2])<.999 else np.array([1.,0.,0.])
        left=np.cross(up,forward);left/=np.linalg.norm(left);up=np.cross(forward,left)
        q=Rotation.from_matrix(np.stack([forward,left,up],axis=1)).as_quat()
        transform=gymapi.Transform();transform.p=gymapi.Vec3(*position);transform.r=gymapi.Quat(*q)
        task.gym.set_camera_transform(cam,env,transform)
        rgb,depth,seg=_render_camera(task,cam);height,width=depth.shape
        view,K,origin=_camera_projection(task,env,cam,width,height)
        assert np.isfinite(view).all() and abs(np.linalg.det(view)-1)<1e-4
        visible=_visible_surface_mask(points,normals,position,view,K,origin,depth,seg,segid,.002);union|=visible
        yy,xx=np.mgrid[:height,:width];rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(K.T)
        xyz=rays*(-depth[...,None]);hom=np.concatenate([xyz,np.ones((height,width,1))],-1)
        world=(hom@np.linalg.inv(view))[...,:3]-origin
        valid=np.isfinite(depth)&(depth>.15)&(depth<2.5);world[~valid]=np.nan;world=world.astype(np.float32)
        target=world[valid&(seg==segid)];scene=world[valid&(seg!=segid)&(seg!=1)&(seg>0)]
        p.mkdir(parents=True,exist_ok=True)
        np.save(p/'camera_00_depth.npy',-depth);np.save(p/'camera_00_segmentation.npy',seg)
        np.save(p/'camera_00_pointmap_robot_base.npy',world)
        np.save(p/'target_partial_robot_base.npy',target);np.save(p/'scene_partial_robot_base.npy',scene)
        Image.fromarray(rgb).save(p/'camera_00_rgb.png');Image.fromarray((seg==segid).astype(np.uint8)*255).save(p/'camera_00_target_mask.png')
        def pcmeta(name,a):
            return dict(path=str(p/name),points=len(a),sha256=sha(a),bounds_min_m=a.min(0).tolist() if len(a) else None,bounds_max_m=a.max(0).tolist() if len(a) else None)
        intr=dict(width=width,height=height,z_near=.15,z_far=2.5,fx=float(-K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]))
        md=dict(coordinate_frame='fetchbench_world',task_index=task_index,target_object_index=obj,target_segmentation_id=segid,scene_config_path=scene_ref,target_visual_rgb=target_rgb,
            target=pcmeta('target_partial_robot_base.npy',target),scene_without_target_or_robot=pcmeta('scene_partial_robot_base.npy',scene),
            rgbd_views=[dict(camera_index=0,rgb_path=str(p/'camera_00_rgb.png'),target_mask_path=str(p/'camera_00_target_mask.png'),depth_path=str(p/'camera_00_depth.npy'),pointmap_robot_base_path=str(p/'camera_00_pointmap_robot_base.npy'),intrinsics=intr,target_mask_pixels=int((seg==segid).sum()))],
            camera_position_world=position.tolist(),camera_quaternion_xyzw=q.tolist(),view_matrix=view.tolist(),projection_matrix=np.asarray(task.gym.get_camera_proj_matrix(task.sim,env,cam)).tolist(),object_state=pose.tolist(),support_center_world=center.tolist(),support_label=label)
        dump(p/'metadata.json',md)
        rows.append(dict(view_index=len(rows),object_index=obj,category=objdir.parent.name,radius_m=r,elevation_deg=el,azimuth_deg=az,visible_sample_count=int(visible.sum()),partial_raw_point_count=len(target),rgbd_capture_dir=str(p)))
        print('CAPTURE',view_id,len(target),int(visible.sum()),flush=True)
    for row in rows:
        row['fraction_full']=row['visible_sample_count']/60000
        row['fraction_observable']=row['visible_sample_count']/int(union.sum()) if union.any() else 0.
    legal=[r for r in rows if r['fraction_observable']>=.05 and r['partial_raw_point_count']>0]
    common=dict(scene=str(cfg.scene.name),task_index=task_index,scene_ref=scene_ref,coordinate_frame='fetchbench_world',support_center_world=center.tolist(),support_label=label,target_visual_rgb=target_rgb,candidate_views=111,observable_union_count=int(union.sum()))
    assert len(rows)==111
    dump(out/'all_views.json',dict(**common,views=rows));dump(out/'legal_partial_views.json',dict(**common,views=legal))
    dump(out/'capture_complete.json',dict(complete=True,candidate_views=111,legal_views=len(legal),cleanup='process exit after flushed output; Gym destruction crashes on this runtime'))
    print('CAPTURE_COMPLETE',len(legal),flush=True)
    # Avoid known renderer teardown segfault. OS reclaims this isolated process;
    # this path is reached ONLY after all 111 outputs and manifests are written.
    sys.stdout.flush();sys.stderr.flush();os._exit(0)

if __name__=='__main__':main()
