"""New isolated cohort using immutable historical algorithms and fixed rankers."""
import argparse
import concurrent.futures as cf
import fcntl
import os
from pathlib import Path
import queue
import time
from run_fk_top1_extension6_20260915 import load, read, save

HERE=Path(__file__).resolve().parent
F=HERE.parent;C=F.parent/'ContactDiffusion'
OUT=C/'outputs/fk_top1_extension6_round3_ab_dro64_20260916'
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',SAM3D_CAMERA_CONVENTION='opencv',SAM3D_REUSE_EXISTING_NORMALIZATION='1')
import sys
sys.path.insert(0,str(OUT/'ranker_pydeps'))

def modules():
 driver=load('round3_driver',HERE/'run_fk_top1_extension6_20260915.py');driver.OUT=OUT;driver.__file__=str(Path(__file__).resolve())
 recovery=load('round3_recovery',HERE/'supplement_extension6_20260916.py');recovery.OUT=OUT;recovery.BASE=OUT
 return driver,recovery

def prepare(m,ref,rec,env):
 resident=load('round3_cache',HERE/'sam3d_resident_batch.py');coverage=[];manifests=[]
 for c in m['cases']:
  obs=Path(c['observations']);legal=read(obs/'visibility/legal_partial_views.json');good=[];bad=[]
  for v in legal['views']:
   view=Path(v['rgbd_capture_dir']).name
   (good if resident.cache_valid(dict(capture_dir=str(obs/'visibility/rgbd_views'/view),output_dir=str(obs/'sam3d'/view))) else bad).append(v)
  coverage.append(dict(case=c['id'],legal_views=len(legal['views']),reconstructed_views=len(good),failed_reconstruction_views=[Path(v['rgbd_capture_dir']).name for v in bad]))
  if legal['views']:
   rec.execute([m['runtime']['contact_python'],HERE/'materialize_fetchbench_world_to_robot_base.py','--source-root',obs,'--output-root',OUT/'dro_inputs'/c['id'],'--task-config',c['task_config'],'--task-index',c['task_index'],'--skip-sam3d'],env,OUT/'logs'/f"materialize_dro_{c['id']}.log",C)
  if not good:continue
  partial=OUT/'finite_observations'/c['id'];rec.link(partial/'visibility/rgbd_views',obs/'visibility/rgbd_views');rec.link(partial/'sam3d',obs/'sam3d')
  save(partial/'visibility/legal_partial_views.json',dict(legal,views=good))
  rec.execute([m['runtime']['contact_python'],HERE/'materialize_fetchbench_world_to_robot_base.py','--source-root',partial,'--output-root',OUT/'generation'/c['id']/'corrected_inputs','--task-config',c['task_config'],'--task-index',c['task_index']],env,OUT/'logs'/f"materialize_fk_{c['id']}.log",C)
  manifests.append(ref.prepare_case(dict(c,observations=str(partial)),m))
 save(OUT/'coverage.json',dict(rows=coverage));return manifests,coverage

def main(preflight_only=False):
 lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 m=read(OUT/'manifest.json');driver,rec=modules();driver.status('preflight');icd,ref=driver.preflight(m)
 if preflight_only:driver.status('preflight_passed');return
 driver.status('capture_all',gpus=[0,1,2,3])
 def capworker(gpu):
  for c in m['cases'][gpu::4]:driver.capture(c,gpu,icd)
 with cf.ThreadPoolExecutor(4) as pool:list(pool.map(capworker,range(4)))
 counts={c['id']:read(Path(c['observations'])/'visibility/capture_complete.json')['legal_views'] for c in m['cases']}
 save(OUT/'capture_status.json',dict(stage='complete',legal_views=counts,time=time.time()))
 env=ref.base.base_env();env['LD_LIBRARY_PATH']+=':/usr/local/cuda-12.8/lib64';env['SAM3D_REUSE_EXISTING_NORMALIZATION']='1'
 jobs=[]
 for c in m['cases']:
  obs=Path(c['observations'])
  for v in read(obs/'visibility/legal_partial_views.json')['views']:
   name=Path(v['rgbd_capture_dir']).name;jobs.append(dict(id=c['id']+'__'+name,capture_dir=str(obs/'visibility/rgbd_views'/name),output_dir=str(obs/'sam3d'/name)))
 driver.status('sam3d',gpus=[0,1,2,3],views=len(jobs))
 with cf.ThreadPoolExecutor(4) as pool:
  futures=[pool.submit(rec.attempt,'sam3d',gpu,rec.sam_worker,gpu,jobs[gpu::4],m,env) for gpu in range(4)]
  for future in futures:future.result()
 manifests,coverage=prepare(m,ref,rec,env)
 slots=queue.Queue()
 for _ in range(2):
  for gpu in range(4):slots.put(gpu)
 with cf.ThreadPoolExecutor(12) as vp:
  for manifest in manifests:
   driver.status('fk_generation_validation',case=manifest['case']['id'],generation_workers=8,validation_workers=12,gpus=[0,1,2,3])
   rec.fk_case(manifest,ref,slots,vp,workers=8)
 drojobs=[]
 for c in m['cases']:
  for v in read(Path(c['observations'])/'visibility/legal_partial_views.json')['views']:
   name=Path(v['rgbd_capture_dir']).name
   for hand in ('barrett','shadowhand'):
    dest=OUT/'dro'/c['id']/'views'/name/'simulation'/c['scene_factory']/('task_%03d'%c['task_index'])/hand
    drojobs.append(dict(case=c,view=name,hand=hand,artifact=str(dest)))
 driver.status('dro_generation_validation',generation_workers=8,validation_workers=12,gpus=[0,1,2,3],batches=len(drojobs))
 def gen(job):
  gpu=slots.get()
  try:return rec.dro_generate(job,gpu,m,env)
  finally:slots.put(gpu)
 with cf.ThreadPoolExecutor(8) as gp,cf.ThreadPoolExecutor(12) as vp:
  pending={gp.submit(gen,j):j for j in drojobs};validation=[]
  for future in cf.as_completed(pending):
   j=pending[future]
   try:
    future.result();rec.event('dro_generate',case=j['case']['id'],view=j['view'],hand=j['hand'],ok=True)
    validation.append(vp.submit(rec.attempt,'dro_validate',dict(case=j['case']['id'],view=j['view'],hand=j['hand']),rec.dro_valid,j,m,env))
   except Exception as ex:rec.event('dro_generate',case=j['case']['id'],view=j['view'],hand=j['hand'],ok=False,error=repr(ex))
  for future in validation:future.result()
 for manifest in manifests:
  driver.status('ranking',case=manifest['case']['id']);rec.attempt('ranking',manifest['case']['id'],driver.score_case,manifest)
 errors=[r for r in rec.EVENTS if r.get('ok') is False]
 driver.summary(ref,m,coverage,errors)
 driver.status('finished',complete=read(OUT/'RESULTS.json')['complete'],errors=len(errors),note='Unavailable/contact-budget failures are separate from validated physical failures.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--feature-case');p.add_argument('--condition');p.add_argument('--hand');args=p.parse_args()
 if args.feature_case:
  worker=load('round3_features',HERE/'recover_six_scene_existing_top1.py');worker.feature_worker(OUT/'top1',args.feature_case,args.condition,args.hand)
 else:
  try:main(args.preflight_only)
  except Exception as ex:save(OUT/'pipeline_status.json',dict(stage='failed',error=repr(ex),time=time.time()));raise
