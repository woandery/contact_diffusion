"""Independent A/B 4x4 and DRO64 branches on the verified two-GPU node."""
import concurrent.futures, fcntl, hashlib, json, os, subprocess, time
from pathlib import Path
import yaml
R=Path(__file__).resolve().parents[1];P=R.parent;C=P/'ContactDiffusion'
SOURCE=C/'outputs/extension3_shelf_basket_drawer_4x4_20260910_v2'
RUN=C/'outputs/extension3_111view_ab4x4_dro64_20260910'
RUN.mkdir(parents=True,exist_ok=True)
lock=(RUN/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def dump(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def status(branch,stage,**kw):dump(RUN/(branch+'_status.json'),dict(stage=stage,time=time.time(),**kw))
def command(cmd,env,log):
 log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a') as f:
  f.write('\nCOMMAND '+json.dumps(cmd)+'\n');f.flush()
  subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
m=json.loads((SOURCE/'manifest.json').read_text());runtime=m['runtime']
assert set(subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).split())=={'0','1'}
for c in m['cases']:
 obs=RUN/'observations_cv'/c['id'];obs.mkdir(parents=True,exist_ok=True)
 if not (obs/'visibility').exists():
  (obs/'visibility').symlink_to(SOURCE/'observations'/c['id']/'visibility',target_is_directory=True)
 c['observations']=str(obs)
 legal=json.loads((Path(c['observations'])/'visibility/legal_partial_views.json').read_text())
 assert legal['task_index']==c['task_index'] and all(v['object_index']==c['object_index'] for v in legal['views'])
 c['legal_views']=len(legal['views'])
assert [c['legal_views'] for c in m['cases']]==[79,45,48]
for key in ['config_barrett','config_shadow']:
 src=Path(runtime[key]);d=yaml.safe_load(src.read_text());old=json.loads(json.dumps(d))
 d['baseline']['protocol_id']='extension3-ab-s4p4-dro64-fk200-env200-w10-20260910'
 for name in ['contact_sets_per_object','contact_sets','num_contact_sets']:
  if name in d['baseline']:d['baseline'][name]=4
 d['baseline']['particles_per_contact_set']=4;d['fk_optimization']['particles']=4
 assert d['fk_optimization']['steps']==200 and d['fk_optimization']['learning_rate']==.0075
 dst=RUN/'configs'/(key+'_s4p4.yaml');dst.parent.mkdir(exist_ok=True)
 dst.write_text(yaml.safe_dump(d,sort_keys=False));runtime[key]=str(dst)
 dump(dst.with_suffix('.audit.json'),dict(source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),original=old,updated=d))
m['experiment_protocol'].update(status='running',methods=['A','B','DRO'],sets=4,particles=4,dro_candidates=64,
 primary_success='final_object_lift_m >= 0.10',gpus=[0,1],cpu_physx=True,record_video=False)
m['experiment_protocol'].pop('reason',None)
m['experiment_protocol'].update(schedule='AB_generation_and_validation_before_DRO',camera_pointmap_convention='opencv',old_proxy_cache_reused=False)
dump(RUN/'manifest.json',m)
env=os.environ.copy();env.update(SAM3D_CAMERA_CONVENTION='opencv',TORCH_HOME=str(SOURCE/'torch_cache'),SAM3D_DINO_REPOSITORY=str(SOURCE/'torch_cache/hub/facebookresearch_dinov2_main'),
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',
 LD_LIBRARY_PATH=f'{SOURCE}/runtime_lib:{P}/miniconda3/envs/contactdiff/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
py=runtime['contact_python'];fetchpy=str(P/'miniconda3/envs/fetchbench/bin/python')
def run_ab():
 sam_gpus=os.environ.get('EXTENSION3_SAM3D_GPUS','0,1')
 assert sam_gpus in ('0','1','0,1')
 status('AB','sam3d_reconstruction',gpus=[int(g) for g in sam_gpus.split(',')])
 command([py,'-u',str(R/'scripts/sam3d_resident_batch.py'),'--manifest',str(RUN/'manifest.json'),'--output',str(RUN/os.environ.get('EXTENSION3_RESIDENT_ATTEMPT','resident_ab_first')),'--gpus',sam_gpus],env,RUN/'logs/sam3d.log')
 status('AB','generation',gpus=[0,1])
 for attempt in range(int(os.environ.get('EXTENSION3_GENERATION_ATTEMPTS','1'))):
  try:
   command([py,'-u',str(R/'scripts/fetchbench_ab_dro_queue.py'),'--manifest',str(RUN/'manifest.json'),'--output',str(RUN/'generation'),
    '--gpus','0,1','--workers-per-gpu',os.environ.get('EXTENSION3_WORKERS_PER_GPU','3'),'--prepare-workers-per-gpu','1','--sets','4','--particles','4','--methods','AB','--run'],env,RUN/'logs/ab_generation.log')
   break
  except subprocess.CalledProcessError:
   if attempt+1==int(os.environ.get('EXTENSION3_GENERATION_ATTEMPTS','1')):raise
 for c in m['cases']:
  status('AB','cpu_physx',case=c['id'])
  e=env.copy();e.update(FETCHBENCH_ROOT=str(R),CONTACTDIFF_ROOT=str(C),CONTACTDIFF_PYTHON=py,FETCHBENCH_PYTHON=fetchpy,
   RUN_ROOT=str(RUN/'generation'/c['id']/'AB'),SOURCE_ROOT=str(RUN/'generation'/c['id']/'corrected_inputs'),
   OBJECT_PREFIX=c['id'],SCENE_ALIAS=c['scene'],TASK_INDEX=str(c['task_index']),POSES_PER_CASE='4',VALIDATION_WORKERS=os.environ.get('EXTENSION3_VALIDATION_WORKERS','2'),
   CONFIG_BARRETT=runtime['config_barrett'],CONFIG_SHADOW=runtime['config_shadow'],TORCH_EXTENSIONS_DIR=str(SOURCE/'torch_extensions'))
  command(['bash',str(R/'scripts/validate_fetchbench_extension_ab_cpu_physx.sh')],e,RUN/'logs'/('ab_validate_'+c['id']+'.log'))
 summaries=list((RUN/'generation').glob('*/AB/cpu_physx_all4/simulation/**/lift_validation/summary.json'))
 assert len(summaries)==sum(c['legal_views'] for c in m['cases'])*4
 assert all(json.loads(p.read_text())['validated_candidates']==4 for p in summaries)
 status('AB','complete')
def run_dro():
 for c in m['cases']:
  corrected=RUN/'dro_inputs'/c['id'];status('DRO','coordinate_materialization',case=c['id'])
  command([py,str(R/'scripts/materialize_fetchbench_world_to_robot_base.py'),'--source-root',c['observations'],'--output-root',str(corrected),
   '--task-config',c['task_config'],'--task-index',str(c['task_index']),'--skip-sam3d'],env,RUN/'logs'/('dro_materialize_'+c['id']+'.log'))
  audit=json.loads((corrected/'coordinate_fix_manifest.json').read_text());assert audit['maximum_roundtrip_error_m']<1e-6
  status('DRO','generation_and_cpu_physx',case=c['id'],gpu=1)
  e=env.copy();e.update(FETCHBENCH_ROOT=str(R),CONTACT_ROOT=str(C),DRO_REPRO_ROOT=str(P/'dro_grasp_reproduction'),DRO_ROOT=runtime['dro_root'],
   DRO_PYDEPS=runtime['dro_pydeps'],DRO_PYTHON=runtime['dro_python'],FETCHBENCH_PYTHON=fetchpy,CHECKPOINT=runtime['checkpoint_dro'],
   INPUT_ROOT=str(corrected),RUN_ROOT=str(RUN/'generation'/c['id']/'DRO_cpu_physx_all64'),TASK_INDEX=str(c['task_index']),
   SCENE_CONFIG=c['scene'],SCENE_FACTORY=c['scene_factory'],OBJECT_LABEL=c['id'],EXPECTED_LEGAL_VIEWS=str(c['legal_views']),
   GPU_COUNT='2',GPU_OFFSET='0',WORKERS_PER_GPU='2',VALIDATION_WORKERS='2',DRO_CANDIDATES='64',GENERATION_ATTEMPTS='4',
   TORCH_EXTENSIONS_DIR=str(SOURCE/'torch_extensions'))
  command(['bash',str(R/'scripts/run_extension3_dro64_worker.sh')],e,RUN/'logs'/('dro_'+c['id']+'.log'))
 status('DRO','complete')
phase=os.environ.get('EXTENSION3_PHASE','ALL')
assert phase in ('ALL','AB','DRO')
if phase=='DRO':
 abstage=json.loads((RUN/'AB_status.json').read_text())['stage']
 if abstage=='complete_finite':
  report=json.loads((RUN/'ab_finite_validation/summary.json').read_text())
  assert report['complete'] and report['finished_batches']==688 and report['planned']==2752
  assert report['executed']==report['finite'] and report['finite']+report['invalid']==2752 and not report['errors']
  for row in report['rows']:
   if row['valid']:
    assert json.loads(Path(row['summary']).read_text())['validated_candidates']==row['valid']
 else:
  assert abstage=='complete'
  summaries=list((RUN/'generation').glob('*/AB/cpu_physx_all4/simulation/**/lift_validation/summary.json'))
  assert len(summaries)==688 and all(json.loads(p.read_text())['validated_candidates']==4 for p in summaries)
else:status('DRO','paused_until_ab_report' if phase=='AB' else 'paused_until_ab_validation_complete')
status('pipeline','running',phase=phase)
errors={}
for name,fn in ([('AB',run_ab)] if phase=='AB' else [('DRO',run_dro)] if phase=='DRO' else [('AB',run_ab),('DRO',run_dro)]):
 try:fn()
 except Exception as e:
  errors[name]=repr(e);status(name,'failed',error=repr(e))
  break  # A/B failure must never start DRO.
 finally:
  command([py,str(R/'scripts/summarize_extension3_ab4x4_dro64.py'),'--run',str(RUN)],env,RUN/'logs/summary.log')
status('pipeline','failed' if errors else 'ab_complete_awaiting_report' if phase=='AB' else 'complete',errors=errors)
