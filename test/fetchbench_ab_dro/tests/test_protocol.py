import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_package import audit
from summarize_results import count_trials, finish, summarize
from run_workflow import plan


class PackageTests(unittest.TestCase):
    def test_source_and_generation_hashes(self):
        result = audit()
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['snapshot_files'], 53)
        self.assertEqual(result['generation_time_hashes_checked'], 12)

    def test_frozen_configs(self):
        import yaml
        for hand in ('barrett', 'shadow'):
            cfg = yaml.safe_load((ROOT/'configs'/('config_'+hand+'.yaml')).read_text())
            self.assertEqual(cfg['baseline']['contact_sets_per_object'], 4)
            self.assertEqual(cfg['baseline']['particles_per_contact_set'], 4)
            self.assertEqual(cfg['baseline']['optimization_steps'], 200)
            self.assertEqual(cfg['fk_optimization']['particles'], 4)
            self.assertEqual(cfg['fk_optimization']['steps'], 200)
            self.assertEqual(cfg['fk_optimization']['learning_rate'], .0075)
            self.assertEqual(cfg['fk_optimization']['palm_distance_weight'], 0)
            self.assertEqual(cfg['diffusion']['num_steps'], 50)

    def test_checkpoints_are_not_pure_input_ablation(self):
        m = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())['weights']
        self.assertNotEqual(m['checkpoint_a']['sha256'], m['checkpoint_b']['sha256'])

    def test_phase_budgets(self):
        manifest = json.loads((ROOT/'configs/historical_manifest.json').read_text())
        run, source = Path('/tmp/new_run'), Path('/tmp/source')
        ab = plan(manifest, run, source, 'ab-generate')[0]['command']
        self.assertEqual(ab[ab.index('--particles')+1], '4')
        self.assertEqual(ab[ab.index('--sets')+1], '4')
        self.assertEqual(ab[ab.index('--methods')+1], 'AB')
        dro = plan(manifest, run, source, 'dro')
        self.assertEqual(len(dro), 6)
        for item in dro[1::2]:
            env = item['environment']
            self.assertEqual(env['DRO_CANDIDATES'], '64')
            self.assertEqual(env['GPU_COUNT'], '2')
            self.assertEqual(env['VALIDATION_WORKERS'], '2')
        validation = plan(manifest, run, source, 'ab-validate')[0]['command']
        self.assertIn('_finite-worker', validation)

    def test_final_height_not_max_or_strict(self):
        self.assertEqual(count_trials([
            dict(final_object_lift_m=.10, max_object_lift_m=.30, success=False),
            dict(final_object_lift_m=.09, max_object_lift_m=.50, success=False),
        ]), (2, 1, 0))

    def test_nonfinite_simulation_result_is_an_error(self):
        with self.assertRaises(ValueError):
            count_trials([dict(final_object_lift_m=float('nan'), success=False)])

    def test_missing_not_failed_and_two_denominators(self):
        row = dict(planned=4, invalid=1, filtered=0, executed=2, height=1, strict=0)
        self.assertIsNone(finish(row)['planned_height_rate'])
        row['executed'] = 3
        result = finish(row)
        self.assertEqual(result['executed_height_rate'], 1/3)
        self.assertEqual(result['planned_height_rate'], 1/4)

    def test_inconsistent_budget_rejected(self):
        with self.assertRaises(ValueError):
            finish(dict(planned=4, invalid=1, filtered=0, executed=4, height=1, strict=0))

    def test_actual_coordinate_transform(self):
        import numpy as np
        path = ROOT/'snapshot/FetchBench-CORL2024/scripts/materialize_fetchbench_world_to_robot_base.py'
        spec = importlib.util.spec_from_file_location('coordinate_snapshot', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        points = np.array([[.07, -.167, .679], [float('nan'), 0, 1]])
        translation = np.array([-.5, .1, .4])
        rotation = module.quaternion_matrix_xyzw(np.array([0, 0, .70710678, .70710678]))
        base = module.transform_points(points, rotation.T, -rotation.T@translation)
        restored = module.transform_points(base, rotation, translation)
        np.testing.assert_allclose(restored[0], points[0], atol=1e-6)
        self.assertTrue(np.isnan(restored[1]).all())
        self.assertGreater(np.linalg.norm(base[0]-points[0]), .1)

    def test_summary_reads_finite_ab_not_obsolete_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            def save(path, data):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            case = dict(id='fixture', observations=str(run/'obs'), scene_factory='Factory', task_index=10)
            save(run/'manifest.json', dict(cases=[case]))
            save(run/'obs/visibility/legal_partial_views.json', dict(views=[dict(rgbd_capture_dir='/unused/view000')]))
            groups = [dict(case='fixture', condition=m, hand=h, planned=4, invalid=0, finite=4,
                           executed=4, height=1, strict=0, missing_finite=0)
                      for m in ('A', 'B') for h in ('barrett', 'shadowhand')]
            save(run/'ab_finite_validation/summary.json', dict(complete=True, errors=[], groups=groups))
            for hand in ('barrett', 'shadowhand'):
                folder = run/'generation/fixture/DRO_cpu_physx_all64/views/view000/simulation/Factory/task_010'/hand
                save(folder/'dro_candidates_environment_filtered.json', dict(
                    environment_filter=dict(generated_candidates=64, retained_candidates=2), records=[{}, {}]))
                for index, lift in enumerate((.10, .09)):
                    save(folder/('lift_validation/candidate_%d.json' % index), dict(final_object_lift_m=lift, max_object_lift_m=.4, success=False))
            result = summarize(run)
            self.assertTrue(result['complete'])
            self.assertEqual(len(result['rows']), 6)
            dro = next(r for r in result['by_method'] if r['method']=='DRO')
            self.assertEqual(dro['height'], 2)
            self.assertEqual(dro['filtered'], 124)
            self.assertEqual(dro['executed_height_rate'], .5)
            self.assertEqual(dro['planned_height_rate'], 2/128)


if __name__ == '__main__':
    unittest.main()
