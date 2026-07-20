import numpy as np
import pytest

from utils.multi_contact_grasp_quality import MultiContactGraspEvaluator


class SphereSdf:
    def transform_pt_obj_to_grid(self, value):
        return np.asarray(value, dtype=np.float64)

    def transform_pt_grid_to_obj(self, value, direction=False):
        return np.asarray(value, dtype=np.float64)

    def is_out_of_bounds(self, point):
        return False

    def on_surface(self, point):
        distance = np.linalg.norm(point) - 1.0
        return abs(distance) <= 1e-7, distance

    def surface_normal(self, point):
        return np.asarray(point) / np.linalg.norm(point)


class SphereMesh:
    center_of_mass = np.zeros(3)

    def bounding_box(self):
        return -np.ones(3), np.ones(3)


class SphereObject:
    sdf = SphereSdf()
    mesh = SphereMesh()


OBJ = SphereObject()
ANTIPODAL = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
THREE_CLOSURE = np.array(
    [[1.0, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [-0.5, -np.sqrt(3) / 2, 0]]
)
THREE_NON_CLOSURE = np.eye(3)
FIVE = np.array(
    [[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0]]
)


def evaluate(points, **kwargs):
    defaults = dict(
        friction_coef=0.5,
        num_cone_faces=16,
        soft_fingers=True,
        finger_radius=0.2,
    )
    defaults.update(kwargs)
    return MultiContactGraspEvaluator.evaluate(OBJ, points, **defaults)


def sphere_projection(_obj, point):
    return point / np.linalg.norm(point)


def test_two_point_antipodal_has_positive_epsilon():
    result = evaluate(ANTIPODAL)
    assert result["valid"]
    assert result["force_closure"]
    assert result["epsilon"] > 0
    assert result["normal_convention"] == "outward"
    assert result["force_normal_convention"] == "inward"


def test_two_point_non_antipodal_has_zero_epsilon():
    result = evaluate([[1, 0, 0], [0, 1, 0]])
    assert result["valid"]
    assert not result["force_closure"]
    assert result["epsilon"] == 0


def test_three_point_closure_and_non_closure():
    closure = evaluate(THREE_CLOSURE, friction_coef=0.8)
    non_closure = evaluate(THREE_NON_CLOSURE, friction_coef=0.2)
    assert closure["force_closure"] and closure["epsilon"] > 0
    assert not non_closure["force_closure"] and non_closure["epsilon"] == 0


def test_five_contacts_and_both_input_layouts():
    rows = evaluate(FIVE)
    columns = evaluate(FIVE.T)
    assert rows["valid"] and columns["valid"]
    assert rows["num_contacts"] == 5
    assert rows["grasp_matrix"].shape == (6, 5 * (16 + 2))
    assert np.isclose(rows["epsilon"], columns["epsilon"])


def test_off_surface_is_invalid_unless_explicitly_projected():
    shifted = ANTIPODAL * 1.02
    invalid = evaluate(shifted)
    assert not invalid["valid"]
    assert "invalid contact" in invalid["failure_reason"]

    projected = evaluate(
        shifted,
        project_contacts=True,
        surface_projection_fn=sphere_projection,
    )
    assert projected["valid"]
    assert all(item["projected"] for item in projected["contact_validity"])


@pytest.mark.parametrize(
    "points,normals,reason",
    [
        (np.array([[np.nan, 0, 0], [-1, 0, 0]]), None, "NaN"),
        (ANTIPODAL, np.array([[2.0, 0, 0], [-1, 0, 0]]), "unit length"),
        (np.array([[1.0, 0, 0], [1.0, 0, 0]]), None, "duplicate"),
    ],
)
def test_invalid_nan_normal_and_duplicate(points, normals, reason):
    result = evaluate(points, contact_normals=normals)
    assert not result["valid"]
    assert reason in result["failure_reason"]


def test_inward_supplied_normals_are_flipped_to_outward():
    result = evaluate(ANTIPODAL, contact_normals=-ANTIPODAL)
    assert result["valid"]
    assert np.allclose(result["contact_normals"], ANTIPODAL)
    assert all(
        item["input_normal_orientation"] == "inward_flipped_to_outward"
        for item in result["contact_validity"]
    )


def test_mu_zero_and_low_rank_degeneracy_are_not_force_closure():
    zero_mu = evaluate(THREE_CLOSURE, friction_coef=0.0)
    hard_two = evaluate(ANTIPODAL, soft_fingers=False)
    assert zero_mu["epsilon"] == 0 and not zero_mu["force_closure"]
    assert hard_two["wrench_rank"] < 6
    assert hard_two["epsilon"] == 0 and not hard_two["force_closure"]


def test_soft_finger_adds_dexnet_torsional_wrenches():
    hard = evaluate(ANTIPODAL, soft_fingers=False)
    soft = evaluate(ANTIPODAL, soft_fingers=True)
    assert hard["grasp_matrix"].shape[1] == 2 * 16
    assert soft["grasp_matrix"].shape[1] == 2 * (16 + 2)
    assert not hard["force_closure"]
    assert soft["force_closure"]


def test_robust_results_are_reproducible_and_variance_is_not_std():
    kwargs = dict(
        contact_normals=None,
        friction_coef=0.8,
        num_cone_faces=12,
        soft_fingers=True,
        finger_radius=0.2,
        num_samples=20,
        point_sigma=0.01,
        friction_sigma=0.04,
        seed=23,
        project_contacts=True,
        surface_projection_fn=sphere_projection,
    )
    first = MultiContactGraspEvaluator.evaluate_robust(OBJ, THREE_CLOSURE, **kwargs)
    second = MultiContactGraspEvaluator.evaluate_robust(OBJ, THREE_CLOSURE, **kwargs)
    assert first == second
    assert first["num_valid_samples"] == 20
    assert np.isclose(first["epsilon_std"] ** 2, first["epsilon_variance"])


def test_robust_supplied_normals_still_project_perturbed_points():
    result = MultiContactGraspEvaluator.evaluate_robust(
        OBJ,
        THREE_CLOSURE,
        contact_normals=THREE_CLOSURE,
        friction_coef=0.8,
        num_cone_faces=12,
        soft_fingers=True,
        finger_radius=0.2,
        num_samples=8,
        point_sigma=0.01,
        normal_sigma=0.01,
        seed=9,
        project_contacts=True,
        surface_projection_fn=sphere_projection,
    )
    assert result["num_valid_samples"] == 8


def test_fixed_total_force_budget_is_independent_of_contact_count():
    target = np.zeros(6)
    two = evaluate(
        ANTIPODAL,
        target_wrench=target,
        force_limit=10.0,
        fixed_total_force_budget=True,
    )
    five = evaluate(
        FIVE,
        target_wrench=target,
        force_limit=10.0,
        fixed_total_force_budget=True,
    )
    assert two["force_budget"]["total_force_limit"] == 10.0
    assert five["force_budget"]["total_force_limit"] == 10.0
    assert two["force_budget"]["per_contact_force_limit"] == 5.0
    assert five["force_budget"]["per_contact_force_limit"] == 2.0


def test_target_wrench_partial_closure_and_resistance():
    result = evaluate(
        THREE_CLOSURE,
        friction_coef=0.8,
        target_wrench=np.array([0.05, 0, 0, 0, 0, 0]),
        force_limit=1.0,
        fixed_total_force_budget=True,
    )
    assert result["partial_closure"]
    assert result["wrench_resistance"] > 0
    assert result["target_wrench_residual"] <= 1e-5
