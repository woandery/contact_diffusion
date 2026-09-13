
from .fetch.fetch_base import FetchBase
from .fetch.fetch_ptd import FetchPointCloudBase
from .fetch.fetch_naive import FetchNaive

isaacgym_task_map = {
    "FetchBase": FetchBase,
    "FetchPointCloudBase": FetchPointCloudBase,
    "FetchNaive": FetchNaive,
}

# Keep optional planners independent so a missing binary dependency does not
# prevent the base benchmark environment from loading.
try:
    from .fetch.fetch_mesh_curobo import FetchMeshCurobo
    from .fetch.repeat.fetch_mesh_curobo_rep import FetchMeshCuroboRep
    from .fetch.fetch_ptd_curobo import FetchPtdCurobo
    from .fetch.repeat.fetch_ptd_curobo_rep import FetchPtdCuroboRep
    from .fetch.fetch_mesh_curobo_datagen import FetchCuroboDataGen

    isaacgym_task_map.update({
        "FetchMeshCurobo": FetchMeshCurobo,
        "FetchMeshCuroboRep": FetchMeshCuroboRep,
        "FetchPtdCurobo": FetchPtdCurobo,
        "FetchPtdCuroboRep": FetchPtdCuroboRep,
        "FetchCuroboDataGen": FetchCuroboDataGen,
    })
except Exception as exc:
    print(f"CuRobo methods excluded ({type(exc).__name__}): {exc}")


try:
    from .fetch.imit.fetch_ptd_imit_e2e import FetchPtdImitE2E
    from .fetch.imit.fetch_ptd_imit_two_stage import FetchPtdImitTwoStage
    from .fetch.imit.fetch_ptd_imit_curobo_cgn import FetchPtdImitCuroboCGN

    isaacgym_task_map.update({
        "FetchPtdImitE2E": FetchPtdImitE2E,
        "FetchPtdImitTwoStage": FetchPtdImitTwoStage,
        "FetchPtdImitCuroboCGN": FetchPtdImitCuroboCGN,
    })
except Exception as exc:
    print(f"Imitation methods excluded ({type(exc).__name__}): {exc}")


try:
    from .fetch.fetch_mesh_pyompl import FetchMeshPyompl
    isaacgym_task_map["FetchMeshPyompl"] = FetchMeshPyompl
except Exception as exc:
    print(f"Mesh PyOMPL method excluded ({type(exc).__name__}): {exc}")


try:
    from .fetch.fetch_ptd_dro_pyompl import FetchPtdDROPyompl
    isaacgym_task_map.update({
        "FetchPtdDROBarrett": FetchPtdDROPyompl,
        "FetchPtdDROShadow": FetchPtdDROPyompl,
    })
except Exception as exc:
    print(f"D(R,O) dexterous methods excluded ({type(exc).__name__}): {exc}")


try:
    from .fetch.fetch_ptd_dro_render import FetchPtdDRORender
    isaacgym_task_map.update({
        "FetchPtdDRORenderBarrett": FetchPtdDRORender,
        "FetchPtdDRORenderShadow": FetchPtdDRORender,
    })
except Exception as exc:
    print(f"D(R,O) static render methods excluded ({type(exc).__name__}): {exc}")


try:
    # OMPL
    from .fetch.fetch_mesh_pyompl import FetchMeshPyompl
    from .fetch.repeat.fetch_mesh_pyompl_rep import FetchMeshPyomplRep
    from .fetch.fetch_ptd_pyompl import FetchPtdPyompl
    from .fetch.repeat.fetch_ptd_pyompl_rep import FetchPtdPyomplRep

    # Cabinet
    from .fetch.fetch_ptd_cabinet import FetchPtdCabinet
    from .fetch.fetch_ptd_cabinet_cgn_beta import FetchPtdCabinetCGNBeta

    # Contact_Graspnet_Pytorch
    from .fetch.fetch_ptd_curobo_cgn_beta import FetchPtdCuroboCGNBeta
    from .fetch.fetch_ptd_pyompl_cgn_beta import FetchPtdPyomplCGNBeta
    from .fetch.fetch_mesh_curobo_cgn_beta import FetchMeshCuroboPtdCGNBeta
    from .fetch.fetch_mesh_pyompl_cgn_beta import FetchMeshPyomplPtdCGNBeta

    from .fetch.repeat.fetch_mesh_curobo_cgn_beta_rep import FetchMeshCuroboPtdCGNBetaRep
    from .fetch.repeat.fetch_ptd_curobo_cgn_beta_rep import FetchPtdCuroboCGNBetaRep
    from .fetch.repeat.fetch_ptd_pyompl_cgn_beta_rep import FetchPtdPyomplCGNBetaRep

    isaacgym_task_map.update({
        "FetchMeshCurobo": FetchMeshCurobo,
        "FetchMeshPyompl": FetchMeshPyompl,
        "FetchMeshCuroboRep": FetchMeshCuroboRep,
        "FetchMeshPyomplRep": FetchMeshPyomplRep,

        "FetchPtdCurobo": FetchPtdCurobo,
        "FetchPtdPyompl": FetchPtdPyompl,

        "FetchCuroboDataGen": FetchCuroboDataGen,

        "FetchPtdCuroboRep": FetchPtdCuroboRep,
        "FetchPtdPyomplRep": FetchPtdPyomplRep,

        "FetchPtdCuroboCGNBeta": FetchPtdCuroboCGNBeta,
        "FetchPtdPyomplCGNBeta": FetchPtdPyomplCGNBeta,
        "FetchMeshCuroboPtdCGNBeta": FetchMeshCuroboPtdCGNBeta,
        "FetchMeshPyomplPtdCGNBeta": FetchMeshPyomplPtdCGNBeta,

        "FetchPtdCabinet": FetchPtdCabinet,
        "FetchPtdCabinetCGNBeta": FetchPtdCabinetCGNBeta,

        "FetchMeshCuroboPtdCGNBetaRep": FetchMeshCuroboPtdCGNBetaRep,
        "FetchPtdCuroboCGNBetaRep": FetchPtdCuroboCGNBetaRep,
        "FetchPtdPyomplCGNBetaRep": FetchPtdPyomplCGNBetaRep,

        "FetchPtdImitE2E": FetchPtdImitE2E,
        "FetchPtdImitTwoStage": FetchPtdImitTwoStage,
        "FetchPtdImitCuroboCGN": FetchPtdImitCuroboCGN,
    })

except Exception as exc:
    print("============================================================")
    print(f"Additional methods excluded: {exc}")
