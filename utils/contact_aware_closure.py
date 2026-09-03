"""Pure helpers for contact-aware dexterous-hand closure."""

from __future__ import annotations

from collections.abc import Sequence


SHADOWHAND_FINGER_PREFIXES = ("FF", "MF", "RF", "LF", "TH")


def shadowhand_finger_groups(
    dof_names: Sequence[str], body_names: Sequence[str]
) -> list[dict[str, object]]:
    """Map ShadowHand digit prefixes to local DOF and rigid-body indices."""

    groups = []
    for prefix in SHADOWHAND_FINGER_PREFIXES:
        dof_indices = [
            index
            for index, name in enumerate(dof_names)
            if name.upper().startswith(f"{prefix}J")
        ]
        body_indices = [
            index
            for index, name in enumerate(body_names)
            if name.lower().startswith(prefix.lower())
        ]
        if not dof_indices or not body_indices:
            raise ValueError(
                f"Incomplete ShadowHand group {prefix}: "
                f"dofs={dof_indices}, bodies={body_indices}"
            )
        groups.append(
            {
                "name": prefix,
                "dof_indices": dof_indices,
                "body_indices": body_indices,
            }
        )
    return groups
