#!/usr/bin/env python3
"""Prepare standalone D(R,O) hand URDFs for FetchBench rendering.

The upstream extended URDFs expose six virtual XYZ/RPY joints followed by the
hand joints.  D(R,O)'s predicted q can therefore be assigned directly in
Isaac Gym without an arm or an IK solver.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


HAND_SOURCES = {
    "barrett": "data/data_urdf/robot/barrett/model_extended.urdf",
    "shadowhand": "data/data_urdf/robot/shadowhand/shadow_hand_right_extended.urdf",
}


def _rewrite_mesh_paths(robot: ET.Element, source_dir: Path) -> None:
    for mesh in robot.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename or filename.startswith("package://"):
            continue
        path = Path(filename)
        if not path.is_absolute():
            path = (source_dir / path).resolve()
        mesh.set("filename", str(path))


def _remove_collision_geometry(robot: ET.Element) -> None:
    """Make the hand visual-only so a rendered grasp cannot disturb objects."""
    for link in robot.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)


def _remove_link_visuals(robot: ET.Element, link_names: set[str]) -> None:
    """Hide bulky model bases while retaining their kinematic links."""
    for link in robot.findall("link"):
        if link.attrib.get("name") not in link_names:
            continue
        for visual in list(link.findall("visual")):
            link.remove(visual)


def _remove_link_collisions(robot: ET.Element, link_names: set[str]) -> None:
    """Remove collision geometry from hidden arm/base shells."""
    for link in robot.findall("link"):
        if link.attrib.get("name") not in link_names:
            continue
        for collision in list(link.findall("collision")):
            link.remove(collision)


def build(
    dro_root: Path,
    output_dir: Path,
    hand_name: str,
    keep_collisions: bool,
    keep_base_visuals: bool,
    source_override: Path | None = None,
    output_name: str | None = None,
) -> Path:
    source = (
        source_override.resolve()
        if source_override is not None
        else (dro_root / HAND_SOURCES[hand_name]).resolve()
    )
    if not source.is_file():
        raise FileNotFoundError(f"Missing D(R,O) hand URDF: {source}")

    robot = ET.parse(source).getroot()
    robot.set("name", f"dro_{hand_name}_standalone")
    _rewrite_mesh_paths(robot, source.parent)
    if not keep_collisions:
        _remove_collision_geometry(robot)
    if hand_name == "shadowhand" and not keep_base_visuals:
        # D(R,O)'s asset includes a large forearm shell. The standalone render
        # should show the dexterous hand rather than something resembling an arm.
        _remove_link_visuals(robot, {"forearm"})
        if keep_collisions:
            # The standalone validator must not let an invisible forearm shell
            # contact the object or furniture on behalf of the hand.
            _remove_link_collisions(robot, {"forearm"})

    output = output_dir / (output_name or f"dro_{hand_name}_standalone.urdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(output, encoding="utf-8", xml_declaration=True)
    return output


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dro-root", type=Path, default=workspace.parent / "DRO-Grasp")
    parser.add_argument("--hand", choices=("barrett", "shadowhand", "all"), default="all")
    parser.add_argument(
        "--keep-collisions",
        action="store_true",
        help="Keep hand collision meshes. The default is visual-only rendering.",
    )
    parser.add_argument(
        "--keep-base-visuals",
        action="store_true",
        help="Keep bulky base visuals such as the ShadowHand forearm shell.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "InfiniGym/assets/urdf/dexterous",
    )
    parser.add_argument(
        "--source-urdf",
        type=Path,
        help="Use an explicit extended URDF (only valid with one --hand).",
    )
    parser.add_argument(
        "--output-name",
        help="Output filename for an explicit single-hand build.",
    )
    args = parser.parse_args()

    if (args.source_urdf is not None or args.output_name is not None) and args.hand == "all":
        parser.error("--source-urdf/--output-name require a single --hand")

    hands = ("barrett", "shadowhand") if args.hand == "all" else (args.hand,)
    for hand_name in hands:
        print(build(
            args.dro_root.resolve(),
            args.output_dir.resolve(),
            hand_name,
            args.keep_collisions,
            args.keep_base_visuals,
            args.source_urdf,
            args.output_name,
        ))


if __name__ == "__main__":
    main()
