from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_SCENE = PACKAGE_ROOT / "sim/assets/environment/base_scene.xml"
TEMPLATE_DIR = PACKAGE_ROOT / "sim/assets/templates"
OBJECT_DIR = PACKAGE_ROOT / "sim/assets/objects"

HAND_TEMPLATES = {
    "shadow_hand_right": TEMPLATE_DIR / "shadow_hand_right.xml",
    "gaia_hand_right": TEMPLATE_DIR / "gaia_hand_right.xml",
    "sharpa_hand_right": TEMPLATE_DIR / "sharpa_hand_right.xml",
}

# Policy wrist RPY uses the Shadow convention. Some target hands need a
# physical wrist-frame correction to preserve the same palm convention.
# A wrist-mounted camera needs the inverse mount correction so that its world
# extrinsics remain equal to the Shadow training camera for the same policy pose.
POLICY_WRIST_EULER_OFFSETS = {
    "shadow_hand_right": (0.0, 0.0, 0.0),
    "gaia_hand_right": (0.0, 0.0, 0.0),
    "sharpa_hand_right": (0.0, 0.0, math.pi / 2.0),
}

_BASE_SECTIONS = ("option", "statistic", "visual", "default")
_ENVIRONMENT_GEOMS = {"floor", "table"}


def list_object_ids() -> tuple[str, ...]:
    if not OBJECT_DIR.is_dir():
        return ()
    return tuple(
        path.name
        for path in sorted(OBJECT_DIR.iterdir())
        if path.is_dir()
        and (path / "visual/simplified.obj").is_file()
        and any((path / "collision").glob("convex_piece_*.obj"))
    )


def resolve_object_assets(object_id: str) -> tuple[Path, tuple[Path, ...]]:
    if Path(object_id).name != object_id:
        raise ValueError(f"Invalid object ID: {object_id!r}")
    object_dir = OBJECT_DIR / object_id
    visual_mesh = object_dir / "visual/simplified.obj"
    collision_meshes = tuple(
        sorted((object_dir / "collision").glob("convex_piece_*.obj"))
    )
    if not visual_mesh.is_file() or not collision_meshes:
        raise KeyError(
            f"Unknown or incomplete object {object_id!r}; "
            f"available={list_object_ids()}"
        )
    return visual_mesh, collision_meshes


def _replace_section(target: ET.Element, source: ET.Element, tag: str) -> None:
    source_node = source.find(tag)
    if source_node is None:
        return
    target_node = target.find(tag)
    if target_node is not None:
        index = list(target).index(target_node)
        target.remove(target_node)
        target.insert(index, deepcopy(source_node))
        return

    order = {
        "compiler": 0,
        "option": 1,
        "size": 2,
        "visual": 3,
        "statistic": 4,
        "default": 5,
        "asset": 6,
        "worldbody": 7,
        "contact": 8,
        "equality": 9,
        "actuator": 10,
        "sensor": 11,
        "keyframe": 12,
    }
    source_order = order.get(tag, 100)
    index = len(target)
    for i, child in enumerate(target):
        if order.get(child.tag, 100) > source_order:
            index = i
            break
    target.insert(index, deepcopy(source_node))


def _merge_assets(target: ET.Element, source: ET.Element) -> None:
    source_asset = source.find("asset")
    if source_asset is None:
        return
    target_asset = target.find("asset")
    if target_asset is None:
        target_asset = ET.Element("asset")
        target.insert(0, target_asset)

    for source_node in source_asset:
        name = source_node.attrib.get("name")
        if name:
            for existing in list(target_asset):
                if existing.tag == source_node.tag and existing.attrib.get("name") == name:
                    target_asset.remove(existing)
        target_asset.insert(0, deepcopy(source_node))


def _replace_environment(target: ET.Element, source: ET.Element) -> None:
    target_world = target.find("worldbody")
    source_world = source.find("worldbody")
    if target_world is None or source_world is None:
        raise ValueError("Both MJCF files must define <worldbody>.")

    for parent in target_world.iter():
        for node in list(parent):
            if node.tag == "camera":
                parent.remove(node)

    for node in list(target_world):
        name = node.attrib.get("name")
        if node.tag == "light":
            target_world.remove(node)
        elif node.tag == "geom" and (name in _ENVIRONMENT_GEOMS or name is None):
            target_world.remove(node)

    insert_index = 0
    for source_node in source_world:
        name = source_node.attrib.get("name")
        is_environment = source_node.tag in {"light", "camera"}
        is_environment_geom = source_node.tag == "geom" and name in _ENVIRONMENT_GEOMS
        if is_environment or is_environment_geom:
            target_world.insert(insert_index, deepcopy(source_node))
            insert_index += 1


def _attach_hand_mount_cameras(
    target: ET.Element,
    source: ET.Element,
    *,
    target_body_name: str = "wrist_rz_link",
    camera_mount_yaw: float = 0.0,
) -> None:
    """Attach camera templates under base ``hand_mount`` to the moving wrist."""
    source_mount = next(
        (
            body
            for body in source.iter("body")
            if body.attrib.get("name") == "hand_mount"
        ),
        None,
    )
    if source_mount is None:
        return
    camera_templates = list(source_mount.iter("camera"))
    if not camera_templates:
        return

    target_body = next(
        (
            body
            for body in target.iter("body")
            if body.attrib.get("name") == target_body_name
        ),
        None,
    )
    if target_body is None:
        raise ValueError(
            f"Hand scene is missing wrist camera mount body {target_body_name!r}."
        )

    camera_names = {camera.attrib.get("name") for camera in camera_templates}
    for parent in target.iter():
        for node in list(parent):
            if node.tag == "camera" and node.attrib.get("name") in camera_names:
                parent.remove(node)
    cosine = math.cos(camera_mount_yaw)
    sine = math.sin(camera_mount_yaw)

    def rotate_z(values: Sequence[float]) -> tuple[float, float, float]:
        x, y, z = values
        return (cosine * x - sine * y, sine * x + cosine * y, z)

    for camera in camera_templates:
        mounted_camera = deepcopy(camera)
        if camera_mount_yaw:
            position = tuple(
                float(value)
                for value in mounted_camera.get("pos", "0 0 0").split()
            )
            if len(position) != 3:
                raise ValueError(f"Camera pos must contain 3 values: {position}")
            mounted_camera.set(
                "pos",
                " ".join(f"{value:.12g}" for value in rotate_z(position)),
            )

            xyaxes_text = mounted_camera.get("xyaxes")
            if xyaxes_text is None:
                raise ValueError(
                    "Cross-hand wrist camera compensation requires camera xyaxes."
                )
            xyaxes = tuple(float(value) for value in xyaxes_text.split())
            if len(xyaxes) != 6:
                raise ValueError(f"Camera xyaxes must contain 6 values: {xyaxes}")
            rotated_axes = rotate_z(xyaxes[:3]) + rotate_z(xyaxes[3:])
            mounted_camera.set(
                "xyaxes",
                " ".join(f"{value:.12g}" for value in rotated_axes),
            )
        target_body.append(mounted_camera)


def _absolutize_files(root: ET.Element, source_dir: Path) -> None:
    """Keep assets valid when the composed XML is written to a temp directory."""
    for node in root.iter():
        file_value = node.attrib.get("file")
        if not file_value:
            continue
        path = Path(file_value)
        if not path.is_absolute():
            node.set("file", str((source_dir / path).resolve()))


def _replace_object(
    root: ET.Element,
    mesh_paths: Sequence[str | Path],
    scale: float | Sequence[float],
    position: Sequence[float],
    quaternion: Sequence[float],
    *,
    visual_mesh: str | Path | None = None,
    collision_meshes: Sequence[str | Path] = (),
) -> None:
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError("Object position and quaternion must have lengths 3 and 4.")
    scale_values = (scale, scale, scale) if isinstance(scale, (int, float)) else tuple(scale)
    if len(scale_values) != 3:
        raise ValueError("Object scale must be a scalar or xyz triplet.")

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("The hand template must define <asset> and <worldbody>.")

    old_object = next(
        (node for node in list(worldbody) if node.tag == "body" and node.get("name") == "object"),
        None,
    )
    if old_object is not None:
        worldbody.remove(old_object)

    object_body = ET.SubElement(
        worldbody,
        "body",
        name="object",
        pos=" ".join(map(str, position)),
        quat=" ".join(map(str, quaternion)),
    )
    ET.SubElement(object_body, "freejoint", name="object_joint")
    ET.SubElement(
        object_body,
        "inertial",
        pos="0 0 0",
        mass="0.10",
        diaginertia="0.0001 0.0001 0.0001",
    )
    scale_text = " ".join(str(float(value)) for value in scale_values)

    if visual_mesh is not None:
        visual_name = "evaluation_object_visual"
        ET.SubElement(
            asset,
            "mesh",
            name=visual_name,
            file=str(Path(visual_mesh).resolve()),
            scale=scale_text,
        )
        ET.SubElement(
            object_body,
            "geom",
            name=visual_name,
            type="mesh",
            mesh=visual_name,
            density="0",
            contype="0",
            conaffinity="0",
            group="2",
            rgba="0.48 0.63 0.74 1",
        )

    if collision_meshes:
        for index, mesh_path in enumerate(collision_meshes):
            mesh_name = f"evaluation_object_collision_{index:03d}"
            ET.SubElement(
                asset,
                "mesh",
                name=mesh_name,
                file=str(Path(mesh_path).resolve()),
                scale=scale_text,
            )
            ET.SubElement(
                object_body,
                "geom",
                name=mesh_name,
                type="mesh",
                mesh=mesh_name,
                density="0",
                group="3",
                rgba="0.48 0.63 0.74 0",
            )
        return

    for index, mesh_path in enumerate(mesh_paths):
        mesh_name = f"evaluation_object_piece_{index:03d}"
        ET.SubElement(
            asset,
            "mesh",
            name=mesh_name,
            file=str(Path(mesh_path).resolve()),
            scale=scale_text,
        )
        ET.SubElement(
            object_body,
            "geom",
            name=mesh_name,
            type="mesh",
            mesh=mesh_name,
            density="0",
            rgba="0.48 0.63 0.74 1",
        )


def compose_grasp_scene(
    *,
    hand: str,
    output_scene: str | Path,
    generated_scene: str | Path | None = None,
    base_scene: str | Path = DEFAULT_BASE_SCENE,
    object_id: str | None = None,
    object_meshes: Iterable[str | Path] | None = None,
    object_scale: float | Sequence[float] = 1.0,
    object_position: Sequence[float] = (0.0, 0.0, 0.035),
    object_quaternion: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
) -> Path:
    """Compose one hand/object scene with the shared grasp environment."""
    if hand not in HAND_TEMPLATES:
        raise KeyError(f"Unknown hand {hand!r}; available={tuple(HAND_TEMPLATES)}")
    generated_path = Path(generated_scene or HAND_TEMPLATES[hand]).resolve()
    base_path = Path(base_scene).resolve()
    output_path = Path(output_scene).resolve()

    target_tree = ET.parse(generated_path)
    target_root = target_tree.getroot()
    base_root = ET.parse(base_path).getroot()
    if target_root.tag != "mujoco" or base_root.tag != "mujoco":
        raise ValueError("Expected <mujoco> roots for generated and base scenes.")

    _absolutize_files(target_root, generated_path.parent)
    meshes = tuple(object_meshes or ())
    if object_id is not None and meshes:
        raise ValueError("Pass either object_id or object_meshes, not both.")
    if object_id is not None:
        visual_mesh, collision_meshes = resolve_object_assets(object_id)
        _replace_object(
            target_root,
            (),
            object_scale,
            object_position,
            object_quaternion,
            visual_mesh=visual_mesh,
            collision_meshes=collision_meshes,
        )
    elif meshes:
        _replace_object(
            target_root,
            meshes,
            object_scale,
            object_position,
            object_quaternion,
        )

    for tag in _BASE_SECTIONS:
        _replace_section(target_root, base_root, tag)
    _merge_assets(target_root, base_root)
    _replace_environment(target_root, base_root)
    _attach_hand_mount_cameras(
        target_root,
        base_root,
        camera_mount_yaw=POLICY_WRIST_EULER_OFFSETS[hand][2],
    )

    target_root.set("model", f"{target_root.get('model', hand)}_grasp_scene")
    ET.indent(target_tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_tree.write(output_path, encoding="unicode")
    return output_path
