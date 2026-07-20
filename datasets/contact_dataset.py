"""Dataset loaders for contact diffusion.

Expected directory layout:

    dataset_root/
      train/n3/sample_00000000.npz
      val/n3/sample_00000000.npz
      test/n3/sample_00000000.npz

Required npz fields:
    object_pc: (2048, 3) float
    contacts: (n, 3) float
    num_contacts: scalar int
    selected_indices: (n,) int
    object_name: scalar string
    robot_name: scalar string
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import glob as globlib
from array import array
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


OBJECT_PC_ASSET_KEYS = (
    "object_pc_asset",
    "object_point_cloud_asset",
    "pc_asset",
    "asset_object_pc",
    "assets_object_pc",
    "asset_pcd",
    "vision_partial_pc_asset",
)
OBJECT_PC_SHARD_KEYS = (
    "object_pc",
    "object_points",
    "points",
    "point_cloud",
    "pc",
    "object_point_cloud",
)


class ContactDatasetV0(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        n: int,
        load_cmap: bool = True,
        load_qpos: bool = True,
        normalize: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.split = str(split)
        self.n = int(n)
        self.load_cmap = bool(load_cmap)
        self.load_qpos = bool(load_qpos)
        self.normalize = bool(normalize)
        self.sample_dir = self.root_dir / self.split / f"n{self.n}"
        if not self.sample_dir.exists():
            raise FileNotFoundError(f"Missing contact dataset directory: {self.sample_dir}")
        self.paths = sorted(self.sample_dir.glob("*.npz"))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .npz samples found in {self.sample_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def _fail(self, path: Path, message: str):
        raise ValueError(f"Invalid contact dataset sample {path}: {message}")

    def _check_finite(self, path: Path, name: str, array: np.ndarray):
        if not np.all(np.isfinite(array)):
            self._fail(path, f"{name} contains NaN or Inf")

    def _maybe_normalize(self, object_pc: np.ndarray, contacts: np.ndarray):
        if not self.normalize:
            return object_pc, contacts
        center = object_pc.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(object_pc - center, axis=1).max()
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        return (object_pc - center) / scale, (contacts - center) / scale

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with np.load(path, allow_pickle=False) as data:
            object_pc = np.asarray(data["object_pc"], dtype=np.float32)
            contacts = np.asarray(data["contacts"], dtype=np.float32)
            num_contacts = int(np.asarray(data["num_contacts"]).item())
            selected_indices = np.asarray(data["selected_indices"], dtype=np.int64)
            object_name = str(np.asarray(data["object_name"]).item())
            robot_name = str(np.asarray(data["robot_name"]).item())

            if object_pc.shape != (2048, 3):
                self._fail(path, f"object_pc shape must be (2048, 3), got {object_pc.shape}")
            if contacts.shape != (self.n, 3):
                self._fail(path, f"contacts shape must be ({self.n}, 3), got {contacts.shape}")
            if num_contacts != self.n:
                self._fail(path, f"num_contacts={num_contacts} does not match n={self.n}")
            if selected_indices.shape != (self.n,):
                self._fail(path, f"selected_indices shape must be ({self.n},), got {selected_indices.shape}")
            if np.any(selected_indices < 0) or np.any(selected_indices >= len(object_pc)):
                self._fail(path, "selected_indices contains out-of-range values")
            if not np.allclose(contacts, object_pc[selected_indices], atol=1e-6, rtol=1e-5):
                self._fail(path, "contacts are not equal to object_pc[selected_indices]")

            self._check_finite(path, "object_pc", object_pc)
            self._check_finite(path, "contacts", contacts)
            object_pc, contacts = self._maybe_normalize(object_pc, contacts)

            item = {
                "object_pc": torch.from_numpy(object_pc.astype(np.float32)),
                "contacts": torch.from_numpy(contacts.astype(np.float32)),
                "num_contacts": torch.tensor(num_contacts, dtype=torch.long),
                "robot_name": robot_name,
                "object_name": object_name,
                "selected_indices": torch.from_numpy(selected_indices),
                "path": str(path),
            }
            if self.load_cmap and "cmap" in data:
                cmap = np.asarray(data["cmap"], dtype=np.float32)
                if cmap.shape != (2048, 1):
                    self._fail(path, f"cmap shape must be (2048, 1), got {cmap.shape}")
                self._check_finite(path, "cmap", cmap)
                item["cmap"] = torch.from_numpy(cmap)
            if self.load_qpos and "qpos" in data:
                qpos = np.asarray(data["qpos"], dtype=np.float32)
                self._check_finite(path, "qpos", qpos)
                item["qpos"] = torch.from_numpy(qpos)
            if "contact_source" in data:
                item["contact_source"] = str(np.asarray(data["contact_source"]).item())
            if "n_definition" in data:
                item["n_definition"] = str(np.asarray(data["n_definition"]).item())
        return item


class ContactFormatDataset(Dataset):
    """Reader for the large `contact_format/v0` manifest + shard layout.

    Each sample is read lazily from one `manifest.jsonl` row and the matching
    `shards/shard_*.npz` entry. Object-side contact points are used by default
    because this project trains contacts in object point-cloud coordinates.
    """

    def __init__(
        self,
        root_dir: str,
        dataset_dir: str,
        split: str,
        n: int,
        num_points: int = 2048,
        contact_field: str = "contact_points",
        load_cmap: bool = False,
        load_qpos: bool = True,
        normalize: bool = False,
        split_fractions: Sequence[float] = (0.98, 0.01, 0.01),
        split_names: Sequence[str] = ("train", "val", "test"),
        max_samples: Optional[int] = None,
        seed: int = 42,
        index_cache_dir: str = ".cache/contactdiffusion/manifest_offsets",
        shard_cache_size: int = 4,
        object_pc_asset_keys: Optional[Sequence[str]] = None,
        success_only: bool = False,
        max_projection_distance: Optional[float] = None,
        allowed_grippers: Optional[Sequence[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.is_absolute():
            self.dataset_dir = self.root_dir / self.dataset_dir
        self.split = str(split)
        self.n = int(n)
        self.num_points = int(num_points)
        self.contact_field = str(contact_field)
        self.load_cmap = bool(load_cmap)
        self.load_qpos = bool(load_qpos)
        self.normalize = bool(normalize)
        self.seed = int(seed)
        self.shard_cache_size = int(shard_cache_size)
        self.object_pc_asset_keys = tuple(object_pc_asset_keys or OBJECT_PC_ASSET_KEYS)
        self.success_only = bool(success_only)
        self.max_projection_distance = None if max_projection_distance is None else float(max_projection_distance)
        self.allowed_grippers = None
        if allowed_grippers is not None:
            self.allowed_grippers = {str(name) for name in allowed_grippers}
        self._shard_cache: OrderedDict[Path, np.lib.npyio.NpzFile] = OrderedDict()
        self._asset_cache: OrderedDict[Path, np.ndarray] = OrderedDict()

        self.manifest_path = self.dataset_dir / "manifest.jsonl"
        self.shard_dir = self.dataset_dir / "shards"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing Contact Format manifest: {self.manifest_path}")
        if not self.shard_dir.exists():
            raise FileNotFoundError(f"Missing Contact Format shard directory: {self.shard_dir}")

        split_names = tuple(str(name) for name in split_names)
        if self.split not in split_names:
            raise ValueError(f"split must be one of {split_names}, got {self.split!r}")
        fractions = np.asarray(split_fractions, dtype=np.float64)
        if fractions.shape != (len(split_names),) or np.any(fractions < 0) or fractions.sum() <= 0:
            raise ValueError("split_fractions must be non-negative and match split_names")
        fractions = fractions / fractions.sum()

        self.offsets = self._load_offsets(
            split_names=split_names,
            split_fractions=fractions,
            max_samples=max_samples,
            index_cache_dir=Path(index_cache_dir) if index_cache_dir else None,
        )
        if len(self.offsets) == 0:
            raise FileNotFoundError(f"No Contact Format rows selected for split={self.split}")

    def __len__(self) -> int:
        return int(len(self.offsets))

    def __del__(self):
        for shard in getattr(self, "_shard_cache", {}).values():
            try:
                shard.close()
            except Exception:
                pass

    def _cache_key(self) -> str:
        stat = self.manifest_path.stat()
        filter_state = {
            "asset_keys": self.object_pc_asset_keys,
            "success_only": self.success_only,
            "max_projection_distance": self.max_projection_distance,
            "allowed_grippers": sorted(self.allowed_grippers) if self.allowed_grippers else None,
        }
        raw = f"{self.manifest_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{json.dumps(filter_state, sort_keys=True)}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _load_offsets(
        self,
        split_names: Sequence[str],
        split_fractions: np.ndarray,
        max_samples: Optional[int],
        index_cache_dir: Optional[Path],
    ) -> np.ndarray:
        key = self._cache_key()
        all_offsets = None
        if index_cache_dir is not None:
            index_cache_dir.mkdir(parents=True, exist_ok=True)
            all_path = index_cache_dir / f"{key}.all.npy"
            if all_path.exists():
                all_offsets = np.load(all_path, mmap_mode="r")
            else:
                built = self._build_manifest_offsets()
                tmp_path = all_path.with_name(f"{all_path.stem}.{os.getpid()}.tmp.npy")
                np.save(tmp_path, built)
                os.replace(tmp_path, all_path)
                all_offsets = np.load(all_path, mmap_mode="r")
        else:
            all_offsets = self._build_manifest_offsets()

        total = int(len(all_offsets))
        edges = np.rint(np.concatenate([[0.0], np.cumsum(split_fractions)]) * total).astype(np.int64)
        edges[-1] = total
        split_idx = split_names.index(self.split)
        offsets = all_offsets[edges[split_idx] : edges[split_idx + 1]]
        if max_samples is not None and int(max_samples) > 0 and len(offsets) > int(max_samples):
            rng = np.random.default_rng(self.seed + split_idx)
            keep = np.sort(rng.choice(len(offsets), size=int(max_samples), replace=False))
            offsets = np.asarray(offsets[keep], dtype=np.int64)
        return offsets

    def _row_matches_filters(self, row: dict) -> bool:
        if self.allowed_grippers is not None:
            gripper = row.get("gripper_name") or row.get("gripper") or row.get("robot_name") or row.get("hand")
            if str(gripper) not in self.allowed_grippers:
                return False
        quality = row.get("quality_flags") or {}
        if self.success_only and not bool(row.get("success", quality.get("success", False))):
            return False
        if self.max_projection_distance is not None:
            max_proj = row.get("max_projection_distance", quality.get("max_proj_m"))
            if max_proj is None or float(max_proj) > self.max_projection_distance:
                return False
        return True

    def _uses_row_filters(self) -> bool:
        return self.allowed_grippers is not None or self.success_only or self.max_projection_distance is not None

    def _build_manifest_offsets(self) -> np.ndarray:
        offsets = array("Q")
        offset = 0
        parse_rows = self._uses_row_filters()
        with open(self.manifest_path, "rb") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    if parse_rows:
                        row = json.loads(stripped.decode("utf-8"))
                        if self._row_matches_filters(row):
                            offsets.append(offset)
                    else:
                        offsets.append(offset)
                offset += len(line)
        return np.frombuffer(offsets, dtype=np.uint64).astype(np.int64, copy=False)

    def _read_row(self, offset: int) -> dict:
        with open(self.manifest_path, "rb") as f:
            f.seek(int(offset))
            return json.loads(f.readline().decode("utf-8"))

    def _shard_path(self, shard_id) -> Path:
        candidates = []
        if isinstance(shard_id, str):
            candidates.append(self.shard_dir / f"shard_{shard_id}.npz")
            try:
                shard_int = int(shard_id)
            except ValueError:
                shard_int = None
        else:
            shard_int = int(shard_id)
        if shard_int is not None:
            candidates.extend(
                [
                    self.shard_dir / f"shard_{shard_int:06d}.npz",
                    self.shard_dir / f"shard_{shard_int}.npz",
                ]
            )
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Cannot find shard for shard_id={shard_id!r} in {self.shard_dir}")

    def _load_shard(self, shard_id):
        path = self._shard_path(shard_id)
        shard = self._shard_cache.get(path)
        if shard is not None:
            self._shard_cache.move_to_end(path)
            return shard
        shard = np.load(path, allow_pickle=False)
        self._shard_cache[path] = shard
        while len(self._shard_cache) > self.shard_cache_size:
            _, old = self._shard_cache.popitem(last=False)
            old.close()
        return shard

    def _resolve_asset_path(self, rel_or_abs: str) -> Path:
        path = Path(rel_or_abs)
        return path if path.is_absolute() else self.root_dir / path

    def _load_array_asset(self, rel_or_abs: str) -> np.ndarray:
        path = self._resolve_asset_path(rel_or_abs)
        cached = self._asset_cache.get(path)
        if cached is not None:
            self._asset_cache.move_to_end(path)
            return cached
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                key = "points" if "points" in data else data.files[0]
                array = np.asarray(data[key])
        elif path.suffix == ".npy":
            array = np.load(path, allow_pickle=False)
        else:
            raise ValueError(f"Unsupported array asset type: {path}")
        array = np.asarray(array)
        self._asset_cache[path] = array
        while len(self._asset_cache) > 64:
            self._asset_cache.popitem(last=False)
        return array

    def _sample_rows(self, array: np.ndarray, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(f"Expected point array shape (M, >=3), got {array.shape}")
        total = int(array.shape[0])
        if total <= 0:
            raise ValueError("Cannot sample from an empty point array")
        replace = total < count
        indices = rng.choice(total, size=count, replace=replace)
        return np.asarray(array[indices, :3], dtype=np.float32), indices.astype(np.int64)

    def _select_contacts(self, contacts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        contacts = np.asarray(contacts, dtype=np.float32)
        if contacts.ndim != 2 or contacts.shape[1] < 3:
            raise ValueError(f"{self.contact_field} must have shape (N, >=3), got {contacts.shape}")
        contacts = contacts[:, :3]
        valid = np.all(np.isfinite(contacts), axis=1)
        contacts = contacts[valid]
        if len(contacts) == 0:
            raise ValueError(f"{self.contact_field} has no finite contact points")
        replace = len(contacts) < self.n
        indices = rng.choice(len(contacts), size=self.n, replace=replace)
        return np.asarray(contacts[indices], dtype=np.float32)

    def _nearest_indices(self, object_pc: np.ndarray, contacts: np.ndarray) -> np.ndarray:
        diff = contacts[:, None, :] - object_pc[None, :, :]
        return np.sum(diff * diff, axis=2).argmin(axis=1).astype(np.int64)

    def _maybe_normalize(self, object_pc: np.ndarray, contacts: np.ndarray):
        if not self.normalize:
            return object_pc, contacts
        center = object_pc.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(object_pc - center, axis=1).max()
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        return (object_pc - center) / scale, (contacts - center) / scale

    def __getitem__(self, idx: int):
        row = self._read_row(int(self.offsets[idx]))
        rng = np.random.default_rng(self.seed + int(idx))
        shard = self._load_shard(row["shard_id"])
        shard_offset = int(row["shard_offset"])
        if self.contact_field not in shard:
            raise KeyError(f"Shard is missing contact field {self.contact_field!r}")
        contacts_raw = np.asarray(shard[self.contact_field][shard_offset])
        contacts = self._select_contacts(contacts_raw, rng)

        pc_rel = next((row.get(key) for key in self.object_pc_asset_keys if row.get(key)), None)
        if pc_rel is None:
            pc_key = next((key for key in OBJECT_PC_SHARD_KEYS if key in shard), None)
            if pc_key is None:
                raise KeyError(
                    "No object point cloud found. "
                    f"Checked row asset keys={self.object_pc_asset_keys}, "
                    f"shard keys={list(shard.files)}, row keys={sorted(row.keys())}"
                )
            object_raw = np.asarray(shard[pc_key][shard_offset])
        else:
            object_raw = self._load_array_asset(pc_rel)
        object_pc, point_indices = self._sample_rows(object_raw, self.num_points, rng)

        item = {
            "object_pc": object_pc,
            "contacts": contacts,
            "num_contacts": np.array(self.n, dtype=np.int64),
            "selected_indices": self._nearest_indices(object_pc, contacts),
            "robot_name": str(
                row.get("gripper_name") or row.get("hand") or row.get("robot_name") or row.get("gripper") or self.dataset_dir.name
            ),
            "object_name": str(row.get("object_name") or row.get("object_id") or row.get("object_code") or ""),
            "path": f"{self.manifest_path}:{idx}",
        }

        if self.load_cmap:
            cu_rel = row.get("contact_union_asset") or row.get("cmap_asset")
            if cu_rel is not None:
                cmap_raw = np.asarray(self._load_array_asset(cu_rel), dtype=np.float32).reshape(-1)
                if len(cmap_raw) == len(object_raw):
                    item["cmap"] = cmap_raw[point_indices].reshape(self.num_points, 1)
                elif len(cmap_raw) == self.num_points:
                    item["cmap"] = cmap_raw.reshape(self.num_points, 1)
                else:
                    raise ValueError(
                        f"CMap length {len(cmap_raw)} does not match object PC length {len(object_raw)}"
                    )
        if self.load_qpos and "qpos" in shard:
            item["qpos"] = np.asarray(shard["qpos"][shard_offset], dtype=np.float32)

        item["object_pc"], item["contacts"] = self._maybe_normalize(item["object_pc"], item["contacts"])
        for key in ("object_pc", "contacts"):
            if not np.all(np.isfinite(item[key])):
                raise ValueError(f"{key} contains NaN or Inf at {item['path']}")

        out = {
            "object_pc": torch.from_numpy(np.asarray(item["object_pc"], dtype=np.float32)),
            "contacts": torch.from_numpy(np.asarray(item["contacts"], dtype=np.float32)),
            "num_contacts": torch.tensor(int(item["num_contacts"]), dtype=torch.long),
            "robot_name": item["robot_name"],
            "object_name": item["object_name"],
            "selected_indices": torch.from_numpy(np.asarray(item["selected_indices"], dtype=np.int64)),
            "path": item["path"],
        }
        if "cmap" in item:
            out["cmap"] = torch.from_numpy(np.asarray(item["cmap"], dtype=np.float32))
        if "qpos" in item:
            out["qpos"] = torch.from_numpy(np.asarray(item["qpos"], dtype=np.float32))
        return out


def contact_collate_fn(batch):
    if len(batch) == 0:
        return None
    out = {
        "object_pc": torch.stack([item["object_pc"] for item in batch], dim=0),
        "contacts": torch.stack([item["contacts"] for item in batch], dim=0),
        "num_contacts": torch.stack([item["num_contacts"] for item in batch], dim=0),
        "robot_name": [item["robot_name"] for item in batch],
        "object_name": [item["object_name"] for item in batch],
        "selected_indices": torch.stack([item["selected_indices"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
    }
    if all("cmap" in item for item in batch):
        out["cmap"] = torch.stack([item["cmap"] for item in batch], dim=0)
    if all("qpos" in item for item in batch):
        out["qpos"] = [item["qpos"] for item in batch]
    if all("contact_source" in item for item in batch):
        out["contact_source"] = [item["contact_source"] for item in batch]
    if all("n_definition" in item for item in batch):
        out["n_definition"] = [item["n_definition"] for item in batch]
    return out


def expand_contact_format_dataset_dirs(root_dir: str, dataset_dirs) -> list[str]:
    if dataset_dirs is None:
        return []
    if isinstance(dataset_dirs, (str, Path)):
        raw_dirs = [str(dataset_dirs)]
    else:
        raw_dirs = [str(item) for item in dataset_dirs]

    root = Path(root_dir)
    expanded = []
    for raw_dir in raw_dirs:
        path = Path(raw_dir)
        pattern = str(path if path.is_absolute() else root / path)
        matches = sorted(globlib.glob(pattern)) if any(ch in pattern for ch in "*?[") else [pattern]
        expanded.extend(matches)

    valid = []
    seen = set()
    for path_str in expanded:
        path = Path(path_str)
        if not (path / "manifest.jsonl").exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            valid.append(str(path.relative_to(root)))
        except ValueError:
            valid.append(str(path))
    return valid


def filter_contact_format_dataset_dirs(
    root_dir: str,
    dataset_dirs: Sequence[str],
    n: int,
    native_n_filter: bool = False,
    allowed_grippers: Optional[Sequence[str]] = None,
) -> list[str]:
    if not native_n_filter and allowed_grippers is None:
        return list(dataset_dirs)
    allowed = {str(name) for name in allowed_grippers} if allowed_grippers is not None else None
    root = Path(root_dir)
    filtered = []
    for one_dir in dataset_dirs:
        path = Path(one_dir)
        full = path if path.is_absolute() else root / path
        meta = {}
        meta_path = full / "dataset_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        gripper = str(meta.get("gripper") or meta.get("gripper_name") or full.name)
        if allowed is not None and gripper not in allowed:
            continue
        if native_n_filter:
            n_tips = meta.get("n_tips")
            if n_tips is None or int(n_tips) != int(n):
                continue
        filtered.append(one_dir)
    return filtered


def build_contact_format_dataset(
    root_dir: str,
    dataset_dir,
    split: str,
    n: int,
    num_points: int = 2048,
    contact_field: str = "contact_points",
    load_cmap: bool = False,
    load_qpos: bool = True,
    normalize: bool = False,
    split_fractions: Sequence[float] = (0.98, 0.01, 0.01),
    split_names: Sequence[str] = ("train", "val", "test"),
    max_samples: Optional[int] = None,
    seed: int = 42,
    index_cache_dir: str = ".cache/contactdiffusion/manifest_offsets",
    shard_cache_size: int = 4,
    object_pc_asset_keys: Optional[Sequence[str]] = None,
    success_only: bool = False,
    max_projection_distance: Optional[float] = None,
    allowed_grippers: Optional[Sequence[str]] = None,
    native_n_filter: bool = False,
) -> Dataset:
    dirs = expand_contact_format_dataset_dirs(root_dir, dataset_dir)
    dirs = filter_contact_format_dataset_dirs(
        root_dir=root_dir,
        dataset_dirs=dirs,
        n=int(n),
        native_n_filter=native_n_filter,
        allowed_grippers=allowed_grippers,
    )
    if not dirs:
        raise FileNotFoundError(
            f"No Contact Format dataset dirs matched: {dataset_dir} for n={int(n)} "
            f"with native_n_filter={native_n_filter} allowed_grippers={allowed_grippers}"
        )

    datasets = [
        ContactFormatDataset(
            root_dir=root_dir,
            dataset_dir=one_dir,
            split=split,
            n=int(n),
            num_points=num_points,
            contact_field=contact_field,
            load_cmap=load_cmap,
            load_qpos=load_qpos,
            normalize=normalize,
            split_fractions=split_fractions,
            split_names=split_names,
            max_samples=max_samples,
            seed=seed + i,
            index_cache_dir=index_cache_dir,
            shard_cache_size=shard_cache_size,
            object_pc_asset_keys=object_pc_asset_keys,
            success_only=success_only,
            max_projection_distance=max_projection_distance,
            allowed_grippers=allowed_grippers,
        )
        for i, one_dir in enumerate(dirs)
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_grouped_contact_loaders(
    root_dir: str,
    split: str,
    n_values: Iterable[int],
    batch_size: int,
    num_workers: int = 0,
    load_cmap: bool = True,
    load_qpos: bool = True,
    normalize: bool = False,
    dataset_type: str = "npz_v0",
    dataset_dir: Optional[str] = None,
    dataset_dirs=None,
    num_points: int = 2048,
    contact_field: str = "contact_points",
    split_fractions: Sequence[float] = (0.98, 0.01, 0.01),
    split_names: Sequence[str] = ("train", "val", "test"),
    max_samples: Optional[int] = None,
    split_max_samples: Optional[dict] = None,
    seed: int = 42,
    index_cache_dir: str = ".cache/contactdiffusion/manifest_offsets",
    shard_cache_size: int = 4,
    object_pc_asset_keys: Optional[Sequence[str]] = None,
    success_only: bool = False,
    max_projection_distance: Optional[float] = None,
    allowed_grippers: Optional[Sequence[str]] = None,
    native_n_filter: bool = False,
    distributed: bool = False,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> Dict[int, DataLoader]:
    loaders: Dict[int, DataLoader] = {}
    for n in n_values:
        if dataset_type in ("npz", "npz_v0", "v0"):
            dataset = ContactDatasetV0(
                root_dir=root_dir,
                split=split,
                n=int(n),
                load_cmap=load_cmap,
                load_qpos=load_qpos,
                normalize=normalize,
            )
        elif dataset_type in ("contact_format", "contact_format_v0"):
            format_dirs = dataset_dirs if dataset_dirs is not None else dataset_dir
            if format_dirs is None:
                raise ValueError("dataset_dir or dataset_dirs is required for dataset_type='contact_format'")
            per_split_max = max_samples
            if split_max_samples is not None and split in split_max_samples:
                per_split_max = split_max_samples[split]
            dataset = build_contact_format_dataset(
                root_dir=root_dir,
                dataset_dir=format_dirs,
                split=split,
                n=int(n),
                num_points=num_points,
                contact_field=contact_field,
                load_cmap=load_cmap,
                load_qpos=load_qpos,
                normalize=normalize,
                split_fractions=split_fractions,
                split_names=split_names,
                max_samples=per_split_max,
                seed=seed,
                index_cache_dir=index_cache_dir,
                shard_cache_size=shard_cache_size,
                object_pc_asset_keys=object_pc_asset_keys,
                success_only=success_only,
                max_projection_distance=max_projection_distance,
                allowed_grippers=allowed_grippers,
                native_n_filter=native_n_filter,
            )
        else:
            raise ValueError(f"Unknown dataset_type={dataset_type!r}")
        sampler = None
        loader_shuffle = shuffle
        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=int(distributed_world_size),
                rank=int(distributed_rank),
                shuffle=shuffle,
                drop_last=drop_last,
            )
            loader_shuffle = False
        loaders[int(n)] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=loader_shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=contact_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=drop_last,
        )
    return loaders


class GroupedContactDataLoaders:
    def __init__(self, loaders: Dict[int, DataLoader], n_sampling: str = "uniform"):
        if n_sampling != "uniform":
            raise NotImplementedError("Only uniform n_sampling is implemented.")
        self.loaders = {int(n): loader for n, loader in loaders.items()}
        self.n_values = sorted(self.loaders.keys())
        self.iterators = {n: iter(loader) for n, loader in self.loaders.items()}
        self.epochs = {n: 0 for n in self.loaders}

    def next(self, n: Optional[int] = None):
        if n is None:
            n = int(random.choice(self.n_values))
        n = int(n)
        try:
            batch = next(self.iterators[n])
        except StopIteration:
            self.epochs[n] += 1
            sampler = getattr(self.loaders[n], "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epochs[n])
            self.iterators[n] = iter(self.loaders[n])
            batch = next(self.iterators[n])
        return n, batch
