"""Standalone `.npz` dataset loader for contact diffusion.

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

import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


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


def build_grouped_contact_loaders(
    root_dir: str,
    split: str,
    n_values: Iterable[int],
    batch_size: int,
    num_workers: int = 0,
    load_cmap: bool = True,
    load_qpos: bool = True,
    normalize: bool = False,
    shuffle: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> Dict[int, DataLoader]:
    loaders: Dict[int, DataLoader] = {}
    for n in n_values:
        dataset = ContactDatasetV0(
            root_dir=root_dir,
            split=split,
            n=int(n),
            load_cmap=load_cmap,
            load_qpos=load_qpos,
            normalize=normalize,
        )
        loaders[int(n)] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
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

    def next(self, n: Optional[int] = None):
        if n is None:
            n = int(random.choice(self.n_values))
        n = int(n)
        try:
            batch = next(self.iterators[n])
        except StopIteration:
            self.iterators[n] = iter(self.loaders[n])
            batch = next(self.iterators[n])
        return n, batch
