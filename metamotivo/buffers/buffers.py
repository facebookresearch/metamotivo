# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import functools
import numbers
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict, List, Union

import numpy as np
import torch

Device = Union[str, torch.device]


@functools.singledispatch
def _to_torch(value: Any, device: Device | None = None) -> Any:
    raise Exception(f"No known conversion for type ({type(value)}) to PyTorch registered. Report as issue on github.")


@_to_torch.register(numbers.Number)
@_to_torch.register(np.ndarray)
def _np_to_torch(value: np.ndarray, device: Device | None = None) -> torch.Tensor:
    tensor = torch.tensor(value)
    if device:
        return tensor.to(device=device)
    return tensor


@_to_torch.register(torch.Tensor)
def _torch_to_torch(value: np.ndarray, device: Device | None = None) -> torch.Tensor:
    tensor = value.clone().detach()
    if device:
        return tensor.to(device=device)
    return tensor


@dataclasses.dataclass
class DictBuffer:
    capacity: int
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.storage = None
        self._idx = 0
        self._is_full = False

    def __len__(self) -> int:
        return self.capacity if self._is_full else self._idx

    def empty(self) -> bool:
        return len(self) == 0

    @torch.no_grad
    def extend(self, data: Dict) -> None:
        if self.storage is None:
            self.storage = {}
            initialize_storage(data, self.storage, self.capacity, self.device)
            self._idx = 0
            self._is_full = False
            # let's store a key for easy inspection
            self._non_nested_key = [k for k, v in self.storage.items() if not isinstance(v, Mapping)][0]

        def add_new_data(data, storage, expected_dim: int):
            for k, v in data.items():
                if isinstance(v, Mapping):
                    # If the value is a dictionary, recursively call the function
                    add_new_data(v, storage=storage[k], expected_dim=expected_dim)
                else:
                    if v.shape[0] != expected_dim:
                        raise ValueError("We expect all keys to have the same dimension")
                    end = self._idx + v.shape[0]
                    if end >= self.capacity:
                        # Wrap data
                        diff = self.capacity - self._idx
                        storage[k][self._idx :] = _to_torch(v[:diff], device=self.device)
                        storage[k][: v.shape[0] - diff] = _to_torch(v[diff:], device=self.device)
                        self._is_full = True
                    else:
                        storage[k][self._idx : end] = _to_torch(v, device=self.device)

        data_dim = data[self._non_nested_key].shape[0]
        add_new_data(data, self.storage, expected_dim=data_dim)
        self._idx = (self._idx + data_dim) % self.capacity

    @torch.no_grad
    def sample(self, batch_size) -> Dict[str, torch.Tensor]:
        self.ind = torch.randint(0, len(self), (batch_size,))
        return extract_values(self.storage, self.ind)

    def get_full_buffer(self) -> Dict:
        if self._is_full:
            return self.storage
        else:
            return extract_values(self.storage, torch.arange(0, len(self)))


def extract_values(d: Dict, idxs: List | torch.Tensor | np.ndarray) -> Dict:
    result = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            result[k] = extract_values(v, idxs)
        else:
            result[k] = v[idxs]
    return result


@dataclasses.dataclass
class TrajectoryBuffer:
    capacity: int
    device: str = "cpu"
    seq_length: int = 1
    output_key_t: list[str] = dataclasses.field(default_factory=lambda: ["observation"])
    output_key_tp1: list[str] = dataclasses.field(default_factory=lambda: ["observation"])

    def __post_init__(self) -> None:
        self._is_full = False
        self.storage = None
        self._idx = 0
        self.priorities = None

    def __len__(self) -> int:
        return self.capacity if self._is_full else self._idx

    def empty(self) -> bool:
        return len(self) == 0

    @torch.no_grad
    def extend(self, data: List[dict]) -> None:
        if self.storage is None:
            self.storage = []
            for _ in range(self.capacity):
                element = {}
                initialize_storage(data[0], element, self.capacity, self.device)
                self.storage.append(element)
            self._idx = 0
            self._is_full = False
            self.priorities = torch.ones(self.capacity, device=self.device, dtype=torch.float32) / self.capacity

        def add(new_data):
            storage = {}
            for k, v in new_data.items():
                if isinstance(v, Mapping):
                    storage[k] = add(v)
                else:
                    storage[k] = _to_torch(v, device=self.device)
                    if len(storage[k].shape) == 1:
                        storage[k] = storage[k].reshape(-1, 1)
            return storage

        for episode in data:
            self.storage[self._idx] = add(new_data=episode)
            self._idx += 1
            if self._idx >= self.capacity:
                self._is_full = True
            self._idx = self._idx % self.capacity

    @torch.no_grad
    def sample(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        if batch_size < self.seq_length:
            raise ValueError(
                f"The batch-size must be bigger than the sequence length, got batch_size={batch_size} and seq_length={self.seq_length}."
            )

        if batch_size % self.seq_length != 0:
            raise ValueError(
                f"The batch-size must be divisible by the sequence length, got batch_size={batch_size} and seq_length={self.seq_length}."
            )
        num_slices = batch_size // self.seq_length

        # self.ep_ind = torch.randint(0, len(self), (num_slices,))
        self.ep_ind = torch.multinomial(self.priorities, num_slices, replacement=True)
        output = defaultdict(list)
        offset = 0
        if len(self.output_key_tp1) > 0:
            offset = 1
            output["next"] = defaultdict(list)
        for ep_idx in self.ep_ind:
            _ep = self.storage[ep_idx.item()]
            length = _ep[self.output_key_t[0]].shape[0]
            time_idx = torch.randint(0, length - self.seq_length - offset, (1,))
            for k in self.output_key_t:
                output[k].append(_ep[k][time_idx : time_idx + self.seq_length])
            for k in self.output_key_tp1:
                output["next"][k].append(_ep[k][time_idx + offset : time_idx + offset + self.seq_length])

        return dict_cat(output)

    def update_priorities(self, priorities: torch.Tensor, idxs: torch.Tensor) -> None:
        self.priorities[idxs] = priorities
        self.priorities = self.priorities / torch.sum(self.priorities)


def initialize_storage(data: Dict, storage: Dict, capacity: int, device: Device) -> None:
    def recursive_initialize(d, s):
        for k, v in d.items():
            if isinstance(v, Mapping):
                s[k] = {}
                recursive_initialize(v, s[k])
            else:
                s[k] = torch.zeros(
                    (capacity, v.shape[1] if len(v.shape) == 2 else v.shape[0]),
                    device=device,
                    dtype=dtype_numpytotorch(v.dtype),
                )

    recursive_initialize(data, storage)


def dtype_numpytotorch(np_dtype: Any) -> torch.dtype:
    if isinstance(np_dtype, torch.dtype):
        return np_dtype
    if np_dtype == np.float16:
        return torch.float16
    elif np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float64:
        return torch.float64
    elif np_dtype == np.int16:
        return torch.int16
    elif np_dtype == np.int32:
        return torch.int32
    elif np_dtype == np.int64:
        return torch.int64
    elif np_dtype == bool:
        return torch.bool
    elif np_dtype == np.uint8:
        return torch.uint8
    else:
        raise ValueError(f"Unknown type {np_dtype}")


def dict_cat(d: Mapping) -> Dict[str, torch.Tensor]:
    res = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            res[k] = dict_cat(v)
        else:
            res[k] = torch.cat(v, dim=0)
    return res
