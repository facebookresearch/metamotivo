# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch
from typing import Any
import numpy as np
import mujoco
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import dataclasses
from humenv import make_humenv
from humenv.rewards import RewardFunction


def get_next(field: str, data: Any):
    if "next" in data and field in data["next"]:
        return data["next"][field]
    elif f"next_{field}" in data:
        return data[f"next_{field}"]
    else:
        raise ValueError(f"No next of {field} found in data.")


@dataclasses.dataclass(kw_only=True)
class BaseHumEnvBenchWrapper:
    model: Any
    numpy_output: bool = True
    _dtype: torch.dtype = dataclasses.field(default_factory=lambda: torch.float32)

    def act(
        self,
        obs: torch.Tensor | np.ndarray,
        z: torch.Tensor | np.ndarray,
        mean: bool = True,
    ) -> torch.Tensor:
        obs = to_torch(obs, device=self.device, dtype=self._dtype)
        z = to_torch(z, device=self.device, dtype=self._dtype)
        if self.numpy_output:
            return self.unwrapped_model.act(obs, z, mean).cpu().detach().numpy()
        return self.unwrapped_model.act(obs, z, mean)

    @property
    def device(self) -> Any:
        # this returns the base torch.nn.module
        return self.unwrapped_model.cfg.device

    @property
    def unwrapped_model(self):
        # this is used to call the base instance of model
        if hasattr(self.model, "unwrapped_model"):
            return self.model.unwrapped_model
        else:
            return self.model

    def __getattr__(self, name):
        # Delegate to the wrapped instance
        return getattr(self.model, name)

    def __deepcopy__(self, memo):
        return type(self)(model=copy.deepcopy(self.model, memo), numpy_output=self.numpy_output, _dtype=copy.deepcopy(self._dtype))

    def __getstate__(self):
        # Return a dictionary containing the state of the object
        return {
            "model": self.model,
            "numpy_output": self.numpy_output,
            "_dtype": self._dtype,
        }

    def __setstate__(self, state):
        # Restore the state of the object from the given dictionary
        self.model = state["model"]
        self.numpy_output = state["numpy_output"]
        self._dtype = state["_dtype"]


@dataclasses.dataclass(kw_only=True)
class RewardWrapper(BaseHumEnvBenchWrapper):
    inference_dataset: Any
    num_samples_per_inference: int
    inference_function: str
    max_workers: int
    process_executor: bool = False
    process_context: str = "spawn"

    def reward_inference(self, task: str, **kwargs) -> torch.Tensor:
        env, _ = make_humenv(task=task, **kwargs)
        if self.num_samples_per_inference < len(self.inference_dataset):
            data = self.inference_dataset.sample(self.num_samples_per_inference)
        else:
            data = self.inference_dataset.get_full_buffer()
        qpos = get_next("qpos", data)
        qvel = get_next("qvel", data)
        action = data["action"]
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().detach().numpy()
            qvel = qvel.cpu().detach().numpy()
            action = action.cpu().detach().numpy()
        rewards = relabel(
            env,
            qpos,
            qvel,
            action,
            env.unwrapped.task,
            max_workers=self.max_workers,
            process_executor=self.process_executor,
        )
        env.close()

        td = {
            "reward": torch.tensor(rewards, dtype=torch.float32, device=self.device),
        }
        if "B" in data:
            td["B_vect"] = data["B"]
        else:
            td["next_obs"] = get_next("observation", data)
        inference_fn = getattr(self.model, self.inference_function, None)
        ctxs = inference_fn(**td).reshape(1, -1)
        return ctxs

    def __deepcopy__(self, memo):
        # Create a new instance of the same type as self
        return type(self)(
            model=copy.deepcopy(self.model, memo),
            numpy_output=self.numpy_output,
            _dtype=copy.deepcopy(self._dtype),
            inference_dataset=copy.deepcopy(self.inference_dataset),
            num_samples_per_inference=self.num_samples_per_inference,
            inference_function=self.inference_function,
            max_workers=self.max_workers,
            process_executor=self.process_executor,
            process_context=self.process_context,
        )

    def __getstate__(self):
        # Return a dictionary containing the state of the object
        return {
            "model": self.model,
            "numpy_output": self.numpy_output,
            "_dtype": self._dtype,
            "inference_dataset": self.inference_dataset,
            "num_samples_per_inference": self.num_samples_per_inference,
            "inference_function": self.inference_function,
            "max_workers": self.max_workers,
            "process_executor": self.process_executor,
            "process_context": self.process_context,
        }

    def __setstate__(self, state):
        # Restore the state of the object from the given dictionary
        self.model = state["model"]
        self.numpy_output = state["numpy_output"]
        self._dtype = state["_dtype"]
        self.inference_dataset = state["inference_dataset"]
        self.num_samples_per_inference = state["num_samples_per_inference"]
        self.inference_function = state["inference_function"]
        self.max_workers = state["max_workers"]
        self.process_executor = state["process_executor"]
        self.process_context = state["process_context"]


@dataclasses.dataclass(kw_only=True)
class GoalWrapper(BaseHumEnvBenchWrapper):
    def goal_inference(self, goal_pose: torch.Tensor) -> torch.Tensor:
        next_obs = to_torch(goal_pose, device=self.device, dtype=self._dtype)
        ctx = self.unwrapped_model.goal_inference(next_obs=next_obs).reshape(1, -1)
        return ctx


@dataclasses.dataclass(kw_only=True)
class TrackingWrapper(BaseHumEnvBenchWrapper):
    def tracking_inference(self, next_obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        next_obs = to_torch(next_obs, device=self.device, dtype=self._dtype)
        ctx = self.unwrapped_model.tracking_inference(next_obs=next_obs)
        return ctx


def to_torch(x: np.ndarray | torch.Tensor, device: torch.device | str, dtype: torch.dtype):
    if len(x.shape) == 1:
        # adding batch dimension
        x = x[None, ...]
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=device, dtype=dtype)
    else:
        x = x.to(dtype)
    return x


def _relabel_worker(
    x,
    model: mujoco.MjModel,
    reward_fn: RewardFunction,
):
    qpos, qvel, action = x
    assert len(qpos.shape) > 1
    assert qvel.shape[0] == qpos.shape[0]
    assert qvel.shape[0] == action.shape[0]
    rewards = np.zeros((qpos.shape[0], 1))
    for i in range(qpos.shape[0]):
        rewards[i] = reward_fn(model, qpos[i], qvel[i], action[i])
    return rewards


def relabel(
    env: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    reward_fn: RewardFunction,
    max_workers: int = 5,
    process_executor: bool = False,
    process_context: str = "spawn",
):
    chunk_size = int(np.ceil(qpos.shape[0] / max_workers))
    args = [(qpos[i : i + chunk_size], qvel[i : i + chunk_size], action[i : i + chunk_size]) for i in range(0, qpos.shape[0], chunk_size)]
    if max_workers == 1:
        result = [_relabel_worker(args[0], model=env.unwrapped.model, reward_fn=reward_fn)]
    else:
        if process_executor:
            import multiprocessing

            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=multiprocessing.get_context(process_context),
            ) as exe:
                f = functools.partial(_relabel_worker, model=env.unwrapped.model, reward_fn=reward_fn)
                result = exe.map(f, args)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                f = functools.partial(_relabel_worker, model=env.unwrapped.model, reward_fn=reward_fn)
                result = exe.map(f, args)

    tmp = [r for r in result]
    return np.concatenate(tmp)
