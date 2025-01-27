# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from typing import Any, Dict
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import torch
import json
import safetensors.torch


def load_model(path: str, device: str | None, cls: Any):
    model_dir = Path(path)
    with (model_dir / "config.json").open() as f:
        loaded_config = json.load(f)
    if device is not None:
        loaded_config["device"] = device
    loaded_agent = cls(**loaded_config)
    safetensors.torch.load_model(loaded_agent, model_dir / "model.safetensors", device=device)
    # loaded_agent.load_state_dict(
    #     torch.load(model_dir / "model.pt", weights_only=True, map_location=device)
    # )
    return loaded_agent


def dict_to_config(source: Mapping, target: Any):
    target_fields = {field.name for field in dataclasses.fields(target)}
    for field in target_fields:
        if field in source.keys() and dataclasses.is_dataclass(getattr(target, field)):
            dict_to_config(source[field], getattr(target, field))
        elif field in source.keys():
            setattr(target, field, source[field])
        else:
            print(f"[WARNING] field {field} not found in source config")


def config_from_dict(source: Dict, config_class: Any):
    target = config_class()
    dict_to_config(source, target)
    return target


__version__ = "0.1.2"
