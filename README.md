# Meta Motivo
**[Meta, FAIR](https://ai.facebook.com/research/)**


# Overview
This repository provides a PyTorch implementation and pre-trained models for Meta Motivo. For details see the paper [Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models](https://metamotivo.metademolab.com/).

### Features

- We provide [**6** pretrained FB-CPR](https://huggingface.co/collections/facebook/meta-motivo-6757761e8fd4a032466fd129) models for controlling the humanoid model defined in [HumEnv](https://github.com/facebookresearch/HumEnv/).
- **Fully reproducible** scripts for evaluating the model in HumEnv.
- **Fully reproducible** [FB-CPR training code in HumEnv](examples/fbcpr_train_humenv.py) for the full results in the paper, and [FB training code in DMC](examples/fb_train_dmc.py) for faster experimentation.

#  Installation

The project is pip installable in your environment.

```
pip install "metamotivo[huggingface,humenv] @ git+https://github.com/facebookresearch/metamotivo.git"
```

It requires Python 3.10+. Optional dependencies include `humenv["bench"]` and `huggingface_hub` for testing/training and loading models from HuggingFace.


# Pretrained models

For reproducibility, we provide all the **5** models (**metamotivo-S-X**) we trained for producing the results in the [paper](https://openreview.net/forum?id=9sOR0nYLtz&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions)), where each model is trained using a different random seed. We also provide our largest and most performant model (**metamotivo-M-1**), which can also be interactively tested in our [demo](https://metamotivo.metademolab.com/).

| Model | # of params | Download |
| :---         |     :---:      |          :---: |
| metamotivo-S-1     |   24.5M     | [link](https://huggingface.co/facebook/metamotivo-S-1)      |
| metamotivo-S-2     |   24.5M     | [link](https://huggingface.co/facebook/metamotivo-S-2)      |
| metamotivo-S-3     |   24.5M     | [link](https://huggingface.co/facebook/metamotivo-S-3)      |
| metamotivo-S-4     |   24.5M     | [link](https://huggingface.co/facebook/metamotivo-S-4)      |
| metamotivo-S-5     |   24.5M     | [link](https://huggingface.co/facebook/metamotivo-S-5)      |
| metamotivo-M-1     |   288M      | [link](https://huggingface.co/facebook/metamotivo-M-1)      |


# Quick start

Once the library is installed, you can easily create an FB-CPR agent and download a pre-trained model from the Hugging Face hub. Note that the model is an instance of `torch.nn.Module` and by default it is initialized in "inference" mode (no_grad and eval mode).

We provide some simple code snippets to demonstrate how to use the model below. For more detailed examples, see our tutorials on [interacting with the model](https://github.com/facebookresearch/metamotivo/blob/main/tutorial.ipynb), [running an evaluation](https://github.com/facebookresearch/metamotivo/blob/main/tutorial_benchmark.ipynb), and [training from scratch](https://github.com/facebookresearch/metamotivo/tree/main/examples).

### Download the pre-trained models

The following code snippet shows how to instantiate the model. 

```python
from metamotivo.fb_cpr.huggingface import FBcprModel

model = FBcprModel.from_pretrained("facebook/metamotivo-S-1")
```

### Download the buffers
For each model we provide:
- The training buffer (that can be used for inference or offline training)
- A small reward inference buffer (that contains the minimum amount of information for doing reward inference)

```python
from huggingface_hub import hf_hub_download
import h5py

local_dir = "metamotivo-S-1-datasets"
dataset = "buffer_inference_500000.hdf5"  # a smaller buffer that can be used for reward inference
# dataset = "buffer.hdf5"  # the full training buffer of the model
buffer_path = hf_hub_download(
        repo_id="facebook/metamotivo-S-1",
        filename=f"data/{dataset}",
        repo_type="model",
        local_dir=local_dir,
    )
hf = h5py.File(buffer_path, "r")
print(hf.keys())

# create a DictBuffer object that can be used for sampling
data = {k: v[:] for k, v in hf.items()}
buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
buffer.extend(data)
```

### The FB-CPR model
The FB-CPR model contains several networks:
- forward net
- backward net
- critic net
- discriminator net
- actor net

We provide functions for evaluating these networks

```python
def backward_map(self, obs: torch.Tensor) -> torch.Tensor: ...
def forward_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...
def actor(self, obs: torch.Tensor, z: torch.Tensor, std: float) -> torch.Tensor: ...
def critic(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...
def discriminator(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor: ...
```

We also provide simple functions for prompting the model and obtaining a context vector `z` representing the task to execute.
```python
#reward prompt (standard and weighted regression)
def reward_inference(self, next_obs: torch.Tensor, reward: torch.Tensor, weight: torch.Tensor | None = None,) -> torch.Tensor: ...
def reward_wr_inference(self, next_obs: torch.Tensor, reward: torch.Tensor) -> torch.Tensor: ...
#goal prompt
def goal_inference(self, next_obs: torch.Tensor) -> torch.Tensor: ...
#tracking prompt
def tracking_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
```
Once we have a context vector `z` we can call the actor to get actions. We provide a function for acting in the environment with a standard interface.
```python
def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
```

Note that these functions do not allow gradient computation and use eval mode since they are expected to be used for inference (`torch.no_grad()` and `model.eval()`). For training, you should directly access the class attributes. For training we also define target networks for the forward, backward and critic networks.


### Execute a policy

This is the minimal example on how to execute a random policy

```python
from humenv import make_humenv
from gymnasium.wrappers import FlattenObservation, TransformObservation
import torch
from metamotivo.fb_cpr.huggingface import FBcprModel

device = "cpu"
env, _ = make_humenv(
    num_envs=1,
    wrappers=[
        FlattenObservation,
        lambda env: TransformObservation(
            env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device), env.observation_space # For gymnasium <1.0.0 remove the last argument: env.observation_space
        ),
    ],
    state_init="Default",
)

model = FBcprModel.from_pretrained("facebook/metamotivo-S-1")
model.to(device)
z = model.sample_z(1)
observation, _ = env.reset()
for i in range(10):
    action = model.act(observation, z, mean=True)
    observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())
```


# Evaluation in HumEnv

For reproducibility of the paper, we provide a way of evaluating the models using `HumEnv`. We provide wrappers that can be used to interface Meta Motivo with `humenv.bench` reward, goal and tracking evaluation. 

Here is an example of how to use the wrappers for reward evaluation:

```python
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.wrappers.humenvbench import RewardWrapper 
import humenv.bench

model = FBcprModel.from_pretrained("facebook/metamotivo-S-1")

# this enable reward relabeling and context inference
model = RewardWrapper(
        model=model,
        inference_dataset=buffer, # see above how to download and create a buffer
        num_samples_per_inference=100_000,
        inference_function="reward_wr_inference",
        max_workers=80,
    )
# create the evaluation from humenv
reward_eval = humenv.bench.RewardEvaluation(
        tasks=["move-ego-0-0"],
        env_kwargs={
            "state_init": "Default",
        },
        num_contexts=1,
        num_envs=50,
        num_episodes=100
    )
scores = reward_eval.run(model)
```

You can do the same for the other evaluations provided in `humenv.bench`. Please refer to `tutorial_benchmark.ipynb` for a full evaluation loop.

# Rendering a reward-based or tracking policy

We show how to render an episode with a reward-based policy.

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
from humenv import STANDARD_TASKS
import mediapy as media

task = STANDARD_TASKS[0]
model = FBcprModel.from_pretrained("facebook/metamotivo-S-1", device="cpu")
rew_model = RewardWrapper(
        model=model,
        inference_dataset=buffer, # see above how to download and create a buffer
        num_samples_per_inference=100_000,
        inference_function="reward_wr_inference",
        max_workers=40,
        process_executor=True,
        process_context="forkserver"
    )
z = rew_model.reward_inference(task)
env, _ = make_humenv(num_envs=1, task=task, state_init="DefaultAndFall", wrappers=[gymnasium.wrappers.FlattenObservation])
done = False
observation, info = env.reset()
frames = [env.render()]
while not done:
    obs = torch.tensor(observation.reshape(1,-1), dtype=torch.float32, device=rew_model.device)
    action = rew_model.act(obs=obs, z=z).ravel()
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    done = bool(terminated or truncated)

media.show_video(frames, fps=30)
```

It is also easy to render a policy for tracking a motion.

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
from metamotivo.wrappers.humenvbench import TrackingWrapper 
from pathlib import Path
from humenv.misc.motionlib import MotionBuffer

model = FBcprModel.from_pretrained("facebook/metamotivo-S-1", device="cpu")
track_model = TrackingWrapper(model=model)
motion_buffer = MotionBuffer(files=ADD_THE_DESIRED_MOTION, base_path=ADD_YOUR_MOTION_ROOT, keys=["qpos", "qvel", "observation"])
ep_ = motion_buffer.get(motion_buffer.get_motion_ids()[0]
ctx = track_model.tracking_inference(next_obs=ep_["observation"][1:])
observation, info = env.reset(options={"qpos": ep_["qpos"][0], "qvel": ep_["qvel"][0]})
done = False
observation, info = env.reset()
frames = [env.render()]
for t in range(len(ctx)):
    obs = torch.tensor(observation.reshape(1,-1), dtype=torch.float32, device=track_model.device)
    action = track_model.act(obs=obs, z=ctx[t]).ravel()
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())

media.show_video(frames, fps=30)
```

# Citation
```
@article{tirinzoni2024metamotivo,
  title={Zero-shot Whole-Body Humanoid Control via Behavioral Foundation Models},
  author={Tirinzoni, Andrea and Touati, Ahmed and Farebrother, Jesse and Guzek, Mateusz and Kanervisto, Anssi and Xu, Yingchen and Lazaric, Alessandro and Pirotta, Matteo},
}
```

# License

Meta Motivo is licensed under the CC BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
