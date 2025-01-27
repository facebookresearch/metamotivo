# Examples

We provide a few examples on how to use the Meta Motivo repository.

## FB: Offline training with ExoRL datasets

[ExoRL](https://github.com/denisyarats/exorl) has been widely used to train offline algorithms. We provide the code for training FB on standard domains such as `walker`, `cheetah`, `quadruped` and `pointmass`. We use the standard tasks in `dm_control`, but you can easily update the script to run the full set of tasks defined in `ExoRL` or in the paper [Fast Imitation via Behavior Foundation Models](https://openreview.net/forum?id=qnWtw3l0jb). We will provide more details below.

To use the provided script you can simply run from terminal

```bash
python fb_train_dmc.py --domain_name walker --dataset_root <root to exorl data>
```

The standard folder structure of ExoRL is `<prefix>/datasets/${DOMAIN}/${ALGO}/buffer` so we expect `dataset_root=<prefix>/datasets`. Since the original creation of ExORL, mujoco has seen many updates. To rerun all the actions and collect a physics consistent data, you may optionally replay the trajectories. We refer to [https://github.com/facebookresearch/mtm/tree/main/research/exorl](https://github.com/facebookresearch/mtm/tree/main/research/exorl) for this.

If you want to run auxiliary tasks and domains such as `walker_flip` or `pointmass` we suggest to download the files from [https://github.com/facebookresearch/offline_rl/tree/main/src/dmc_tasks](https://github.com/facebookresearch/offline_rl/tree/main/src/dmc_tasks) into `examples/dmc_tasks`. You can thus simply modify `fb_train_dmc.py` as follows:

- add import
```
from dmc_tasks import dmc
```
- add new tasks
```
ALL_TASKS = {
    "walker": ["walk", "run", "stand", "flip", "spin"],
    "cheetah": ["walk", "run", "walk_backward", "run_backward"],
    "pointmass": ["reach_top_left", "reach_top_right", "reach_bottom_right", "reach_bottom_left", "loop", "square", "fast_slow"],
    "quadruped": ["jump", "walk", "run", "stand"],
}
```
- use `dmc.make` for environment creation. For example, replace `suite.load(domain_name=self.cfg.domain_name,task_name=task,environment_kwargs={"flat_observation": True},)` with `dmc.make(f"{self.cfg.domain_name}_{task}")`.
- This changes the way of getting the observation from `time_step.observation["observations"]` to simply `time_step.observation`. Update the file accordingly.


## FB-CPR: Online training with HumEnv

We provide a complete code for training FB-CPR as described in the paper [Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models](https://ai.meta.com/research/publications/zero-shot-whole-body-humanoid-control-via-behavioral-foundation-models/). 

**IMPORTANT!** We assume you have already preprocessed the AMASS motions as described [here](https://github.com/facebookresearch/humenv/tree/main/data_preparation). In addition, we assume you also downloaded the `test_train_split` sub-folder.

The script is setup with the S configuration (i.e., paper configuration) and can be run by simply calling

```bash
python fbcpr_train_humenv.py --compile --motions test_train_split/large1_small1_train_0.1.txt --motions_root <root to AMASS motions> --prioritization
```

There are several parameters that can be changed to do evaluation more modular, checkpoint the models, etc. We refer to the code for more details.

If you would like to train our largest model (the one deployed in the [demo](https://metamotivo.metademolab.com/)), replace the following line

```
model, hidden_dim, hidden_layers = "simple", 1024, 2
```

with

```
model, hidden_dim, hidden_layers = "residual", 2048, 12
```

NOTE: we recommend that you use compile=True on a A100 GPU or better, as otherwise training can be very slow.
