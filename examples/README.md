# Examples

We provide a few examples on how to use the Meta Motivo repository.

## Offline training with ExoRL datasets

[ExoRL](https://github.com/denisyarats/exorl) has been widely used to train offline algorithms. We provide the code for training FB on standard domains such as `walker`, `cheetah`, `quadruped` and `pointmass`. We use the standard tasks in `dm_control`, but you can easily update the script to run the full set of tasks defined in `ExoRL` or in the paper [Fast Imitation via Behavior Foundation Models](https://openreview.net/forum?id=qnWtw3l0jb). We will provide more details below.

To use the provided script you can simply run from terminal

```python
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
