<p align="center">
    <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>
    <a href="https://pypi.org/project/rewarped/">
    <img src="https://img.shields.io/pypi/v/rewarped" /></a>
    <a href="https://arxiv.org/abs/2412.12089">
    <img src="https://img.shields.io/badge/arxiv-2412.12089-b31b1b" /></a>
</p>

# Rewarped

Rewarped 🌀 is a platform for reinforcement learning in parallel differentiable multiphysics simulation, built on [`NVIDIA/warp`](https://github.com/NVIDIA/warp). Rewarped supports:

- **parallel environments**: to train RL agents at scale.
- **differentiable simulation**: to compute batched analytic gradients for optimization.
- **multiphysics**: (CRBA, FEM, MPM, XPBD) physics solvers and coupling to support interaction between rigid bodies, articulations, and various deformables.

We use Rewarped to *re-implement* a variety of RL tasks from prior works, and demonstrate that first-order RL algorithms (which use differentiable simulation to compute first-order analytic gradients) can be scaled to a range of challenging manipulation and locomotion tasks that involve interaction between rigid bodies, articulations, and deformables.

> For control and reinforcement learning algorithms, see [`etaoxing/mineral`](https://github.com/etaoxing/mineral).

# Setup

We have tested on the following environment: RTX 4090, Ubuntu 22.04, CUDA 12.5, Python 3.10, PyTorch 2.

```bash
conda create -n rewarped python=3.10
conda activate rewarped

pip install torch torchvision
pip install gym==0.23.1
pip install rewarped

# --- Example: trajectory optimization
python -m rewarped.envs.warp_examples.bounce --num_envs 4
# will create a `.usd` file in `outputs/`
# use MacOS Preview or alternatives to view

# --- Example: (first-order) reinforcement learning
pip install git+https://github.com/etaoxing/mineral

python -m mineral.scripts.run \
  task=Rewarped agent=DFlexAntSAPO task.env.env_name=Ant task.env.env_suite=dflex \
  logdir="workdir/RewarpedAnt4M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
  agent.shac.max_epochs=2000 agent.shac.max_agent_steps=4.1e6 \
  agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
  agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
  run=train_eval seed=1000
```

> See [`mineral/docs/rewarped.md`](https://github.com/etaoxing/mineral/blob/main/docs/rewarped.md) for scripts to reproduce the original paper experiments on `rewarped==1.3.0`.

# Usage

## Tasks

<table>
  <tbody>
  <tr>
    <td>
      <a href="./rewarped/envs/dflex/ant.py"><img src="./docs/assets/antrun.png"/></a>
    </td>
    <td>
      <a href="./rewarped/envs/isaacgymenvs/allegro_hand.py"><img src="./docs/assets/handreorient.png"/></a>
    </td>
    <td>
      <a href="./rewarped/envs/plasticinelab/rolling_pin.py"><img src="./docs/assets/rollingflat.png"/></a>
    </td>
  </tr>
  <tr>
    <td align="center">dflex &#8226; AntRun</td>
    <td align="center">isaacgymenvs &#8226; HandReorient</td>
    <td align="center">plasticinelab &#8226; RollingFlat</td>
  </tr>
  <tr>
    <td>
      <a href="./rewarped/envs/gradsim/jumper.py"><img src="./docs/assets/softjumper.png"/></a>
    </td>
    <td>
      <a href="./rewarped/envs/dexdeform/flip.py"><img src="./docs/assets/handflip.png"/></a>
    </td>
    <td>
      <a href="./rewarped/envs/softgym/transport.py"><img src="./docs/assets/fluidmove.png"/></a>
    </td>
  </tr>
  <tr>
    <td align="center">gradsim &#8226; SoftJumper</td>
    <td align="center">dexdeform &#8226; HandFlip</td>
    <td align="center">softgym &#8226; FluidMove</td>
  </tr>
</tbody>
</table>

## Creating a new task

Copy and modify [`rewarped/envs/template/task.py`](rewarped/envs/template/task.py) into `rewarped/envs/<suite>/<task>.py`. Put assets in `assets/<suite>/`.

# Contributing

Contributions are welcome! Please refer to [`CONTRIBUTING.md`](CONTRIBUTING.md).

# Versioning

Versions take the format X.Y.Z, where:
- X.Y matches to version X.Y.* of `NVIDIA/warp`.
- Z is incremented for bug fixes or new features.

Compatibility with newer or older versions of `NVIDIA/warp` may work, but is not guaranteed or supported. There can be some breaking changes between minor versions.

# Acknowledgements

Differentiable Simulation
- [`NVIDIA/warp`](https://github.com/NVIDIA/warp)
- [`NVlabs/DiffRL`](https://github.com/NVlabs/DiffRL)
- MPM
  - [`PingchuanMa/SGA`](https://github.com/PingchuanMa/SGA), [`PingchuanMa/NCLaw`](https://github.com/PingchuanMa/NCLaw)
  - [`sizhe-li/DexDeform`](https://github.com/sizhe-li/DexDeform)
  - [`hzaskywalker/PlasticineLab`](https://github.com/hzaskywalker/PlasticineLab)

Tasks (alphabetical)
- [`sizhe-li/DexDeform`](https://github.com/sizhe-li/DexDeform)
- [`NVlabs/DiffRL`](https://github.com/NVlabs/DiffRL)
- [`hzaskywalker/PlasticineLab`](https://github.com/hzaskywalker/PlasticineLab)
- [`gradsim/gradsim`](https://github.com/gradsim/gradsim)
- [`isaac-sim/IsaacGymEnvs`](https://github.com/isaac-sim/IsaacGymEnvs)
- [`leggedrobotics/legged_gym`](https://github.com/leggedrobotics/legged_gym)
- [`Xingyu-Lin/softgym`](https://github.com/Xingyu-Lin/softgym)
- ...

# Citing

```bibtex
@inproceedings{xing2025stabilizing,
  title={Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation},
  author={Xing, Eliot and Luk, Vernon and Oh, Jean},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=DRiLWb8bJg}
}
```
