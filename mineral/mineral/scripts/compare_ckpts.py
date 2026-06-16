import functools
import os
import pprint
import sys

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint


def make_envs(config):
    from .. import envs

    task_suite = config.task.get("suite", "isaacgymenvs")
    TaskSuite = getattr(envs, task_suite)
    return TaskSuite.make_envs(config)


def make_agent(config, env, logdir_suffix):
    from .. import agents

    AgentCls = getattr(agents, config.agent.algo)
    agent_logdir = os.path.join(config.logdir, logdir_suffix)
    os.makedirs(agent_logdir, exist_ok=True)
    return AgentCls(config, logdir=agent_logdir, accelerator=None, datasets=None, env=env)


def slice_obs(obs, env_idx):
    if isinstance(obs, dict):
        return {k: v[env_idx : env_idx + 1] for k, v in obs.items()}
    return obs[env_idx : env_idx + 1]


def normalize_obs(agent, obs):
    if agent.obs_rms is None:
        return obs
    return {k: agent.obs_rms[k].normalize(v) for k, v in obs.items()}


def main(config: DictConfig):
    if "isaacgym" in sys.modules or "isaacgymenvs" in sys.modules:
        from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    else:
        from .utils import set_np_formatting, set_seed

    from .utils import limit_threads

    limit_threads(1)
    set_np_formatting()

    import torch

    if not torch.cuda.is_available():
        config.device_id = -1

    ckpt_a = config.get("ckpt_a", "")
    ckpt_b = config.get("ckpt_b", "")
    assert ckpt_a and ckpt_b, "Pass both +ckpt_a=... and +ckpt_b=..."

    ckpts = [to_absolute_path(ckpt_a), to_absolute_path(ckpt_b)]
    compare_episodes = int(config.get("compare_episodes", 1))
    sample = bool(config.get("compare_sample", False))

    if int(config.task.env.numEnvs) != len(ckpts):
        print(f"Overriding task.env.numEnvs from {config.task.env.numEnvs} to {len(ckpts)}")
        config.task.env.numEnvs = len(ckpts)

    assert config.seed >= 0
    set_seed = functools.partial(set_seed, torch_deterministic=config.torch_deterministic)
    set_seed(config.seed)

    config.sim_device = f"cuda:{config.device_id}" if config.device_id >= 0 else "cpu"
    config.rl_device = f"cuda:{config.device_id}" if config.device_id >= 0 else "cpu"
    config.graphics_device_id = config.device_id if config.device_id >= 0 else 0

    resolved_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print(pprint.pformat(resolved_config, compact=True, indent=1), "\n")
    print("Compare checkpoints:")
    for i, ckpt in enumerate(ckpts):
        print(f"  env{i}: {ckpt}")

    cprint("Making Env", "green", attrs=["bold"])
    env = make_envs(config)
    print("-" * 20)
    print(f"Env: {env}")

    agents = []
    for i, ckpt in enumerate(ckpts):
        agent = make_agent(config, env, f"compare_agent_{i}")
        print(f"Loading checkpoint for env{i}: {ckpt}")
        agent.load(ckpt, ckpt_keys=config.ckpt_keys)
        agent.set_eval()
        agents.append(agent)

    obs = env.reset()
    obs = agents[0]._convert_obs(obs)

    episode_rewards = torch.zeros(len(ckpts), dtype=torch.float32, device=env.device)
    episode_lengths = torch.zeros(len(ckpts), dtype=torch.int32, device=env.device)
    completed_episodes = [0 for _ in ckpts]

    cprint("Running Comparison", "green", attrs=["bold"])
    while min(completed_episodes) < compare_episodes:
        action_batches = []
        with torch.no_grad():
            for env_idx, agent in enumerate(agents):
                agent_obs = normalize_obs(agent, slice_obs(obs, env_idx))
                action_batches.append(agent.get_actions(agent_obs, sample=sample))
        actions = torch.cat(action_batches, dim=0)

        obs, rew, done, _ = env.step(actions)
        obs = agents[0]._convert_obs(obs)

        episode_rewards += rew
        episode_lengths += 1

        done_ids = done.nonzero(as_tuple=False).squeeze(-1).tolist()
        for env_idx in done_ids:
            completed_episodes[env_idx] += 1
            print(
                f"env{env_idx} episode {completed_episodes[env_idx]}: "
                f"reward={episode_rewards[env_idx].item():.2f}, length={int(episode_lengths[env_idx].item())}"
            )
            episode_rewards[env_idx] = 0.0
            episode_lengths[env_idx] = 0

    if hasattr(env, "renderer") and env.renderer is not None:
        env.renderer.save()
        print(f"Saved render output under {env.render_dir}")


if __name__ == "__main__":
    c = []
    hydra.main(
        config_name="config",
        config_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "../cfgs"),
        version_base="1.1",
    )(lambda x: c.append(x))()
    config = c[0]

    task_suite = config.task.get("suite", "isaacgymenvs")
    if task_suite == "isaacgymenvs":
        from ..envs.isaacgymenvs import import_isaacgym

        import_isaacgym()

    main(config)
