import functools
import os
import pprint
import sys

import hydra
import numpy as np
import wandb
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint

import torch
from copy import deepcopy

def make_envs(config):
    from .. import envs

    task_suite = config.task.get('suite', 'isaacgymenvs')
    TaskSuite = getattr(envs, task_suite)

    return TaskSuite.make_envs(config)


def make_datasets(config, env):
    if not hasattr(config.agent, 'datasets'):
        return None

    from .. import envs

    task_suite = config.task.get('suite', 'isaacgymenvs')
    TaskSuite = getattr(envs, task_suite)
    return TaskSuite.make_datasets(config, env)


def save_run_metadata(logdir, run_name, run_id, resolved_config):
    run_metadata = {
        'logdir': logdir,
        'run_name': run_name,
        'run_id': run_id,
    }
    yaml.dump(run_metadata, open(os.path.join(logdir, 'run_metadata.yaml'), 'w'), default_flow_style=False)
    yaml.dump(resolved_config, open(os.path.join(logdir, 'resolved_config.yaml'), 'w'), default_flow_style=False)


def main(config: DictConfig):
    if 'isaacgym' in sys.modules or 'isaacgymenvs' in sys.modules:
        from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    else:
        from .utils import set_np_formatting, set_seed

    # set numpy formatting for printing only
    set_np_formatting()

    from .utils import limit_threads

    limit_threads(1)

    import torch

    if not torch.cuda.is_available():
        config.device_id = -1

    assert config.seed >= 0
    set_seed = functools.partial(set_seed, torch_deterministic=config.torch_deterministic)
    set_seed(0)

    # --- Setup Run ---
    logdir = config.logdir
    os.makedirs(logdir, exist_ok=True)

    if config.ckpt:
        config.ckpt = to_absolute_path(config.ckpt)

    if config.multi_gpu:
        from accelerate import Accelerator

        accelerator = Accelerator()
        rank = int(os.getenv('LOCAL_RANK', '0'))
        if str(accelerator.device) == 'cuda':
            pass
        else:
            assert rank == accelerator.device.index, print(rank, accelerator.device, accelerator.device.index)

        config.sim_device = f'cuda:{rank}'
        config.rl_device = f'cuda:{rank}'
        config.graphics_device_id = rank

        # if rank != 0:
        #     f = open(os.path.join(config.logdir, 'log_{rank}.txt', 'w'))
        #     sys.stdout = f
    else:
        accelerator = None
        rank = 0

        # use the same device for sim and rl
        config.sim_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'
        config.rl_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'
        config.graphics_device_id = config.device_id if config.device_id >= 0 else 0

    resolved_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print(pprint.pformat(resolved_config, compact=True, indent=1), '\n')

    if rank == 0:
        os.environ['WANDB_START_METHOD'] = 'thread'
        # connect to wandb
        wandb_config = OmegaConf.to_container(config.wandb, resolve=True)

        if wandb_config.get('group', None) is not None:
            run_group = wandb_config['group']
        else:
            run_group = logdir.split('/')[-2]
            wandb_config['group'] = run_group

        wandb_run = wandb.init(
            **wandb_config,
            dir=logdir,
            config=resolved_config,
        )
        run_name, run_id = wandb_run.name, wandb_run.id
        print(f'run_group: {run_group}, run_name: {run_name}, run_id: {run_id}')
        save_run_metadata(logdir, run_name, run_id, resolved_config)

    # generate random seeds (deterministically given config.seed)
    # should be same across workers since each worker should add its global rank
    to_seed = ['setup', 'train', 'eval', 'env_train', 'env_eval']
    rng = np.random.RandomState(config.seed)
    seeds = rng.randint(0, int(1e6), len(to_seed))
    seeds = {k: int(seeds[i]) for i, k in enumerate(to_seed)}
    cprint(f'Base Seed: {config.seed}, Seeds: {seeds}', 'green', attrs=['bold'])
    set_seed(seeds['setup'] + rank)

    # --- Make Envs, Datasets, Agent ---
    cprint('Making Envs', 'green', attrs=['bold'])
    env = make_envs(config)
    # breakpoint()
    print('-' * 20)
    print(f'Env: {env}')

    datasets = make_datasets(config, env)
    print(f'Datasets: {datasets}')

    from .. import agents

    set_seed(seeds['eval'] + rank)
    AgentCls = getattr(agents, config.agent.algo)
    agent = AgentCls(config, logdir=logdir, accelerator=accelerator, datasets=datasets, env=env)
    

    counter = 0
    loss_array = []
    loss_array_TD = []
    range_1 = torch.linspace(-1,1,40)
    range_2 = torch.linspace(-1,1,40)
    with torch.no_grad():
        loss_array = []
        for x1 in range_1:
            for x2 in range_2:
                
                counter+=1
                print(f'[{counter}/{40*40}]')

                if config.ckpt:
                    print(f'Loading checkpoint: {config.ckpt}')
                    agent.load(config.ckpt, ckpt_keys=config.ckpt_keys)
                for k, v in agent.actor.named_parameters():
                    v.requires_grad = False
                agent.initialize_env()
                agent.actor.actor_mlp.mlp[0].weight[0,0] += x1
                agent.actor.actor_mlp.mlp[0].weight[0,1] += x2
                
                rew_acc = torch.zeros((agent.horizon_len + 1, agent.num_envs), dtype=torch.float32, device=agent.device)
                gamma = torch.ones(agent.num_envs, dtype=torch.float32, device=agent.device)
                next_values = torch.zeros((agent.horizon_len + 1, agent.num_envs), dtype=torch.float32, device=agent.device)
                avg_next_values = torch.zeros((agent.horizon_len + 1, agent.num_envs), dtype=torch.float32, device=agent.device)
                rew_buf = torch.zeros((agent.horizon_len, agent.num_envs), dtype=torch.float32, device=agent.device) # local buffer that saves grad
                entropy_buf = torch.zeros((agent.horizon_len, agent.num_envs), dtype=torch.float32, device=agent.device) # local buffer that saves grad

                returns = torch.zeros(agent.num_envs, dtype=torch.float32, device=agent.device)
                logprobs = torch.zeros(agent.num_envs, dtype=torch.float32, device=agent.device)
                distr_ents = torch.zeros(agent.num_envs, dtype=torch.float32, device=agent.device)

                with torch.no_grad():
                    if agent.obs_rms is not None:
                        obs_rms = deepcopy(agent.obs_rms)

                    alpha = agent.get_alpha(scalar=True) if agent.with_autoent else agent.entropy_coef

                # initialize trajectory to cut off gradients between episodes.
                obs = agent.env.initialize_trajectory()
                obs = agent._convert_obs(obs)

                if agent.obs_rms is not None:
                    # update obs rms
                    with torch.no_grad():
                        for k, v in obs.items():
                            agent.obs_rms[k].update(v)
                    # normalize the current obs
                    obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}

                for i in range(agent.horizon_len):

                    # take env step
                    z = agent.actor_encoder(obs)
                    actions, mu, sigma, distr = agent.get_actions(obs, z=z, sample=False, dist=True)

                    if agent.with_logprobs:
                        logprob = distr.log_prob(actions).sum(dim=-1)
                        distr_ent = distr.entropy().sum(dim=-1)

                    obs, rew, done, extra_info = agent.env.step(actions)
                    obs = agent._convert_obs(obs)

                    # scale the reward
                    rew = agent.reward_shaper(rew)
                    rew_buf[i] = rew
                    if agent.obs_rms is not None:
                        # update obs rms
                        with torch.no_grad():
                            for k, v in obs.items():
                                agent.obs_rms[k].update(v)
                        # normalize the current obs
                        obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}

                    # value bootstrap when episode terminates
                    if agent.share_encoder:
                        z_target = z
                    else:
                        z_target = agent.encoder_target(obs)
                    pred_val, avg_pred_val = agent.critic_target(z_target, return_type="min_and_avg")
                    pred_val, avg_pred_val = pred_val.squeeze(-1), avg_pred_val.squeeze(-1)
                    next_values[i + 1] = pred_val
                    avg_next_values[i + 1] = avg_pred_val

                    done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
                    if len(done_env_ids) > 0:
                        terminal_obs = extra_info['obs_before_reset']
                        terminal_obs = agent._convert_obs(terminal_obs)

                        for id in done_env_ids:
                            nan = False
                            # TODO: some elements of obs_dict (for logging) may be nan, add regex to ignore these
                            for k, v in terminal_obs.items():
                                if (
                                    (torch.isnan(v[id]).sum() > 0)
                                    or (torch.isinf(v[id]).sum() > 0)
                                    or ((torch.abs(v[id]) > 1e6).sum() > 0)
                                ):  # ugly fix for nan values
                                    print(f'nans at terminal obs: {k}. nan: {torch.isnan(v[id]).sum() > 0}, inf: {(torch.isinf(v[id]).sum() > 0)}, cap: {(torch.abs(v[id]) > 1e6).sum() > 0}.')
                                    nan = True
                                    break
                            if nan:
                                next_values[i + 1, id] = 0.0
                                avg_next_values[i + 1, id] = 0.0
                            elif agent.episode_lengths[id] < agent.max_episode_length:  # early termination
                                next_values[i + 1, id] = 0.0
                                avg_next_values[i + 1, id] = 0.0
                            else:  # otherwise, use terminal value critic to estimate the long-term performance
                                real_obs = {k: v[id.reshape(1)] for k, v in terminal_obs.items()}
                                if agent.obs_rms is not None:
                                    real_obs = {k: obs_rms[k].normalize(v) for k, v in real_obs.items()}
                                real_z_target = agent.encoder_target(real_obs)
                                real_next_values, avg_real_next_values = agent.critic_target(real_z_target, return_type="min_and_avg")
                                real_next_values, avg_real_next_values = real_next_values.squeeze(-1), avg_real_next_values.squeeze(-1)
                                next_values[i + 1, id] = real_next_values
                                avg_next_values[i + 1, id] = avg_real_next_values

                    if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                        print('next value error')
                        raise ValueError
                    if (avg_next_values[i + 1] > 1e6).sum() > 0 or (avg_next_values[i + 1] < -1e6).sum() > 0:
                        print('avg next value error')
                        raise ValueError

                    # https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/agents/sac/sac_learner.py#L169
                    next_vs = avg_next_values if agent.actor_loss_avgcritics else next_values

                    # compute actor loss
                    if agent.entropy_in_return:
                        # operations to entropy should be out of place since cloning them further below
                        entropy = distr_ent if agent.use_distr_ent else -1.0 * logprob
                        entropy = entropy.clone()
                        if agent.offset_by_target_entropy:
                            entropy = (entropy + abs(agent.target_entropy)) * 0.5
                        if agent.scale_by_target_entropy:
                            entropy = entropy * (1.0 / abs(agent.target_entropy))
                        entropy_buf[i] = entropy
                        rew_acc[i + 1, :] = rew_acc[i, :] + gamma * (rew + alpha * entropy)
                    else:
                        rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew
                    if i < agent.horizon_len - 1:
                        rets = rew_acc[i + 1, done_env_ids] + agent.gamma * gamma[done_env_ids] * next_vs[i + 1, done_env_ids]
                        returns[done_env_ids] += rets
                    else:
                        # terminate all envs at the end of optimization iteration
                        rets = rew_acc[i + 1, :] + agent.gamma * gamma * next_vs[i + 1, :]
                        returns += rets

                    if agent.with_logprobs:
                        logprobs += logprob
                        distr_ents += distr_ent

                    # compute gamma for next step
                    gamma = gamma * agent.gamma

                    # clear up gamma and rew_acc for done envs
                    gamma[done_env_ids] = 1.0
                    rew_acc[i + 1, done_env_ids] = 0.0
                
                # compute SHAC loss
                returns /= agent.horizon_len
                logprobs /= agent.horizon_len
                distr_ents /= agent.horizon_len

                returns = returns.squeeze(-1)
                if agent.entropy_in_return or agent.no_actor_entropy:  # entropy will also be discounted
                    actor_loss = -returns.mean()
                elif agent.with_autoent or agent.entropy_coef is not None:  # here entropy is not discounted
                    alpha = agent.get_alpha(scalar=True) if agent.with_autoent else agent.entropy_coef
                    entropy = distr_ents if agent.use_distr_ent else -1.0 * logprobs
                    if agent.offset_by_target_entropy:
                        entropy = (entropy + abs(agent.target_entropy)) * 0.5
                    if agent.scale_by_target_entropy:
                        entropy = entropy * (1.0 / abs(agent.target_entropy))
                    actor_loss = ((alpha * -entropy) - returns).mean()
                else:
                    actor_loss = -returns.mean()
                loss_array.append((x1.item(), x2.item(), actor_loss.item()))

                # compute SHACTD loss
                Vnext = torch.zeros(agent.num_envs, dtype=torch.float32, device=agent.device)
                Vtilde = torch.zeros((agent.horizon_len, agent.num_envs), dtype=torch.float32, device=agent.device)

                for i in reversed(range(agent.horizon_len)):
                    rew = rew_buf[i] + alpha * entropy_buf[i] if agent.entropy_in_return else rew_buf[i]
                    Vnext = rew + agent.gamma * ((1-agent.done_mask[i]) * ((1-agent.lam) * next_vs[i+1] + agent.lam * Vnext)
                                                            + agent.done_mask[i] * next_vs[i+1])
                    Vtilde[i] = Vnext

                returns = Vtilde[0]            
                
                returns /= agent.horizon_len
                logprobs /= agent.horizon_len
                distr_ents /= agent.horizon_len

                returns = returns.squeeze(-1)
                if agent.entropy_in_return or agent.no_actor_entropy:  # entropy will also be discounted
                    actor_loss = -returns.mean()
                elif agent.with_autoent or agent.entropy_coef is not None:  # here entropy is not discounted
                    alpha = agent.get_alpha(scalar=True) if agent.with_autoent else agent.entropy_coef
                    entropy = distr_ents if agent.use_distr_ent else -1.0 * logprobs
                    if agent.offset_by_target_entropy:
                        entropy = (entropy + abs(agent.target_entropy)) * 0.5
                    if agent.scale_by_target_entropy:
                        entropy = entropy * (1.0 / abs(agent.target_entropy))
                    actor_loss = ((alpha * -entropy) - returns).mean()
                else:
                    actor_loss = -returns.mean()
                loss_array_TD.append((x1.item(), x2.item(), actor_loss.item()))

    loss_array = np.array(loss_array)
    loss_array_TD = np.array(loss_array_TD)
    np.save(f'loss_landscape_{config.agent.shac.name}', loss_array)
    np.save(f'loss_landscape_{config.agent.shac.name}TD', loss_array_TD)

if __name__ == '__main__':
    c = []
    hydra.main(
        config_name='config',
        config_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), '../cfgs'),
        version_base='1.1',
    )(lambda x: c.append(x))()
    config = c[0]

    task_suite = config.task.get('suite', 'isaacgymenvs')
    if task_suite == 'isaacgymenvs':
        from ..envs.isaacgymenvs import import_isaacgym

        import_isaacgym()  # (need to import before torch)

    main(config)
