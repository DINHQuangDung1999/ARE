#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/rewarped:${PYTHONPATH:-}"

python -m mineral.scripts.run \
    task=Rewarped \
    agent=DFlexSAPOGAE \
    task.env.env_suite=dflex\
    task.env.env_name=Go2 \
    task.env.numEnvs=4\
    task.env.render=True\
    logdir=logs_eval/RewarpedGo2-SAPOGAE/seed1 \
    agent.network.actor_kwargs.mlp_kwargs.units=\[256,128\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,128\] \
    run=eval seed=1 \
    ckpt=logs/RewarpedGo2-SAPOGAE/seed1/ckpt/best_rewards1067.57.pth\
    wandb.mode=disabled 