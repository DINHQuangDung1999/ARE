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
  task.env.numEnvs=32\
  logdir=logs/RewarpedGo2-SAPOGAE/seed1 \
  agent.shac.max_epochs=2000 \
  agent.shac.max_agent_steps=1e7 \
  agent.shac.horizon_len=32\
  agent.network.actor_kwargs.mlp_kwargs.units=\[256,128\] \
  agent.network.critic_kwargs.mlp_kwargs.units=\[256,128\] \
  run=train_eval seed=1 \
  wandb.name=Go2-SAPO-1 \
  wandb.mode=online \
  wandb.project=Test
