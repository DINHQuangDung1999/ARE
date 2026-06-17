#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/rewarped:${PYTHONPATH:-}"

for seed in {1,}
do
  python -m mineral.scripts.run \
    task=Rewarped \
    agent=DFlexSAPOGAE \
    task.env.env_suite=dflex\
    task.env.env_name=Go2veltrack \
    task.env.numEnvs=64\
    logdir=../logs/Go2veltrack-SAPOGAEAdpt/seed${seed} \
    agent.shac.max_epochs=2500 \
    agent.shac.max_agent_steps=1e7 \
    agent.shac.horizon_len=32\
    agent.shac.actor_loss_type=adpt-gae\
    agent.shac.num_lambda_batches=4\
    agent.network.actor_kwargs.mlp_kwargs.units=\[256,128\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,128\] \
    run=train_eval seed=${seed} \
    wandb.name=Go2veltrack-SAPOGAEAdpt-${seed} \
    wandb.mode=online \
    wandb.project=Test
done
