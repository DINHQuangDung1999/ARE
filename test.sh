python -m mineral.scripts.run \
  task=Rewarped \
  agent=DFlexSAPOGAE \
  task.env.env_suite=dflex\
  task.env.env_name=Go2 \
  task.env.numEnvs=4\
  logdir=logs/RewarpedGo2-SAPOGAE/seed1 \
  agent.shac.max_epochs=2000 \
  agent.shac.max_agent_steps=1e7 \
  agent.shac.horizon_len=16\
  agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
  agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
  run=train_eval seed=1 \
  wandb.name=Go2-SAPO-1 \
  wandb.mode=disabled \
  wandb.project=MineralRewarpedGo2
