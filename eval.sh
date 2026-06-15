python -m mineral.scripts.run \
    task=Rewarped \
    agent=DFlexSAPOGAE \
    task.env.env_suite=dflex\
    task.env.env_name=Go2 \
    task.env.numEnvs=16\
    task.env.render=True\
    agent.network.actor_kwargs.mlp_kwargs.units=\[256,128\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,128\] \
    run=eval seed=1 \
    ckpt=/home/dung-admin/ws/test_mineralrewarped/go2/outputs/best_rewards4166.53.pth\
    wandb.mode=disabled 