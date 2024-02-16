def main():
    wandb_run = WandbLogger(
        project="ROAR_PY_RL",
        # entity="roar",
        exp_name=run_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S"),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # wandb_run = type("", (), {})()
    # wandb_run.exp_name = run_name
    # wandb_run.id = 0

    env = get_env(wandb_run)
    # env = gym.make("CarRacing-v2", render_mode="human")
    device = "cuda" if th.cuda.is_available() else "cpu"

    models_path = f"models/{wandb_run.exp_name}"
    # latest_model_path = find_latest_model(Path(models_path)) # note: log path w timestamp won't work w this method

    policy = nn.Sequential(  # default MLPPolicy for SAC (w cnn)
        base_models.MLPPolicy(
            env.observation_space.shape[-1], env.action_space.shape[0] * 2
        ),
    )

    critic = base_models.ObsActionNetwork(
        nn.Sequential(
            base_models.MLPPolicy(
                env.observation_space.shape[-1] + env.action_space.shape[0], 1
            )
        )
    )
    model = sac.SACModel(
        policy,
        critic,
        utils.create_action_space(
            env.action_space.low, env.action_space.high, th.float32, th.device(device)
        ),
        device=device,
    )
    sac_for_sb3 = utils.TRLForSB3(model, env, logger=wandb_run)
    # sac_for_sb3 = utils.TRLForSB3(model, env)

    wandb_callback = WandbCallback(
        gradient_save_freq=1,
        model_save_path=f"models/{wandb_run.exp_name}",
        verbose=2,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=MODEL_SAVE_FREQ, verbose=2, save_path=f"{models_path}/logs"
    )
    callbacks = CallbackList([wandb_callback, checkpoint_callback])

    # event_callback = EveryNTimesteps(
    #     n_steps=MODEL_SAVE_FREQ, callback=checkpoint_callback
    # ) # redundant?
    # callbacks = CallbackList([wandb_callback, checkpoint_callback, event_callback])

    sac_for_sb3.learn(
        total_timesteps=1e7,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False,
        learning_starts=100,
    )
