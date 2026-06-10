import sys
import os

os.environ['JAX_LOG_COMPILES'] = '0'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

import jax

import numpy as np
import random
import wandb

import logging
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from pathlib import Path

import recurrl_jax as rjx
import recurrl_jax.utils.wrappers as rjxw

import env_wrapper as rjx_leap

from recurrl_jax.model_fns import flatten_repr_model

import hydra
from hydra.core.plugins import Plugins
from hydra.core.global_hydra import GlobalHydra
from hydra.core.config_search_path import ConfigSearchPath
from hydra import compose, initialize_config_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def make_env(env_config, trainer_config, global_config):
    num_envs = env_config.get('num_envs', trainer_config.get('num_envs', 8192))

    env = rjx_leap.LeapHandGymWrapper(
        num_envs=num_envs,
        use_domain_randomization=env_config.get('use_domain_randomization', True),
        normalize_obs=True,
        action_scale=env_config.get('action_scale', 0.6),
        action_ema_alpha=env_config.get('action_ema_alpha', 0.0),
        grasp_cache_path=env_config.get('grasp_cache_path', None),
        wrench_force_scale=env_config.get('wrench_force_scale', 5.0),
        wrench_torque_scale=env_config.get('wrench_torque_scale', 0.5),
        alive_bonus=env_config.get('alive_bonus', 2.0),
        wrench_resistance_scale=env_config.get('wrench_resistance_scale', 1.0),
        slip_vel_scale=env_config.get('slip_vel_scale', 0.1),
        torque_scale=env_config.get('torque_scale', 0.0001),
        action_rate_scale=env_config.get('action_rate_scale', 0.01),
        wrench_ramp_alpha=env_config.get('wrench_ramp_alpha', 0.8),
        wrench_push_steps=tuple(env_config.get('wrench_push_steps', (20, 80))),
        wrench_rest_steps=tuple(env_config.get('wrench_rest_steps', (10, 50))),
        # NOUVEAU — coefficients de recompense tactile
        contact_bonus_scale=env_config.get('contact_bonus_scale', 1.0),
        pressure_bonus_scale=env_config.get('pressure_bonus_scale', 0.5),
        palm_bonus_scale=env_config.get('palm_bonus_scale', 0.3),
        low_pressure_scale=env_config.get('low_pressure_scale', 0.5),
        high_pressure_scale=env_config.get('high_pressure_scale', 0.2),
        reward_scale=env_config.get('reward_scale', 0.01),
        update_norm_stats=True,
    )

    env = rjxw.VectorEpisodeStatisticsWrapper(env)
    return env


def make_eval_env(env_config, trainer_config, global_config, train_envs):
    shared_rms = train_envs.env.running_mean_std if hasattr(train_envs, 'env') and hasattr(train_envs.env, 'running_mean_std') else None

    eval_env = rjx_leap.LeapHandGymWrapper(
        num_envs=1,
        use_domain_randomization=False,
        normalize_obs=True,
        action_scale=env_config.get('action_scale', 0.6),
        action_ema_alpha=env_config.get('action_ema_alpha', 0.0),
        grasp_cache_path=env_config.get('grasp_cache_path', None),
        shared_running_mean_std=shared_rms,
        reward_scale=env_config.get('reward_scale', 0.01),
        update_norm_stats=False,
    )

    eval_env = rjxw.SqueezeWrapper(eval_env)
    return eval_env

def make_video_render_fn(eval_env):
    import mujoco

    if hasattr(eval_env, 'env'):
        base_env = eval_env.env
    else:
        base_env = eval_env

    mjx_env = base_env.env
    mj_model = mjx_env.mj_model
    renderer = mujoco.Renderer(mj_model, height=480, width=640)

    def render_fn(env):
        if hasattr(env, 'env'):
            base = env.env
        else:
            base = env

        mjx_env = base.env
        mjx_data = mjx_env.mjx_data_batch
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = np.array(mjx_data.qpos[0])
        mj_data.qvel[:] = np.array(mjx_data.qvel[0])
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data)
        return renderer.render()

    return render_fn


@hydra.main(version_base=None, config_path="config", config_name="default_config")
def main(config: DictConfig):
    logger.info("[LEAP Hand Example]\n" + str(OmegaConf.to_yaml(config)))

    tags = config.tags.split(',') if config.tags is not None else []

    if config.use_wandb:
        run = wandb.init(
            project=config.project_name,
            tags=tags,
            settings=wandb.Settings(start_method="fork"),
            config=OmegaConf.to_container(config)
        )
    else:
        run = None

    key = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    trainer_config = config.trainer
    env_config = config.task

    def video_render_fn_factory(eval_env):
        return make_video_render_fn(eval_env)

    kwargs = {
        'global_args': config,
        'trainer_config': trainer_config,
        'env_config': env_config,
        'seed': config.seed,
        'key': key,
        'wandb_run': run,
    }

    trainer = rjx.Trainer(
        env_factory=make_env,
        eval_env_factory=make_eval_env,
        repr_fn=flatten_repr_model(),
        is_continuous=True,
        video_render_fn=None,
        **kwargs
    )

    if config.get('render_videos', False) and trainer.agent.eval_env is not None:
        trainer.video_render_fn = make_video_render_fn(trainer.agent.eval_env)

    pbar = tqdm(total=config.steps)
    step_count = 0
    last_step_count = 0
    _profile_done = False

    with logging_redirect_tqdm():
        while True:
            if step_count > 0 and not _profile_done:
                with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
                    loss, metrics, step_count = trainer.step()
                    jax.effects_barrier()
                _profile_done = True
            else:
                loss, metrics, step_count = trainer.step()

            pbar.update(n=step_count - last_step_count)
            last_step_count = step_count

            if metrics is not None:
                logger.info(f"Seed: {config.seed} Steps: {step_count} Metrics: {metrics}")
                if config.use_wandb:
                    run.log({'seed': config.seed, **metrics}, step=step_count)

            if step_count >= config.steps:
                break

    pbar.close()

    if config.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
