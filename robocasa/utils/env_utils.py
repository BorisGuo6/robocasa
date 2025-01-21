from robocasa.utils.dataset_registry import (
    get_ds_path,
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
)
from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset
from robosuite.controllers import load_composite_controller_config
import os
import robosuite
import imageio
import numpy as np
from tqdm import tqdm
from termcolor import colored


def create_env(
    env_name,
    # robosuite-related configs
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=128,
    camera_heights=128,
    seed=None,
    render_onscreen=False,
    # robocasa-related configs
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=render_onscreen,
        has_offscreen_renderer=(not render_onscreen),
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=(not render_onscreen),
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)
    return env


def create_env_from_metadata(
    env_meta,
    env_name=None,
    render=False,
    render_offscreen=False,
    use_image_obs=False,
    use_depth_obs=False,
):
    """
    Create environment.

    Args:
        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        env_name (str): name of environment. Only needs to be provided if making a different
            environment from the one in @env_meta.

        render (bool): if True, environment supports on-screen rendering

        render_offscreen (bool): if True, environment supports off-screen rendering. This
            is forced to be True if @use_image_obs is True.

        use_image_obs (bool): if True, environment is expected to render rgb image observations
            on every env.step call. Set this to False for efficiency reasons, if image
            observations are not required.

        use_depth_obs (bool): if True, environment is expected to render depth image observations
            on every env.step call. Set this to False for efficiency reasons, if depth
            observations are not required.
    """
    if env_name is None:
        env_name = env_meta["env_name"]
    env_type = get_env_type(env_meta=env_meta)
    env_kwargs = env_meta["env_kwargs"]

    env = create_env(
        env_type=env_type,
        env_name=env_name,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        use_depth_obs=use_depth_obs,
        **env_kwargs,
    )
    check_env_version(env, env_meta)
    return env


def run_random_rollouts(env, num_rollouts, num_steps, video_path=None):
    video_writer = None
    if video_path is not None:
        video_writer = imageio.get_writer(video_path, fps=20)

    info = {}
    num_success_rollouts = 0
    for rollout_i in tqdm(range(num_rollouts)):
        obs = env.reset()
        for step_i in range(num_steps):
            # sample and execute random action
            action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
            obs, _, _, _ = env.step(action)

            if video_writer is not None:
                video_img = env.sim.render(
                    height=512, width=768, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(video_img)

            if env._check_success():
                num_success_rollouts += 1
                break

    if video_writer is not None:
        video_writer.close()
        print(colored(f"Saved video of rollouts to {video_path}", color="yellow"))

    info["num_success_rollouts"] = num_success_rollouts

    return info


if __name__ == "__main__":
    # select random task to run rollouts for
    env_name = np.random.choice(
        list(SINGLE_STAGE_TASK_DATASETS) + list(MULTI_STAGE_TASK_DATASETS)
    )
    env = create_eval_env(env_name=env_name)
    info = run_random_rollouts(
        env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4"
    )
