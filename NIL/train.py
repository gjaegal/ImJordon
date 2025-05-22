from bro.bro_learner import BRO
from replay_buffer import ParallelReplayBuffer
# from utils import mute_warning, log_to_wandb_if_time_to, evaluate_if_time_to, make_env

import argparse
import pathlib

import cv2
import gymnasium as gym
import time
import os
import logger
import utils
import numpy as np
import torch
import envs
import jax.numpy as jnp

os.environ["MUJOCO_GL"] = "egl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="environment test")
    parser.add_argument("--env", help="e.g. h1-walk-v0")
    parser.add_argument("--keyframe", default=None)
    parser.add_argument("--policy_path", default=None)
    parser.add_argument("--mean_path", default=None)
    parser.add_argument("--var_path", default=None)
    parser.add_argument("--policy_type", default=None)
    parser.add_argument("--blocked_hands", default="False")
    parser.add_argument("--small_obs", default="False")
    parser.add_argument("--obs_wrapper", default="False")
    parser.add_argument("--sensors", default="")
    parser.add_argument("--render_mode", default="rgb_array")  # "human" or "rgb_array".
    # NOTE: to get (nicer) 'human' rendering to work, you need to fix the compatibility issue between mujoco>3.0 and gymnasium: https://github.com/Farama-Foundation/Gymnasium/issues/749
    parser.add_argument("--log_video", default="True")
    args = parser.parse_args()

    kwargs = vars(args).copy()
    kwargs.pop("env")
    kwargs.pop("render_mode")
    if kwargs["keyframe"] is None:
        kwargs.pop("keyframe")
    print(f"arguments: {kwargs}")

    # Log directory
    data_path = "../vid_logs"
    log_dir = data_path + "/" + "walk_test_" + time.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    mylogger = logger.Logger(log_dir=log_dir)

    # TODO Generated reference video
    # generated_video = ...

    # TODO 가비아에서 make custom env 오류 해결
    env = gym.make(args.env, render_mode=args.render_mode, **kwargs)
    # env = gym.make("Humanoid-v5", render_mode=args.render_mode)


    seed = 0
    fps = 4
    # agent
    print("observation space: ", env.observation_space.sample().shape)
    print("action space: ", env.action_space.sample().shape)
    dummy_obs = jnp.zeros((1, 51))
    dummy_ac = jnp.zeros((1, 19))
    updates_per_step = 10

    agent = BRO(
        seed,
        dummy_obs, # env.observation_space.sample()[0, np.newaxis],
        dummy_ac, # env.action_space.sample()[0, np.newaxis],
        num_seeds=1,
        updates_per_step=updates_per_step,
        distributional=True,
    )
    # Replay buffer
    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1], 1000000, num_seeds=1)
    
    ob, _ = env.reset()

    start_training_step = 5
    seg_masks = []
    generated_seg_masks= []
    # TODO for step in range(len(generated_video))
    for step in range(20):
        if ob.shape[0] != 1:
            ob = ob.reshape(1, -1)
        print("ob", ob.shape)

        actions = agent.sample_actions_o(ob, temperature=1.0)
        actions = actions.squeeze(0)

        print("ac", actions.shape)

        next_ob, rewards, terminated, truncated, info = env.step(actions)
        print("STEP: ", step)

        # extract joint positions / torques / velocities as numpy array
        # TODO past frame action, foot contact with ground, stability
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        position, velocity, torque = base_env.get_observations()
        # position: 19 dim
        # velocity: 19 dim
        
        # joint_positions = []
        # data = env.unwrapped.data
        # for i in range(data.nq):
        #     joint_positions.append(data.qpos[i])
        # joint_positions = np.array(joint_positions)

        # joint_velocities = data.qvel.copy()
        # joint_velcities = np.array(joint_velocities)
        # joint_torques = data.actuator_force.copy()
        # joint_torques = np.array(joint_torques)
        # joint_torques = data.qfrc_actuator.copy()

        # TODO extract segmenation masked image -> 우리 환경에서 제대로 작동하는지 확인
        # seg_mask = env.render(mode="depth", camera_name="track")
        # seg_masks.append(seg_mask)

        # TODO extract segmentation masked image from generated video
        # generated_seg_mask = SAM(generated_video[step])
        # generated_seg_masks.append(generated_seg_mask)

        # TODO clip of past 8 frames
        # CLIP(seg_masks[:], generated_seg_masks[:])

        # TODO =============== REWARD ====================
        alpha, beta, gamma = 1.0, 1.0, 1.0
        regularization_reward = 0.0
        iou_reward = 0.0
        l2_reward = 0.0
        # regularization_reward = REGULARIZATION(joint_positions, joint_velocities, joint_torques, ...)
        # iou_reward = VIDEO_SIMULARITY(seg_mask, generated_seg_mask)
        # l2_reward = IMAGE_SIMULARTIY(CLIP())
        nil_reward = alpha* l2_reward + beta * iou_reward + gamma * regularization_reward

        # TODO masks = env.generate_masks(terminated, truncated)
        masks = [1.0]
        if not truncated:
            replay_buffer.insert(ob, actions, nil_reward, masks, truncated, next_ob)
        ob = next_ob
        # TODO ob, terminated, truncated, reward_mask = env.reset_when_done(ob, terminated, truncated)

        # Train
        if step > start_training_step:
            batches = replay_buffer.sample_parallel_multibatch(batch_size=128, num_batches=10)
            # import pdb; pdb.set_trace()
            infos = agent.update(batches, updates_per_step, step)
        



    if args.log_video == "True":
        # simulate with trained agent
        trajs = utils.rollout_n_trajectories(env, policy=agent, ntraj=1, max_traj_length=5000, render=True, seg_render=False)

        # tensorboard에 video 형식으로 저장
        mylogger.log_trajs_as_videos(trajs, step=0, max_videos_to_save=1, fps=10, video_title="test_basketball")

    del trajs
    env.close()