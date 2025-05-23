import argparse
import pathlib

import cv2
import gymnasium as gym
import time
import os

import humanoid_bench
from .env import ROBOTS, TASKS
from humanoid_bench import utils
from humanoid_bench import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="HumanoidBench environment test")
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

    # # Test offscreen rendering
    # print(f"Test offscreen mode...")
    # env = gym.make(args.env, render_mode="rgb_array", **kwargs)
    # ob, _ = env.reset()
    # if isinstance(ob, dict):
    #     print(f"ob_space = {env.observation_space}")
    #     print(f"ob = ")
    #     for k, v in ob.items():
    #         print(f"  {k}: {v.shape}")
    # else:
    #     print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
    # print(f"ac_space = {env.action_space.shape}")

    # img = env.render()
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("test_env_img.png", rgb_img)

    # Test online rendering with interactive viewer
    print(f"Test onscreen mode...")
    env = gym.make(args.env, render_mode=args.render_mode, **kwargs)
    ob, _ = env.reset()
    if isinstance(ob, dict):
        print(f"ob_space = {env.observation_space}")
        print(f"ob = ")
        for k, v in ob.items():
            print(f"  {k}: {v.shape}")
            assert (
                v.shape == env.observation_space.spaces[k].shape
            ), f"{v.shape} != {env.observation_space.spaces[k].shape}"
        assert ob.keys() == env.observation_space.spaces.keys()
    else:
        print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
        assert env.observation_space.shape == ob.shape
    print(f"ac_space = {env.action_space.shape}")
    # print("observation:", ob)
    img = env.render()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("test_env_img.png", rgb_img)

    # Log directory
    data_path = "/home/jordon/humanoid-bench/vid_logs"
    log_dir = data_path + "/" + "basketball_test" + time.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    mylogger = logger.Logger(log_dir=log_dir)

    # Test video logging
    if args.log_video == "True":
        trajs = utils.rollout_n_trajectories(env, policy=None, ntraj=5, max_traj_length=5000, render=True)
        mylogger.log_trajs_as_videos(trajs, step=0, max_videos_to_save=2, fps=10, video_title="test_basketball")
    else:
        pass
    env.close()
