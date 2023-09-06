import sys
import logging
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig

import run_seed_fn
from helpers.utils import create_obs_config

import torch.multiprocessing as mp

from torch.multiprocessing import set_start_method, get_start_method


import os
import pickle
import gc
import logging
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from termcolor import colored

from rlbench import CameraConfig, ObservationConfig
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.runners.offline_train_runner_single_process import OfflineTrainRunnerSingleProcess
from yarr.utils.stat_accumulator import SimpleAccumulator

from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv


import warnings
warnings.filterwarnings("ignore")


def run(cfg: DictConfig,
             obs_config: ObservationConfig,
             cams,
             multi_task,
             seed) -> None:

    task = cfg.rlbench.tasks[0]
    tasks = cfg.rlbench.tasks


    replay_path = os.path.join(cfg.replay.path, 'seed%d' % seed)

    if cfg.method.name == 'GNFACTOR_BC':
        from agents import gnfactor_bc
        replay_buffer = gnfactor_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution, single_process=True, cfg=cfg)

        gnfactor_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, 0,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method)

        agent = gnfactor_bc.launch_utils.create_agent(cfg)
        
    elif cfg.method.name == 'PERACT_BC':
        replay_buffer = peract_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution, single_process=True)

        peract_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, 0,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method)

        agent = peract_bc.launch_utils.create_agent(cfg)

    
    else:
        raise ValueError('Method %s does not exists.' % cfg.method.name)

    wrapped_replay = PyTorchReplayBuffer(replay_buffer, num_workers=cfg.framework.num_workers)
    stat_accum = SimpleAccumulator(eval_video_fps=30)


    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, 'seed%d' % seed, 'weights')
    logdir = os.path.join(cwd, 'seed%d' % seed)

    rank = 0
    train_runner = OfflineTrainRunnerSingleProcess(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        train_device=rank,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        wandb_logger=None,
        world_size=0,
        cfg=cfg)

    # from viztracer import VizTracer
    # with VizTracer():
    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()

@hydra.main(config_name='config', config_path='conf')
def main(cfg: DictConfig) -> None:
    cfg_yaml = OmegaConf.to_yaml(cfg)
    # logging.info('\n' + cfg_yaml) # too much to print

    cfg.rlbench.cameras = cfg.rlbench.cameras \
        if isinstance(cfg.rlbench.cameras, ListConfig) else [cfg.rlbench.cameras]
    obs_config = create_obs_config(cfg.rlbench.cameras,
                                   cfg.rlbench.camera_resolution,
                                   cfg.method.name,
                                   use_nerf_multi_view=True)
    multi_task = len(cfg.rlbench.tasks) > 1

    cwd = os.getcwd()
    logging.info('CWD:' + os.getcwd())

    if cfg.framework.start_seed >= 0:
        # seed specified
        start_seed = cfg.framework.start_seed
    elif cfg.framework.start_seed == -1 and \
            len(list(filter(lambda x: 'seed' in x, os.listdir(cwd)))) > 0:
        # unspecified seed; use largest existing seed plus one
        largest_seed =  max([int(n.replace('seed', ''))
                             for n in list(filter(lambda x: 'seed' in x, os.listdir(cwd)))])
        start_seed = largest_seed + 1
    else:
        # start with seed 0
        start_seed = 0

    seed_folder = os.path.join(os.getcwd(), 'seed%d' % start_seed)
    os.makedirs(seed_folder, exist_ok=True)

    with open(os.path.join(seed_folder, 'config.yaml'), 'w') as f:
        f.write(cfg_yaml)

    # check if previous checkpoints already exceed the number of desired training iterations
    # if so, exit the script
    weights_folder = os.path.join(seed_folder, 'weights')
    if os.path.isdir(weights_folder) and len(os.listdir(weights_folder)) > 0:
        weights = os.listdir(weights_folder)
        latest_weight = sorted(map(int, weights))[-1]
        if latest_weight >= cfg.framework.training_iterations:
            logging.info('Agent was already trained for %d iterations. Exiting.' % latest_weight)
            sys.exit(0)

    # run train jobs with multiple seeds (sequentially)
    for seed in range(start_seed, start_seed + cfg.framework.seeds):
        logging.info('Starting seed %d.' % seed)

        world_size = cfg.ddp.num_devices
        run(cfg, obs_config, cfg.rlbench.cameras, multi_task, seed)


if __name__ == '__main__':
    main()
