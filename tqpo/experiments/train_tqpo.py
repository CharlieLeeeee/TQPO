import random
import numpy as np
import torch
import sys
import pprint

from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.projects.tqpo.experiments.minibatch_rl import MinibatchRlConstrained
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.utils.launching.affinity import affinity_from_code

from rlpyt.projects.tqpo.tqpo import tqpo
from rlpyt.projects.tqpo.tqpo_model import tqpoModel
from rlpyt.projects.tqpo.tqpo_agent import tqpoLstmAgent

from rlpyt.projects.tqpo.safety_gym_envs.safety_gym_env import safety_gym_make, SafetyGymTrajInfo

from rlpyt.projects.tqpo.experiments.config_tqpo import configs
from rlpyt.projects.tqpo.safety_gym_envs.config_safety_gym_env import config0
from rlpyt.projects.tqpo.safety_gym_envs.safety_gym_env import register_more_envs


def build(slot_affinity_code="0slt_0gpu_1cpu_1cpr",
          log_dir="test",
          run_ID="0",
          config_key="LSTM",
          variant_dir=None,
          ):
    seed = 100 + int(run_ID.split("_")[0])*100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    total_batch_size = config["sampler"]["batch_T"] * config["sampler"]["batch_B"]
    print("train_tqpo.py/build_and_train : total_batch_size : ", total_batch_size)
    print("train_tqpo.py/build_and_train : master_torch_threads : ", affinity["master_torch_threads"])

    new_batch_config = {"sampler": {
        "batch_T": int(total_batch_size // affinity["master_torch_threads"]),
        "batch_B": int(affinity["master_torch_threads"]),
    }}
    config = update_config(config, new_batch_config)

    print("rlpyt.projects.tqpo.experiments.train_tqpo.py | config")
    pprint.pprint(config)

    print("rlpyt.projects.tqpo.experiments.train_tqpo.py | affinity")
    pprint.pprint(affinity)

    register_more_envs([{'id': 'SimpleEnv-v0', 'config': config0}])

    sampler = CpuSampler(
        EnvCls=safety_gym_make,
        env_kwargs=config["env"],
        TrajInfoCls=SafetyGymTrajInfo,
        **config["sampler"]
    )

    algo = tqpo(**config["algo"])
    agent = tqpoLstmAgent(ModelCls=tqpoModel, model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRlConstrained(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        cost_limit=config["algo"]["cost_limit"],
        seed=seed,
        **config["runner"],
    )
    name = "tqpo_" + config["env"]["id"]
    return runner, log_dir, run_ID, name, config, agent


def build_and_train(
        slot_affinity_code="0slt_0gpu_1cpu_1cpr",
        log_dir="test",
        run_ID="0",
        config_key="LSTM",
        ):
    runner, log_dir, run_ID, name, config, _ = build(slot_affinity_code, log_dir, run_ID, config_key)
    with logger_context(log_dir, run_ID, name, config):
        runner.train()

import pickle

def save_args(args, filename='args.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(args, f)

def load_args(filename='args.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f"Command line arguments: {sys.argv}")
        save_args(sys.argv[1:])
        runner, log_dir, run_ID, name, config, _ = build(*sys.argv[1:])
    else:
        args = load_args()
        print(f"Loaded arguments: {args}")
        runner, log_dir, run_ID, name, config, _ = build(*args)
    with logger_context(log_dir, run_ID, name, config):
        runner.train()




