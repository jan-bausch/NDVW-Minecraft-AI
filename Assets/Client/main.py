import sys

import ray
from learning_process import LearningProcess
from generating import GenerationInfo
from replaying import ReplayMemory, ReplayingInfo
from all_models import get_model
from generation_client import launch_generation_clients
from config import Config
from checkpointing import load_checkpoint


if __name__ == "__main__":
    ray.init()
    path = sys.argv[1]

    config: Config = None
    generation_infos = None
    generation_models = None
    learner_infos = None
    learner_model = None
    replay_memory = None

    if path.startswith("checkpoints/"):
        (
            config,
            generation_infos,
            generation_models,
            learner_infos,
            learner_model,
            replay_memory,
        ) = load_checkpoint(path)
    else:
        config = Config.parse(path)
        print(vars(config))

        generation_infos = [
            GenerationInfo(
                i, 0, 0, i >= config.generation_servers - config.validation_servers
            )
            for i in range(config.generation_servers)
        ]
        learner_infos = ReplayingInfo([], 0)
        learner_model = get_model(config)
        generation_models = [
            get_model(config) for _ in range(config.generation_servers)
        ]
        for i in range(config.generation_servers):
            generation_models[i].set_seed(config.seed + 1 + i)

    if replay_memory is None:
        learner_infos.current_trajectory_indexes = []
        learner_infos.current_step = 0
        replay_memory = ReplayMemory.remote(config)

    clients = launch_generation_clients(
        config, generation_models, generation_infos, replay_memory
    )

    learning_process = LearningProcess.remote(
        config,
        learner_model.get_params(),
        learner_model.get_state(),
        clients,
        replay_memory,
        learner_infos,
    )

    learning_process.run.remote()

    ray.get([learning_process.run.remote()])
