import time
import ray

from checkpointing import checkpoint
from replaying import ReplayMemory, ReplayingInfo, Transition
from all_models import get_model
from model import Model
from config import Config
from generation_client import Client


@ray.remote
class LearningProcess:
    config: Config
    model: Model
    generation_processes: list[Client]
    replay_memory: ReplayMemory
    replay_info: ReplayingInfo
    steps = 0

    def __init__(
        self,
        config: Config,
        model_params: dict,
        model_state: dict,
        generation_processes: list[Client],
        replay_memory: ReplayMemory,
        replay_info: ReplayingInfo,
    ):
        self.config = config
        self.model = get_model(config.model)
        self.model.set_params(model_params)
        self.model.set_state(model_state)
        self.generation_processes = generation_processes
        self.replay_memory = replay_memory
        self.replay_info = replay_info

    def run(self):
        while True:
            self.train_trajectory()
            self.replay_info.current_step = 0

    def train_trajectory(self):
        trajectories = []
        if len(self.replay_info.current_trajectory_indexes) == 0:
            trajectories = self.replay_memory.sample_trajectories(
                self.config.batch_size
            )
            self.replay_info.current_trajectory_indexes = [
                i for i in range(len(trajectories))
            ]
        else:
            trajectories = self.replay_memory.get_trajectories(
                self.replay_info.current_trajectory_indexes
            )

        if len(trajectories) == 0:
            print("No complete trajectories yet")
            time.sleep(1)
            return

        while self.replay_info.current_step < len(trajectories[0]):
            print(
                f"Step {self.replay_info.current_step} of trajectory {self.replay_info.current_trajectory_indexes[0]}"
            )
            self.train_step(trajectories)
            self.replay_info.current_step += 1
            self.steps += 1

            if self.steps % 100 == 0:
                print("Checkpoint time")

                generation_infos = ray.get(
                    [
                        generation_process.get_info.remote()
                        for generation_process in self.generation_processes
                    ]
                )
                generation_models_params_states = ray.get(
                    [
                        generation_process.get_model.remote()
                        for generation_process in self.generation_processes
                    ]
                )
                generation_models = [
                    get_model(self.config.model)
                    for _ in range(self.config.generation_servers)
                ]
                for i in range(self.config.generation_servers):
                    generation_models[i].set_params(
                        generation_models_params_states[i][0]
                    )
                    generation_models[i].set_state(
                        generation_models_params_states[i][1]
                    )

                checkpoint(
                    self.config,
                    generation_infos,
                    generation_models,
                    self.model,
                    self.replay_info,
                    self.replay_memory,
                )

            if self.steps % 100 == 0:
                print("Params update time")

                for client in self.generation_processes:
                    client.update_params.remote(self.model.get_params())

    def train_step(self, trajectories: list[list[Transition]]):
        states = [
            trajectories[i][self.replay_info.current_step].state
            for i in range(len(trajectories))
        ]
        actions = [
            trajectories[i][self.replay_info.current_step].action
            for i in range(len(trajectories))
        ]
        rewards = [
            trajectories[i][self.replay_info.current_step].reward
            for i in range(len(trajectories))
        ]

        self.model.train(states, actions, rewards)
