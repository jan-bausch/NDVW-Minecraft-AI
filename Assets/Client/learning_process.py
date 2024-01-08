import asyncio
import time
import ray
import torch

from checkpointing import checkpoint
from replaying import ReplayMemory, ReplayingInfo, Transition
from all_models import get_model
from model import Model
from config import Config
from generation_client import Client


@ray.remote(num_gpus=0.5, num_cpus=1)
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
        self.model = get_model(config)
        self.model.set_params(model_params)
        self.model.set_state(model_state)
        self.model.to_device(config.device)
        self.generation_processes = generation_processes
        self.replay_memory = replay_memory
        self.replay_info = replay_info

    async def run(self):
        while True:
            await self.train_trajectory()
            self.model.reset_state()
            self.replay_info.current_step = 0
            self.replay_info.current_trajectory_indexes = []

    async def train_trajectory(self):
        trajectories = []
        trajectories_indices = []
        if len(self.replay_info.current_trajectory_indexes) == 0:
            trajectories, trajectories_indices = await self.replay_memory.sample_trajectories.remote(
                self.config.batch_size
            )
            self.replay_info.current_trajectory_indexes = [
                i for i in range(len(trajectories))
            ]
        else:
            trajectories_indices = self.replay_info.current_trajectory_indexes
            trajectories = await self.replay_memory.get_trajectories.remote(
                trajectories_indices
            )

        self.replay_info.current_trajectory_indexes = trajectories_indices

        if len(trajectories) == 0:
            print("No complete trajectories yet")
            time.sleep(60)
            return

        print(f"New replay trajectories.\nTrajectories count: {len(trajectories)}")
        
        #for i, t in enumerate(trajectories):
        #    print(f"Trajectory {i} length: {len(t)}")

        while self.replay_info.current_step < len(trajectories[0]):
            print(
                f"Step {self.replay_info.current_step} of trajectory {self.replay_info.current_trajectory_indexes[0]}"
            )
            self.train_step(trajectories)
            self.replay_info.current_step += 1
            self.steps += 1

            if self.steps % 200 == 0:
                print("Checkpoint time")

                generation_infos = await asyncio.gather(*
                    [
                        generation_process.get_info.remote()
                        for generation_process in self.generation_processes
                    ]
                )
                generation_models_params_states = await asyncio.gather(*
                    [
                        generation_process.get_model.remote()
                        for generation_process in self.generation_processes
                    ]
                )
                generation_models_params = []
                generation_models_states = []
                for i in range(self.config.generation_servers):
                    generation_models_params.append(
                        torch.load(generation_models_params_states[i][0], map_location=torch.device("cpu"))
                    )
                    generation_models_states.append(
                        torch.load(generation_models_params_states[i][1], map_location=torch.device("cpu"))
                    )

                checkpoint(
                    self.config,
                    generation_infos,
                    generation_models_params,
                    generation_models_states,
                    self.replay_info,
                    self.model.get_params(),
                    self.model.get_state(),
                    self.replay_memory,
                )

            if self.steps % 25 == 15:
                print("Params update time")

                torch.save(self.model.get_params(), "params_temp.pt")
                for client in self.generation_processes:
                    client.update_model.remote("params_temp.pt")

    def train_step(self, trajectories: list[list[Transition]]):
        #states_sizes = [len(trajectories[i]) for i in range(len(trajectories))]
        #print(f"States sizes: {states_sizes}")
        # print(f"Min states size: {min(states_sizes)}")
        # print(f"Max states size: {max(states_sizes)}")

        try:
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
        except Exception as e:
            print(f'Error in train_step: {self.replay_info.current_step}')
            print(e)
