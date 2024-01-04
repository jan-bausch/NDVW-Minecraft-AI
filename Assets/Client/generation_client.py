import asyncio
import csv
import random
import socket
import struct
import time

import ray
import torch

from model import Model
from all_models import get_model
from generating import GenerationInfo
from config import Config
from replaying import ReplayMemory, Transition

FIRST_PORT = 8080


def launch_generation_clients(
    config: Config,
    models: list[Model],
    generation_infos: list[GenerationInfo],
    replay_memory: ReplayMemory,
):
    clients = []

    for i in range(config.generation_servers):
        client = Client.remote(config, generation_infos[i], models[i].get_params(), models[i].get_state(), replay_memory)
        clients.append(client)
        time.sleep(1.0)

    for client in clients:
        client.run.remote()

    return clients


@ray.remote(num_gpus=0.05, num_cpus=0.5)
class Client:
    config: Config
    model: Model
    generation_infos: GenerationInfo
    socket: socket.socket
    seed: int
    replay_memory: ReplayMemory
    next_params: dict = None

    def __init__(
        self,
        config: Config,
        generation_infos: GenerationInfo,
        model_params: dict,
        model_state: dict,
        replay_memory: ReplayMemory
    ):
        print(f"Connecting to server {generation_infos.server_index}, {config.address}:{FIRST_PORT + generation_infos.server_index}")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(
            (config.address, FIRST_PORT + generation_infos.server_index)
        )
        print(f"Connected to server {generation_infos.server_index}")

        seed = (
            config.seed
            + generation_infos.current_episode
            + 100_000 * generation_infos.server_index
        )
        client_socket.sendall(
            struct.pack(
                "<ii",
                -1 * seed,
                config.parallel_worlds,
            )
        )

        self.config = config
        self.model = get_model(config)
        self.model.set_params(model_params)
        self.model.set_state(model_state)
        self.model.to_device(config.device)
        self.generation_infos = generation_infos
        self.socket = client_socket
        self.seed = seed
        self.replay_memory = replay_memory
    
    def update_model(self, params_temp_file: str):
        #print available devices
        print(f"Available devices: {torch.cuda.device_count()}. Current device: {torch.cuda.current_device()}. Taking device {self.config.device}")
        self.next_params = torch.load(params_temp_file, map_location=self.config.device)

    async def run(self):
        worlds_data = [[] for _ in range(self.config.parallel_worlds)]
        episode_changing_now = False
        trajectory_ids = await asyncio.gather(*[self.replay_memory.new_trajectory.remote() for _ in range(self.config.parallel_worlds)])

        replay_memory_pushs = []

        while True:
            await asyncio.sleep(0.01)

            data = bytearray()
            while len(data) < 4:
                packet = self.socket.recv(4 - len(data))
                if not packet:
                    return
                data.extend(packet)
            length = struct.unpack("<I", data)[0]

            if length == 0:
                continue

            data = bytearray()
            while len(data) < length:
                packet = self.socket.recv(length - len(data))
                if not packet:
                    print("Packet was not as large as planned")
                    print(data)
                    return
                data.extend(packet)

            world_id, reward_signal, *game_state = struct.unpack(
                f"<if{len(data)//4 - 2}i", data
            )

            if (
                not episode_changing_now
                and self.generation_infos.current_step
                % self.config.steps_per_episode
                == self.config.steps_per_episode - 1
            ):
                print(f"New episode for server {self.generation_infos.server_index}")
                
                if not self.generation_infos.validation:
                    print("Waiting for replay memory pushs")
                    await asyncio.gather(*replay_memory_pushs)
                    
                    print("Completing trajectories")
                    await asyncio.gather(*[
                        self.replay_memory.complete_trajectory.remote(trajectory_id) for trajectory_id in trajectory_ids
                    ])

                replay_memory_pushs = []
                
                self.generation_infos.current_step = 0
                self.generation_infos.current_episode += 1
                self.seed += 1

                worlds_data = [[] for _ in range(self.config.parallel_worlds)]
                
                print("Sending new episode instruction")
                self.socket.sendall(
                    struct.pack(
                        "<ii",
                        -1 * self.seed,
                        self.config.parallel_worlds,
                    )
                )

                self.model.reset_state()

                if not self.generation_infos.validation:
                    print("Waiting for new trajectories")
                    trajectory_ids = await asyncio.gather(*[self.replay_memory.new_trajectory.remote() for _ in range(self.config.parallel_worlds)])

                episode_changing_now = True
                continue

            worlds_data[world_id].append({"state": game_state, "reward": reward_signal})

            step_ready = True
            for world_id in range(self.config.parallel_worlds):
                if len(worlds_data[world_id]) == 0:
                    step_ready = False
                    break

            if step_ready:
                #print("step ready")

                worlds_states = [None for _ in range(self.config.parallel_worlds)]
                worlds_rewards = [None for _ in range(self.config.parallel_worlds)]
                for world_id in range(self.config.parallel_worlds):
                    worlds_states[world_id] = worlds_data[world_id][0]["state"]
                    worlds_rewards[world_id] = worlds_data[world_id][0]["reward"]
                    worlds_data[world_id].pop(0)

                actions = []

                if self.generation_infos.validation:
                    actions = self.model.inference(worlds_states)
                else:
                    actions = self.model.inference_train(worlds_states)

                    for world_id in range(self.config.parallel_worlds):
                        replay_memory_pushs.append(self.replay_memory.push.remote(
                            Transition(
                                worlds_states[world_id],
                                actions[world_id],
                                worlds_rewards[world_id],
                                abs(worlds_rewards[world_id]),
                            ),
                            trajectory_ids[world_id],
                        ))

                #print(random.sample(actions, min(10, len(actions))))

                #print("sending actions")
                for world_id in range(self.config.parallel_worlds):
                    self.socket.sendall(
                        struct.pack("<ii", world_id, actions[world_id])
                    )

                self.generation_infos.current_step += 1
                episode_changing_now = False

                if self.next_params is not None:
                    print("Applying model update")
                    self.model.set_params(self.next_params)
                    self.next_params = None

                validation_indicator =  "_validation" if self.generation_infos.validation else ""
                with open(f"generation_server_{self.generation_infos.server_index}{validation_indicator}_data.csv", mode="a") as file:
                    csv_writer = csv.writer(file)
                    file.seek(0, 1)

                    if file.tell() == 0:
                        csv_writer.writerow(
                            [
                                "episode",
                                "step",
                                "world_id",
                                "reward",
                                "action",
                            ]
                        )

                    for world_id in range(self.config.parallel_worlds):
                        csv_writer.writerow(
                            [
                                self.generation_infos.current_episode,
                                self.generation_infos.current_step,
                                world_id,
                                worlds_rewards[world_id],
                                actions[world_id],
                            ]
                        )

            # except Exception as e:
            #     print(f"Server: {self.generation_infos.server_index} Exception: {e}")


    def get_info(self):
        return self.generation_infos
    
    def get_model(self):
        torch.save(self.model.get_params(), "params_temp.pt")
        torch.save(self.model.get_state(), "state_temp.pt")
        return "params_temp.pt", "state_temp.pt"