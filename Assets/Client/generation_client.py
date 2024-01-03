import csv
import random
import socket
import struct

import ray

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
        info = generation_infos[i]
        model = models[i]
        client = Client.remote(config, info, model.get_params(), model.get_state(), replay_memory)
        clients.append(client)

    for client in clients:
        client.run.remote()

    return clients


@ray.remote
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
        self.model = get_model(config.model)
        self.model.set_params(model_params)
        self.model.set_state(model_state)
        self.generation_infos = generation_infos
        self.socket = client_socket
        self.seed = seed
        self.replay_memory = replay_memory
    
    def update_model(self, model_params: dict):
        self.next_params = model_params

    def run(self):
        worlds_data = []
        episode_changing_now = False
        trajectory_ids = [self.replay_memory.new_trajectory.remote() for _ in range(self.config.parallel_worlds)]

        while True:
            try:
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
                    for trajectory_id in trajectory_ids:
                        self.replay_memory.complete_trajectory.remote(trajectory_id)
                    
                    self.generation_infos.current_step += 1
                    self.seed += 1

                    worlds_data = []
                    self.socket.sendall(
                        struct.pack(
                            "<ii",
                            -1 * self.seed,
                            self.config.parallel_worlds,
                        )
                    )
                    print(f"New episode for server {self.generation_infos.server_index}")
                    
                    if self.next_params is not None:
                        self.model.set_params(self.next_params)
                        self.next_params = None

                    self.model.reset_state()

                    trajectory_ids = [self.replay_memory.new_trajectory.remote() for _ in range(self.config.parallel_worlds)]

                    episode_changing_now = True
                    continue

                worlds_data.append({"state": game_state, "reward": reward_signal})

                step_ready = True
                for world_id in range(self.config.parallel_worlds):
                    if len(worlds_data[world_id]) == 0:
                        step_ready = False
                        break

                if step_ready:
                    print("step ready")

                    worlds_states = [None for _ in range(self.config.parallel_worlds)]
                    worlds_rewards = [None for _ in range(self.config.parallel_worlds)]
                    for world_id in range(self.config.parallel_worlds):
                        worlds_states[world_id] = worlds_data[world_id][0]["state"]
                        worlds_rewards[world_id] = worlds_data[world_id][0]["reward"]
                        worlds_data[world_id].pop(0)

                    actions = self.model.inference(worlds_states)

                    for world_id in range(self.config.parallel_worlds):
                        self.replay_memory.push.remote(
                            Transition(
                                worlds_states[world_id],
                                worlds_rewards[world_id],
                                actions[world_id],
                                1.0,
                            ),
                            trajectory_ids[world_id],
                        )

                    print(random.sample(actions, min(10, len(actions))))

                    print("sending actions")
                    for world_id in range(self.config.parallel_worlds):
                        socket.sendall(
                            struct.pack("<ii", world_id, actions[world_id])
                        )

                    self.generation_infos.current_step += 1
                    episode_changing_now = False

                    with open(f"generation_server_{self.generation_infos.server_index}_data.csv", mode="a") as file:
                        csv_writer = csv.writer(file)
                        file.seek(0, 2)

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

            except Exception as e:
                print(f"Server: {self.generation_infos.server_index} Exception: {e}")


    def get_info(self):
        return self.replay_info
    
    def get_model(self):
        return self.model.get_params(), self.model.get_state()