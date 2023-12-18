import socket
import struct
import random
import torch
import torch.nn as nn
import argparse
import csv
import time
import copy

# TCP server address and port
SERVER_PORT = 8080

SEED = 3
PARALLEL_EPISODES = 64
PARALLEL_SERVERS = 4
TRAIN_VALID_SPLIT = 0.8
TRAIN_EPISODES = int(PARALLEL_EPISODES * TRAIN_VALID_SPLIT)
VALID_EPISODES = PARALLEL_EPISODES - TRAIN_EPISODES
STEPS_PER_EPISODE = 300

servers_lstm_hidden = [None for i in range(PARALLEL_SERVERS)]
target_servers_lstm_hidden = [None for i in range(PARALLEL_SERVERS)]


class DQN(nn.Module):
    frame_dim: tuple[int, int]
    frame_channels: int
    other_dim: int

    def __init__(self, frame_dim, frame_channels, other_dim, output_dim):
        super(DQN, self).__init__()
        self.frame_dim = frame_dim
        self.frame_channels = frame_channels
        self.other_dim = other_dim
        self.conv1 = nn.Conv2d(
            in_channels=frame_channels, out_channels=16, kernel_size=8, stride=4
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(self.calculate_conv_output_dim() + other_dim, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.lstm_num_layers = 1
        self.lstm_hidden_dim = 64
        self.lstm = nn.LSTM(
            input_size=128,
            num_layers=self.lstm_num_layers,
            hidden_size=self.lstm_hidden_dim,
            batch_first=True,
        )
        self.fc3 = nn.Linear(64, output_dim)
        self.init_hidden(1)

    def init_hidden(self, batch_size):
        self.lstm_hidden = (
            torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim),
            torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim),
        )

    def reset_lstm(self, batch_size):
        self.init_hidden(batch_size)

    def get_lstm_hidden(self):
        return (
            self.lstm_hidden[0].detach().clone(),
            self.lstm_hidden[1].detach().clone(),
        )

    def set_lstm_hidden(self, hidden):
        self.lstm_hidden = (
            hidden[0].detach().clone(),
            hidden[1].detach().clone(),
        )

    def calculate_conv_output_dim(self):
        # Calculate the output dimension after convolution layers
        x = torch.zeros(1, self.frame_channels, *self.frame_dim)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return int(torch.prod(torch.tensor(x.size())))

    def forward(self, x: torch.Tensor):
        self.lstm.flatten_parameters()
        x_other = x[:, : self.other_dim]
        x_frame = x[:, self.other_dim :]
        x = x_frame.view(
            x.size(0), self.frame_channels, self.frame_dim[0], self.frame_dim[1]
        )  # Reshape input to (batch_size, channels, height, width)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = torch.cat((x_other, x), dim=1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = x.unsqueeze(1)
        lstm_out, self.lstm_hidden = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
FRAME_DIM = (64, 64)
OTHER_DIM = 2
FRAME_CHANNELS = 3
OUTPUT_DIM = 11

model = DQN(FRAME_DIM, FRAME_CHANNELS, OTHER_DIM, OUTPUT_DIM)
model.to(device)
target_model = copy.deepcopy(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

BASE_EPSILON = 1
MIN_EPSILON = 0.1
BATCH_SIZE = PARALLEL_EPISODES  # no memory replay, learning directly from stream
EXPLORATORY_FRAMES = 4_000_000
N_STEPS = 10
MAX_TARGET_INDEPENDENCE_STEPS = 30
MAX_TARGET_INDEPENDENCE_FRAMES = (
    BATCH_SIZE * PARALLEL_SERVERS * MAX_TARGET_INDEPENDENCE_STEPS
)
target_independence_frames = 0

previous_data = [
    {
        "states": [],
        "actions": [],
        "rewards": [],
    }
    for _ in range(PARALLEL_SERVERS)
]

previous_episodes = [0 for _ in range(PARALLEL_SERVERS)]

total_q_values = 0
total_rewards = 0
steps_counters = [0 for _ in range(PARALLEL_SERVERS)]

episodes_counters = [0 for _ in range(PARALLEL_SERVERS)]

trained_frames_count = 0


# Function to save model weights into a 'weights' directory
def save_weights(model, frames, epoch):
    weights_path = (
        f"training_data/weights/model_weights_frames_{frames}_epoch_{epoch}.pt"
    )
    torch.save(model.state_dict(), weights_path)


# Function to load model weights
def load_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path))


last_times = [time.time()]


def get_calls_per_second():
    global last_times

    if len(last_times) > 10:
        last_times.pop(0)
    last_times.append(time.time())
    total_time = last_times[-1] - last_times[0]
    total_calls = len(last_times) - 1
    return total_calls / total_time


def get_previous_states(server_index, t):
    previous_states = previous_data[server_index]["states"]
    if len(previous_states) < t:
        return None
    return previous_states[-t]


def get_previous_action(server_index, t):
    previous_actions = previous_data[server_index]["actions"]
    if len(previous_actions) < t:
        return None
    return previous_actions[-t]


def get_previous_reward(server_index, t):
    previous_rewards = previous_data[server_index]["rewards"]
    if len(previous_rewards) < t:
        return None
    return previous_rewards[-t]


def push_previous_data(server_index, state, action, reward):
    previous_data[server_index]["states"].append(state)
    previous_data[server_index]["actions"].append(action)
    previous_data[server_index]["rewards"].append(reward)
    if len(previous_data[server_index]["states"]) > N_STEPS:
        previous_data[server_index]["states"].pop(0)
        previous_data[server_index]["actions"].pop(0)
        previous_data[server_index]["rewards"].pop(0)


def neural_model_no_training(server_index, worlds_states):
    global last_times
    global previous_episodes

    worlds_states_tensor = torch.tensor(
        worlds_states, dtype=torch.float32, device=device
    )

    calls_per_sec = get_calls_per_second()
    print(f"Calls per second: {calls_per_sec}")

    if previous_episodes[server_index] < episodes_counters[server_index]:
        previous_episodes[server_index] = episodes_counters[server_index]
        model.reset_lstm(BATCH_SIZE)
        target_model.reset_lstm(BATCH_SIZE)

    with torch.no_grad():
        actions_values = model(worlds_states_tensor)
        variances = torch.var(actions_values, dim=0)
        print(f"Q values min variance: {variances.min().item()}")

        actions = [0 for _ in range(PARALLEL_EPISODES)]
        for i in range(PARALLEL_EPISODES):
            actions[i] = torch.argmax(actions_values[i]).item()

    return actions


def neural_model(server_index, worlds_states, worlds_rewards):
    global previous_data
    global last_times
    global target_model
    global trained_frames_count
    global target_independence_frames
    global previous_episodes

    print(f"Step: {steps_counters[server_index]}")

    calls_per_sec = get_calls_per_second()
    print(f"Calls per second: {calls_per_sec}")

    worlds_states_tensor = torch.tensor(
        worlds_states, dtype=torch.float32, device=device
    )

    if previous_episodes[server_index] < episodes_counters[server_index]:
        previous_episodes[server_index] = episodes_counters[server_index]
        model.reset_lstm(BATCH_SIZE)
        target_model.reset_lstm(BATCH_SIZE)

        if server_index == 0:
            save_weights(model, trained_frames_count, sum(steps_counters))

    loss_value = None
    prediction_actions_values = None

    previous_states = get_previous_states(server_index, N_STEPS)
    previous_actions = get_previous_action(server_index, N_STEPS)
    previous_rewards_n_steps = [
        get_previous_reward(server_index, i) for i in range(1, N_STEPS + 1)
    ][::-1]

    if previous_states is not None and previous_actions is not None:
        targets = torch.tensor(
            [0 for _ in range(BATCH_SIZE)], dtype=torch.float32, device=device
        )
        predictions = torch.tensor(
            [0 for _ in range(BATCH_SIZE)], dtype=torch.float32, device=device
        )
        gamma = 0.99
        previous_worlds_states_tensor = torch.tensor(
            previous_states, dtype=torch.float32, device=device
        )
        prediction_actions_values = model(previous_worlds_states_tensor)
        target_actions_values = target_model(worlds_states_tensor)

        variances = torch.var(target_actions_values, dim=0)
        print(f"Q* values min variance: {variances.min().item()}")
        variances = torch.var(prediction_actions_values, dim=0)
        print(f"Q values min variance: {variances.min().item()}")

        for i in range(BATCH_SIZE):
            target_action = torch.argmax(target_actions_values[i]).item()
            target_action_value = target_actions_values[i][target_action]
            action = previous_actions[i]
            predictions[i] = prediction_actions_values[i][action]
            targets[i] = worlds_rewards[i]
            for j in range(N_STEPS):
                targets[i] += previous_rewards_n_steps[j][i] * (gamma ** (j + 1))
            targets[i] += target_action_value * (gamma ** (N_STEPS + 1))

        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()

        trained_frames_count += BATCH_SIZE
        target_independence_frames += BATCH_SIZE

        print(f"Trained frames: {trained_frames_count}")
        if target_independence_frames >= MAX_TARGET_INDEPENDENCE_FRAMES:
            print("Updating target model")
            target_independence_frames = 0
            target_model = DQN(FRAME_DIM, FRAME_CHANNELS, OTHER_DIM, OUTPUT_DIM)
            target_model.load_state_dict(model.state_dict())
            target_model.to(device)

    actions_values = prediction_actions_values
    if actions_values is None:
        actions_values = model(worlds_states_tensor)

    actions = [None for _ in range(PARALLEL_EPISODES)]
    exploration_progress = min(1, trained_frames_count / EXPLORATORY_FRAMES)
    print(f"Exploration progress: {exploration_progress}")
    epsilon = BASE_EPSILON - exploration_progress * (BASE_EPSILON - MIN_EPSILON)
    for i in range(TRAIN_EPISODES):
        if random.random() < epsilon:
            actions[i] = random.randint(0, OUTPUT_DIM - 1)
        else:
            actions[i] = torch.argmax(actions_values[i]).item()
    for i in range(TRAIN_EPISODES, PARALLEL_EPISODES):
        actions[i] = torch.argmax(actions_values[i]).item()

    push_previous_data(server_index, worlds_states, actions, worlds_rewards)

    with open(args.csv_filename, mode="a") as file:
        avg_train_q_value = torch.mean(actions_values[:TRAIN_EPISODES]).item()
        avg_q_value = torch.mean(actions_values[TRAIN_EPISODES:]).item()
        avg_train_reward = sum(worlds_rewards[:TRAIN_EPISODES]) / TRAIN_EPISODES
        avg_reward = sum(worlds_rewards[TRAIN_EPISODES:]) / VALID_EPISODES
        actions_tensor = torch.tensor(
            actions[TRAIN_EPISODES:], dtype=torch.float32, device=device
        )
        actions_variance = torch.var(actions_tensor).item()

        csv_writer = csv.writer(file)
        file.seek(0, 2)  # Move to the end of the file
        if file.tell() == 0:  # Check if the file is empty
            csv_writer.writerow(
                [
                    "Epoch",
                    "Server",
                    "Average_Q_Value",
                    "Average_Train_Q_Value",
                    "Average_Reward",
                    "Average_Train_Reward",
                    "Loss",
                    "Calls_per_sec",
                    "Actions_variance",
                ]
            )
        csv_writer.writerow(
            [
                steps_counters[server_index],
                server_index,
                avg_q_value,
                avg_train_q_value,
                avg_reward,
                avg_train_reward,
                loss_value,
                calls_per_sec,
                actions_variance,
            ]
        )
        print(
            f"Average Q-value: {avg_q_value}\nAverage Train Q-value: {avg_train_q_value}\nAverage reward: {avg_reward}\nAverage Train reward: {avg_train_reward}\nLoss: {loss_value}"
        )

    if trained_frames_count % (30 * BATCH_SIZE) == 0 and trained_frames_count > 0:
        save_weights(model, trained_frames_count, sum(steps_counters))

    return actions


def receive_frame_data():
    global steps_counters
    global episodes_counters

    client_sockets = []

    for i in range(PARALLEL_SERVERS):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((args.address, SERVER_PORT + i))
        client_sockets.append(client_socket)
        print(f"Connected to server {i}")
        client_socket.sendall(
            struct.pack(
                "<ii",
                -1 * (SEED + (100000 * i) + episodes_counters[i]),
                PARALLEL_EPISODES,
            )
        )

    worlds_data = [
        [[] for _ in range(PARALLEL_EPISODES)] for _ in range(PARALLEL_SERVERS)
    ]

    episode_changing_now = [False for _ in range(PARALLEL_SERVERS)]

    server_index = 0
    while True:
        try:
            client_socket = client_sockets[server_index]

            # Receive data from the server
            data = bytearray()
            while len(data) < 4:  # Read 4 bytes for the data length
                packet = client_socket.recv(4 - len(data))
                # print("Packet received")
                if not packet:
                    return
                data.extend(packet)
            length = struct.unpack("<I", data)[0]

            # print(f"Message length {length}")

            if length == 0:
                continue

            # Read the actual data
            data = bytearray()
            while len(data) < length:
                packet = client_socket.recv(length - len(data))
                if not packet:
                    print("Packet was not as large as planned")
                    print(data)
                    return
                data.extend(packet)

            # Unpack received binary data
            world_id, reward_signal, *game_state = struct.unpack(
                f"<if{len(data)//4 - 2}i", data
            )

            if (
                not episode_changing_now[server_index]
                and steps_counters[server_index] % STEPS_PER_EPISODE
                == STEPS_PER_EPISODE - 1
            ):
                episodes_counters[server_index] += 1

                worlds_data[server_index] = [[] for _ in range(PARALLEL_EPISODES)]
                client_socket.sendall(
                    struct.pack(
                        "<ii",
                        -1 * (SEED + episodes_counters[server_index]),
                        PARALLEL_EPISODES,
                    )
                )
                print(f"New episode for server {server_index}")
                episode_changing_now[server_index] = True
                continue

            worlds_data[server_index][world_id].append({"state": game_state, "reward": reward_signal})

            step_ready = True
            for world_id in range(PARALLEL_EPISODES):
                if len(worlds_data[server_index][world_id]) == 0:
                    step_ready = False
                    break

            if step_ready:
                print("step ready")
                if servers_lstm_hidden[server_index] is None:
                    model.init_hidden(1)
                    servers_lstm_hidden[server_index] = model.get_lstm_hidden()
                    target_model.init_hidden(1)
                    target_servers_lstm_hidden[
                        server_index
                    ] = target_model.get_lstm_hidden()

                lstm_hidden = servers_lstm_hidden[server_index]
                model.set_lstm_hidden(lstm_hidden)
                target_lstm_hidden = target_servers_lstm_hidden[server_index]
                target_model.set_lstm_hidden(target_lstm_hidden)

                worlds_states = [None for _ in range(PARALLEL_EPISODES)]
                worlds_rewards = [None for _ in range(PARALLEL_EPISODES)]
                for world_id in range(PARALLEL_EPISODES):
                    worlds_states[world_id] = worlds_data[server_index][world_id][0]["state"]
                    worlds_rewards[world_id] = worlds_data[server_index][world_id][0]["reward"]
                    worlds_data[server_index][world_id].pop(0)

                actions = [None for _ in range(PARALLEL_EPISODES)]
                if args.no_train:
                    actions = neural_model_no_training(server_index, worlds_states)
                else:
                    actions = neural_model(server_index, worlds_states, worlds_rewards)

                print(random.sample(actions, min(10, len(actions))))

                print("sending actions")
                for world_id in range(PARALLEL_EPISODES):
                    client_socket.sendall(
                        struct.pack("<ii", world_id, actions[world_id])
                    )

                steps_counters[server_index] += 1
                episode_changing_now[server_index] = False
                servers_lstm_hidden[server_index] = model.get_lstm_hidden()
                target_servers_lstm_hidden[
                    server_index
                ] = target_model.get_lstm_hidden()

                if PARALLEL_SERVERS > 3:
                    new_server_index = random.randint(0, PARALLEL_SERVERS - 1)
                    if new_server_index == server_index:
                        new_server_index = (new_server_index + 1) % PARALLEL_SERVERS
                    print(f"Server {server_index} -> {new_server_index}")
                    server_index = new_server_index
                else:
                    new_server_index = (server_index + 1) % PARALLEL_SERVERS
                    print(f"Server {server_index} -> {new_server_index}")
                    server_index = new_server_index

        except ConnectionResetError as e:
            print(f"Connection was reset: {e}")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((args.address, SERVER_PORT + i))
            client_sockets[server_index] = client_socket
            print(f"Reconnected to server {i}")
            client_socket.sendall(
                struct.pack(
                    "<ii",
                    -1 * (SEED + (100000 * i) + episodes_counters[server_index]),
                    PARALLEL_EPISODES,
                )
            )

    for client_socket in client_sockets:
        client_socket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optional arguments to load model weights and specify CSV file name."
    )
    parser.add_argument(
        "-w", "--load_weights", type=str, help="Path to model weights file"
    )
    parser.add_argument(
        "-a", "--address", type=str, help="Server address", default="127.0.0.1"
    )
    parser.add_argument(
        "-c",
        "--csv_filename",
        type=str,
        default="averages.csv",
        help="Path to CSV file to store averages",
    )
    parser.add_argument(
        "-n",
        "--no_train",
        action="store_true",
        help="Whether to disable learning",
    )
    args = parser.parse_args()

    if args.load_weights:
        # Load weights if provided through command line arguments
        snapshot_epochs = int(args.load_weights.split("_")[-1][:-3])
        steps_counters = [snapshot_epochs // PARALLEL_SERVERS for _ in range(PARALLEL_SERVERS)]
        trained_frames_count = int(
            (snapshot_epochs - (BATCH_SIZE / PARALLEL_EPISODES)) * BATCH_SIZE
        )
        episodes_counters = [
            int(steps_counters[i] / STEPS_PER_EPISODE) for i in range(PARALLEL_SERVERS)
        ]
        previous_episodes = copy.deepcopy(episodes_counters)
        print(
            f"Loading weights from epoch {snapshot_epochs}, episode {episodes_counters[0]}, trained frames {trained_frames_count}"
        )
        load_weights(model, args.load_weights)
        target_model = copy.deepcopy(model)

    receive_frame_data()
