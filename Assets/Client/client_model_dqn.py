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
PARALLEL_EPISODES = 16
TRAIN_VALID_SPLIT = 0.8
TRAIN_EPISODES = int(PARALLEL_EPISODES * TRAIN_VALID_SPLIT)
VALID_EPISODES = PARALLEL_EPISODES - TRAIN_EPISODES
STEPS_PER_EPISODE = 300


class DQN(nn.Module):
    frame_dim: tuple[int, int]
    frame_channels: int
    other_dim: int

    def __init__(self, frame_dim, frame_channels, other_dim, output_dim):
        super(DQN, self).__init__()
        self.frame_dim = frame_dim
        self.frame_channels = frame_channels
        self.other_dim = other_dim
        self.conv1 = nn.Conv2d(in_channels=frame_channels, out_channels=16, kernel_size=8, stride=4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(self.calculate_conv_output_dim() + other_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def calculate_conv_output_dim(self):
        # Calculate the output dimension after convolution layers
        x = torch.zeros(1, self.frame_channels, *self.frame_dim)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return int(torch.prod(torch.tensor(x.size())))

    def forward(self, x: torch.Tensor):
        x_other = x[:, :self.other_dim]
        x_frame = x[:, self.other_dim:]
        x = x_frame.view(
            x.size(0), self.frame_channels, self.frame_dim[0], self.frame_dim[1]
        )  # Reshape input to (batch_size, channels, height, width)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = torch.cat((x_other, x), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
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
BATCH_SIZE = PARALLEL_EPISODES # no memory replay, learning directly from stream
EXPLORATORY_FRAMES = 1_000_000
MAX_TARGET_INDEPENDENCE_FRAMES = 128
target_independence_frames = 0

previous_states = None
previous_actions = None

total_q_values = 0
total_rewards = 0
steps_counter = 0
episodes_counter = 0
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


last_times = [time.time() for _ in range(100)]


def neural_model(worlds_states, worlds_rewards):
    global previous_states
    global previous_actions
    global last_times
    global target_model
    global trained_frames_count
    global target_independence_frames

    print(f"Step: {steps_counter}")

    last_times.pop(0)
    last_times.append(time.time())
    total_time = last_times[-1] - last_times[0]
    total_calls = len(last_times) - 1
    print(f"Calls per second: {total_calls / total_time}")

    worlds_states_tensor = torch.tensor(
        worlds_states, dtype=torch.float32, device=device
    )

    loss_value = None
    prediction_actions_values = None
    if previous_states is not None and previous_actions is not None:
        targets = torch.tensor(
            [0 for _ in range(BATCH_SIZE)], dtype=torch.float32, device=device
        )
        predictions = torch.tensor(
            [0 for _ in range(BATCH_SIZE)], dtype=torch.float32, device=device
        )
        gamma = 0.95
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
            targets[i] = worlds_rewards[i] + gamma * target_action_value

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
            target_model = copy.deepcopy(model)

    actions_values = prediction_actions_values
    if actions_values is None:
        actions_values = model(worlds_states_tensor)
    
    actions = [None for _ in range(PARALLEL_EPISODES)]
    exploration_progress = min(1, trained_frames_count / EXPLORATORY_FRAMES)
    print(f"Exploration progress: {exploration_progress}")
    epsilon = BASE_EPSILON - exploration_progress * (BASE_EPSILON - MIN_EPSILON)
    for i in range(TRAIN_EPISODES):
        if not args.no_train and random.random() < epsilon:
            actions[i] = random.randint(0, OUTPUT_DIM-1)
        else:
            actions[i] = torch.argmax(actions_values[i]).item()
    for i in range(TRAIN_EPISODES, PARALLEL_EPISODES):
        actions[i] = torch.argmax(actions_values[i]).item()

    previous_states = copy.deepcopy(worlds_states)
    previous_actions = copy.deepcopy(actions)

    if args.no_train:
        print(actions)
        return actions

    with open(args.csv_filename, mode="a") as file:
        avg_train_q_value = torch.mean(actions_values[:TRAIN_EPISODES]).item()
        avg_q_value = torch.mean(actions_values[TRAIN_EPISODES:]).item()
        avg_train_reward = sum(worlds_rewards[:TRAIN_EPISODES]) / TRAIN_EPISODES
        avg_reward = sum(worlds_rewards[TRAIN_EPISODES:]) / VALID_EPISODES

        csv_writer = csv.writer(file)
        file.seek(0, 2)  # Move to the end of the file
        if file.tell() == 0:  # Check if the file is empty
            csv_writer.writerow(
                ["Epoch", "Average_Q_Value", "Average_Train_Q_Value", "Average_Reward", "Average_Train_Reward", "Loss", "Calls_per_sec"]
            )
        csv_writer.writerow(
            [
                steps_counter,
                avg_q_value,
                avg_train_q_value,
                avg_reward,
                avg_train_reward,
                loss_value,
                total_calls / total_time,
            ]
        )
        print(
            f"Average Q-value: {avg_q_value}\nAverage Train Q-value: {avg_train_q_value}\nAverage reward: {avg_reward}\nAverage Train reward: {avg_train_reward}\nLoss: {loss_value}"
        )

    if trained_frames_count % 50 * BATCH_SIZE == 0 and trained_frames_count > 0:
        save_weights(model, trained_frames_count, steps_counter)

    return actions


def receive_frame_data():
    global steps_counter
    global episodes_counter

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.address, SERVER_PORT))

    print("Connected")

    worlds_data = [[] for _ in range(PARALLEL_EPISODES)]
    client_socket.sendall(
        struct.pack("<ii", -1 * (SEED + episodes_counter), PARALLEL_EPISODES)
    )

    episode_changing_now = False

    try:
        while True:
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
                not episode_changing_now
                and steps_counter % STEPS_PER_EPISODE == STEPS_PER_EPISODE - 1
            ):
                episodes_counter += 1

                worlds_data = [[] for _ in range(PARALLEL_EPISODES)]
                client_socket.sendall(
                    struct.pack(
                        "<ii", -1 * (SEED + episodes_counter), PARALLEL_EPISODES
                    )
                )
                print("New episode")
                episode_changing_now = True
                continue

            worlds_data[world_id].append(
                {"state": game_state, "reward": reward_signal}
            )

            step_ready = True
            for world_id in range(PARALLEL_EPISODES):
                if len(worlds_data[world_id]) == 0:
                    step_ready = False
                    break

            if step_ready:
                print("step ready")
                worlds_states = [None for _ in range(PARALLEL_EPISODES)]
                worlds_rewards = [None for _ in range(PARALLEL_EPISODES)]
                for world_id in range(PARALLEL_EPISODES):
                    worlds_states[world_id] = worlds_data[world_id][0]["state"]
                    worlds_rewards[world_id] = worlds_data[world_id][0]["reward"]
                    worlds_data[world_id].pop(0)

                actions = neural_model(worlds_states, worlds_rewards)
                print("sending actions")
                for world_id in range(PARALLEL_EPISODES):
                    client_socket.sendall(
                        struct.pack("<ii", world_id, actions[world_id])
                    )

                steps_counter += 1
                episode_changing_now = False

    finally:
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
        action='store_true',
        help="Whether to disable learning",
    )
    args = parser.parse_args()

    if args.load_weights:
        # Load weights if provided through command line arguments
        snapshot_epochs = int(args.load_weights.split("_")[-1][:-3])
        steps_counter = snapshot_epochs
        trained_frames_count = int((snapshot_epochs-(BATCH_SIZE/PARALLEL_EPISODES)) * BATCH_SIZE)
        episodes_counter = int(steps_counter / STEPS_PER_EPISODE)
        print(
            f"Loading weights from epoch {snapshot_epochs}, episode {episodes_counter}, trained frames {trained_frames_count}"
        )
        load_weights(model, args.load_weights)
        target_model = copy.deepcopy(model)

    receive_frame_data()