import csv
import torch
import torch.nn as nn
from torchrl.modules import NoisyLinear

from model import Model
from config import Config

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels // 2, kernel_size=1
        )
        self.bn1 = nn.BatchNorm2d(channels // 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channels // 2,
            out_channels=channels // 2,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(channels // 2)
        self.conv3 = nn.Conv2d(
            in_channels=channels // 2, out_channels=channels, kernel_size=1
        )
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += residual
        x = self.relu(x)
        return x


class DownscalingResidualBlock(nn.Module):
    def __init__(self, channels, s):
        super(DownscalingResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels // s, kernel_size=1
        )
        self.bn1 = nn.BatchNorm2d(channels // s)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channels // s,
            out_channels=channels // s,
            kernel_size=3,
            padding=1,
            stride=s,
        )
        self.bn2 = nn.BatchNorm2d(channels // s)
        self.conv3 = nn.Conv2d(
            in_channels=channels // s, out_channels=(4 * channels) // s, kernel_size=1
        )
        self.bn3 = nn.BatchNorm2d((4 * channels) // s)
        self.downscale = nn.Conv2d(
            in_channels=channels,
            out_channels=(4 * channels) // s,
            kernel_size=1,
            stride=s,
        )
        self.downscale_bn = nn.BatchNorm2d((4 * channels) // s)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        residual = self.downscale(residual)
        residual = self.downscale_bn(residual)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += residual
        x = self.relu(x)
        return x


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
            in_channels=self.frame_channels, out_channels=64, kernel_size=9
        )  # 3x64x64 -> 64x56x56

        self.bnconv1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.residual1 = ResidualBlock(64)
        self.residual2 = ResidualBlock(64)
        self.dresidual1 = DownscalingResidualBlock(64, 2)  # 64x56x56 -> 128x28x28
        self.residual3 = ResidualBlock(128)
        self.residual4 = ResidualBlock(128)
        self.dresidual2 = DownscalingResidualBlock(128, 2)  # 128x28x28 -> 256x14x14
        self.residual5 = ResidualBlock(256)
        self.residual6 = ResidualBlock(256)
        self.dresidual3 = DownscalingResidualBlock(256, 2)  # 256x14x14 -> 512x7x7
        self.residual7 = ResidualBlock(512)
        self.residual8 = ResidualBlock(512)

        self.fc1 = NoisyLinear(self.calculate_conv_output_dim() + other_dim, 256, std_init=0.5)
        self.bnfc1 = nn.BatchNorm1d(256)
        self.fc2 = NoisyLinear(256, 128, std_init=0.5)
        self.lstm_num_layers = 1
        self.lstm_hidden_dim = 64
        self.lstm = nn.LSTM(
            input_size=128,
            num_layers=self.lstm_num_layers,
            hidden_size=self.lstm_hidden_dim,
            batch_first=True,
        )
        self.fc3 = NoisyLinear(64, output_dim, std_init=0.5)
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
        x = self.bnconv1(x)
        x = self.relu(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.dresidual1(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.dresidual2(x)
        x = self.residual5(x)
        x = self.residual6(x)
        x = self.dresidual3(x)
        x = self.residual7(x)
        x = self.residual8(x)

        return int(torch.prod(torch.tensor(x.size())))

    def forward(self, x: torch.Tensor):
        self.lstm.flatten_parameters()
        x_other = x[:, : self.other_dim]
        x_frame = x[:, self.other_dim :]

        x = x_frame.view(
            x.size(0), self.frame_channels, self.frame_dim[0], self.frame_dim[1]
        )  # Reshape input to (batch_size, channels, height, width)

        x = self.conv1(x)
        x = self.bnconv1(x)
        x = self.relu(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.dresidual1(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.dresidual2(x)
        x = self.residual5(x)
        x = self.residual6(x)
        x = self.dresidual3(x)
        x = self.residual7(x)
        x = self.residual8(x)

        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = torch.cat((x_other, x), dim=1)
        x = self.fc1(x)
        x = self.bnfc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = x.unsqueeze(1)
        lstm_out, self.lstm_hidden = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc3(x)
        return x
    
    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()
    

class DqnLstmNoisyResModel(Model):
    config: Config
    net: DQN
    target_net: DQN
    device: torch.device
    seed: int

    gamma: float
    learning_rate: float
    n_step: int

    previous_n: list[dict]

    trained_frames_count: int = 0
    target_independence_frames: int = 0
    
    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.net = DQN(
            (config.input_visual_dim[0], config.input_visual_dim[1]),
            config.input_visual_dim[2],
            config.input_state_dim,
            config.output_dim,
        )
        self.target_net = DQN(
            (config.input_visual_dim[0], config.input_visual_dim[1]),
            config.input_visual_dim[2],
            config.input_state_dim,
            config.output_dim,
        )
        self.target_net.load_state_dict(self.net.state_dict())
        self.device = torch.device("cpu")
        self.net.to(self.device)
        self.target_net.to(self.device)
        self.seed = 0
        _config = config.get_parseable_config()
        self.gamma = _config.getfloat("model", "gamma")
        self.learning_rate = _config.getfloat("model", "learning_rate")
        self.n_step = _config.getint("model", "n_step")
        self.previous_n = []
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def inference(self, states: list[list[int]], force_greedy: bool = False) -> list[int]:
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        self.net.eval()
        with torch.no_grad():
            return self.net(states_tensor).argmax(dim=1).tolist()

    def inference_train(self, states: list[list[int]], force_greedy: bool = False) -> list[int]:
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        self.net.train()
        with torch.no_grad():
            return self.net(states_tensor).argmax(dim=1).tolist()

    def get_params(self) -> dict:
        return self.net.state_dict()
    
    def set_params(self, params: dict):
        self.net.load_state_dict(params)

    def get_state(self) -> dict:
        return {
            "lstm_hidden": self.net.get_lstm_hidden(),
            "previous_n": self.previous_n,
            "optimizer": self.optimizer.state_dict(),
        }

    def set_state(self, state: dict):
        self.net.set_lstm_hidden(state["lstm_hidden"])
        self.previous_n = state["previous_n"]
        self.optimizer.load_state_dict(state["optimizer"])
    
    def reset_state(self):
        self.net.reset_lstm(1)
        self.previous_n = []

    def to_device(self, device: torch.device):
        self.device = device
        self.net.to(device)
        self.target_net.to(device)

    def get_device(self) -> torch.device:
        return self.device

    def set_seed(self, seed: int):
        self.seed = seed

    def _push_previous(self, states: list[list[int]], rewards: list[float], actions: list[int]):
        if len(self.previous_n) >= self.n_step:
            self.previous_n.pop(0)
        self.previous_n.append({"states": states, "rewards": rewards, "actions": actions})

    def train(self, states: list[list[int]], actions: list[int], rewards: list[float]):
        self.net.train()
        self.target_net.train()
        batch_size = len(states)

        worlds_states_tensor = torch.tensor(
            states, dtype=torch.float32, device=self.device
        )

        n_steps = min(self.n_step, len(self.previous_n))
        if n_steps == 0:
            self._push_previous(states, rewards, actions)
            return

        previous_state = self.previous_n[-n_steps]["states"]
        previous_actions = self.previous_n[-n_steps]["actions"]

        previous_rewards = [previous["rewards"] for previous in self.previous_n[:n_steps]]

        targets = torch.tensor(
            [0 for _ in range(batch_size)], dtype=torch.float32, device=self.device
        )
        predictions = torch.tensor(
            [0 for _ in range(batch_size)], dtype=torch.float32, device=self.device
        )
        previous_worlds_states_tensor = torch.tensor(
            previous_state, dtype=torch.float32, device=self.device
        )

        prediction_actions_values = self.net(previous_worlds_states_tensor)
        target_actions_values = self.target_net(worlds_states_tensor)

        #variances = torch.var(target_actions_values, dim=0)
        #print(f"Q* values min variance: {variances.min().item()}")
        #variances = torch.var(prediction_actions_values, dim=0)
        #print(f"Q values min variance: {variances.min().item()}")

        for i in range(batch_size):
            target_action = torch.argmax(target_actions_values[i]).item()
            target_action_value = target_actions_values[i][target_action]
            action = previous_actions[i]
            predictions[i] = prediction_actions_values[i][action]
            targets[i] = rewards[i]
            for j in range(n_steps):
                targets[i] += previous_rewards[-(j+1)][i] * (self.gamma ** (j + 1))
            targets[i] += target_action_value * (self.gamma ** (n_steps + 1))

        loss = self.criterion(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_value = loss.item()

        self.net.reset_noise()
        self.target_net.reset_noise()

        self.trained_frames_count += batch_size
        self.target_independence_frames += batch_size

        self._push_previous(states, rewards, actions)

        print(f"Trained frames: {self.trained_frames_count}")
        if self.target_independence_frames >= 10_000:
            print("Updating target model")
            self.target_independence_frames = 0
            self.target_net = DQN(
                (self.config.input_visual_dim[0], self.config.input_visual_dim[1]),
                self.config.input_visual_dim[2],
                self.config.input_state_dim,
                self.config.output_dim,
            )
            self.target_net.load_state_dict(self.net.state_dict())
            self.target_net.to(self.device)     

        with open(f"{self.config.model}_training_data.csv", mode="a") as file:
            csv_writer = csv.writer(file)
            file.seek(0, 1)

            if file.tell() == 0:
                csv_writer.writerow(
                    [
                        "trained_frames_count",
                        "q_values",
                        "q_target_values",
                        "loss",
                    ]
                )

            prediction_q_values_string = ' '.join(map(str, prediction_actions_values.mean(dim=0).tolist()))
            target_q_values_string = ' '.join(map(str, target_actions_values.mean(dim=0).tolist()))
            csv_writer.writerow(
                [
                    self.trained_frames_count,
                    prediction_q_values_string,
                    target_q_values_string,
                    loss_value,
                ]
            )