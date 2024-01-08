import configparser

import torch

class Config:
    seed: int
    address: str
    parallel_worlds: int
    batch_size: int
    generation_servers: int
    validation_servers: int
    replay_trajectories: int
    steps_per_episode: int
    input_visual_dim: tuple[int, int, int]
    input_state_dim: int
    output_dim: int
    model: str
    device: str
    _config: configparser.ConfigParser

    def __init__(self, config: configparser.ConfigParser):
        self.seed=config.getint('config', 'seed')
        self.address=config.get('config', 'address')
        self.parallel_worlds=config.getint('config', 'parallel_worlds')
        self.batch_size=config.getint('config', 'batch_size')
        self.generation_servers=config.getint('config', 'generation_servers')
        self.validation_servers=config.getint('config', 'validation_servers')
        self.replay_trajectories=config.getint('config', 'replay_trajectories')
        self.steps_per_episode=config.getint('config', 'steps_per_episode')
        self.input_visual_dim=tuple(map(int, config.get('config', 'input_visual_dim').split(',')))
        self.input_state_dim=config.getint('config', 'input_state_dim')
        self.output_dim=config.getint('config', 'output_dim')
        self.model=config.get('config', 'model')
        self._config=config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def get_parseable_config(self) -> configparser.ConfigParser:
        return self._config

    def parse(path: str) -> "Config":
        config = configparser.ConfigParser()
        config.read(path)
        print(f"Loaded config from {path}")
        print(f"Config: {config}")
        return Config(config)

    def loaded_episodes(self) -> int:
        if self.load_checkpoint is None:
            return 0
        raise NotImplementedError
        