import random
from attr import dataclass
import pickle
import typing
import ray

from config import Config


@dataclass
class Transition:
    state: list
    action: int
    reward: float
    importance: float = 1.0

    def serialize(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data: bytes) -> 'Transition':
        return pickle.loads(data)


@ray.remote
class ReplayMemory:
    trajectories: list[list[Transition]]
    trajectories_sum_importance: list[float]
    trajectories_complete: list[bool]

    def __init__(self, config: Config):
        self.config = config
        self.trajectories = []
        self.trajectories_sum_importance = []
        self.trajectories_complete = []

    def new_trajectory(self) -> int:
        self.trajectories.append([])
        self.trajectories_sum_importance.append(0.0)
        self.trajectories_complete.append(False)
        return len(self.trajectories) - 1
    
    def complete_trajectory(self, trajectory_index: int):
        self.trajectories_complete[trajectory_index] = True

    def push(self, transition: Transition, trajectory_index: int):
        if len(self.trajectories) <= trajectory_index:
            self.trajectories.append([])
        self.trajectories[trajectory_index].append(transition)
        self.trajectories_sum_importance[trajectory_index] += transition.importance

        if len(self.trajectories[trajectory_index]) > self.config.replay_transitions:
            self.trajectories_sum_importance[trajectory_index] -= self.trajectories[trajectory_index][0].importance
            self.trajectories[trajectory_index].pop(0)

    def sample_trajectories(self, batch_size: int) -> (list[list[Transition]], list[int]):
        complete_trajectories_indices = [i for i, complete in enumerate(self.trajectories_complete) if complete]
        complete_trajectories = [self.trajectories[i] for i in complete_trajectories_indices]
        complete_trajectories_sum_importance = [self.trajectories_sum_importance[i] for i in complete_trajectories_indices]
        random_indices = random.choices(range(len(complete_trajectories)), weights=complete_trajectories_sum_importance, k=min(batch_size, len(complete_trajectories)))
        random_trajectories = [complete_trajectories[i] for i in random_indices]
        return random_trajectories, random_indices

    def get_trajectories(self, indices: list[int]) -> list[list[Transition]]:
        complete_trajectories_indices = [i for i, complete in enumerate(self.trajectories_complete) if complete]
        complete_trajectories = [self.trajectories[i] for i in complete_trajectories_indices]
        return [complete_trajectories[i] for i in indices]

    def serialize(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data: bytes) -> 'ReplayMemory':
        return pickle.loads(data)


@dataclass
class ReplayingInfo:
    current_trajectory_indexes: list[int]
    current_step: int
    validation = False

    def to_config_text(self) -> str:
        return f"""[replaying]
current_trajectory_indexes={','.join(map(str, self.current_trajectory_indexes))}
current_step={self.current_step}
validation={self.validation}"""
    
    def from_config(config: Config) -> "ReplayingInfo":
        _config = config.get_parseable_config()

        trajectories_indexes = _config.get("replaying", "current_trajectory_indexes")
        trajectories_indexes = list(map(int, trajectories_indexes.split(',')))
        return ReplayingInfo(
            current_trajectory_indexes=trajectories_indexes,
            current_step=_config.getint("replaying", "current_step"),
            validation=_config.getboolean("replaying", "validation")
        )