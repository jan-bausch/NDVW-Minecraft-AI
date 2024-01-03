import random
import time
from attr import dataclass
import pickle
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
    overlap = 0

    def __init__(self, config: Config):
        self.config = config
        self.trajectories = []
        self.trajectories_sum_importance = []
        self.trajectories_complete = []
        self.last_print_time = time.time()
        self.transitions_since_last_print = 0  # Track transitions added since last print

    def new_trajectory(self) -> int:
        if len(self.trajectories) > self.config.replay_trajectories:
            self.trajectories[self.overlap] = []
            self.trajectories_sum_importance[self.overlap] = 0.0
            self.trajectories_complete[self.overlap] = False
            index = self.overlap
            self.overlap = (self.overlap + 1) % self.config.replay_trajectories
            return index

        self.trajectories.append([])
        self.trajectories_sum_importance.append(0.0)
        self.trajectories_complete.append(False)
        return len(self.trajectories) - 1
    
    def complete_trajectory(self, trajectory_index: int):
        self.trajectories_complete[trajectory_index] = True

    def push(self, transition: Transition, trajectory_index: int):
        current_time = time.time()
        time_since_last_print = current_time - self.last_print_time

        if time_since_last_print >= 10:
            transitions_per_hour = self.transitions_since_last_print * 3600 / time_since_last_print
            current_memory_size = sum([len(trajectory) for trajectory in self.trajectories])
            print(f"Transitions per hour: {transitions_per_hour}")
            print(f"Complete trajectories per hour: {transitions_per_hour / self.config.steps_per_episode}")
            print(f"Current memory size: {current_memory_size}")
            
            # Reset count and update last print time
            self.transitions_since_last_print = 0
            self.last_print_time = current_time

        self.trajectories[trajectory_index].append(transition)
        self.trajectories_sum_importance[trajectory_index] += transition.importance
        
        # Increment transitions added since last print
        self.transitions_since_last_print += 1

    def sample_trajectories(self, batch_size: int) -> (list[list[Transition]], list[int]):
        complete_trajectories_indices = [i for i, complete in enumerate(self.trajectories_complete) if complete]
        complete_trajectories = [self.trajectories[i] for i in complete_trajectories_indices]
        complete_trajectories_sum_importance = [self.trajectories_sum_importance[i] for i in complete_trajectories_indices]
        if len(complete_trajectories) == 0:
            return [], []
        random_indices = random.choices(range(len(complete_trajectories)), weights=complete_trajectories_sum_importance, k=min(batch_size, len(complete_trajectories)))
        random_trajectories = [complete_trajectories[i] for i in random_indices]
        return random_trajectories, random_indices

    def get_trajectories(self, indices: list[int]) -> list[list[Transition]]:
        complete_trajectories_indices = [i for i, complete in enumerate(self.trajectories_complete) if complete]
        complete_trajectories = [self.trajectories[i] for i in complete_trajectories_indices]
        return [complete_trajectories[i] for i in indices]

    def serialize(self) -> bytes:
        return pickle.dumps({
            "trajectories": self.trajectories,
            "trajectories_sum_importance": self.trajectories_sum_importance,
            "trajectories_complete": self.trajectories_complete,
        })

    @classmethod
    def deserialize(cls, config: Config, data: bytes) -> 'ReplayMemory':
        data = pickle.loads(data)
        memory = ReplayMemory.remote(config)
        memory.trajectories = data["trajectories"]
        memory.trajectories_sum_importance = data["trajectories_sum_importance"]
        memory.trajectories_complete = data["trajectories_complete"]
        return memory


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