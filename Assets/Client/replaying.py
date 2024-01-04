import math
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
    trajectories_age: list[int]
    trajectories_complete: list[bool]
    trajectory_replacement_probability: list[float]
    age = 0.0

    def __init__(self, config: Config):
        self.config = config
        self.trajectories = []
        self.trajectories_sum_importance = []
        self.trajectories_complete = []
        self.trajectories_age = []
        self.last_print_time = time.time()
        self.transitions_since_last_print = 0  # Track transitions added since last print

    def _compute_trajectory_replacement_score(self, index: int) -> float:
        if not self.trajectories_complete[index]:
            return 0.0
        
        age = 1 / (self.trajectories_age[index]+1)
        importance = 1 / (self.trajectories_sum_importance[index] + 1)

        return age * importance

    def new_trajectory(self) -> int:
        if len(self.trajectories) > self.config.replay_trajectories:
            trajectories_replacement_probability = [
                self._compute_trajectory_replacement_score(i) for i in range(len(self.trajectories))
            ]

            index = random.choices(range(len(self.trajectories)), weights=trajectories_replacement_probability)[0]
            self.trajectories[index] = []
            self.trajectories_sum_importance[index] = 0.0
            self.trajectories_age[index] = self.age
            self.trajectories_complete[index] = False
            
            return index

        self.trajectories.append([])
        self.trajectories_sum_importance.append(0.0)
        self.trajectories_age.append(self.age)
        self.trajectories_complete.append(False)

        self.age += 0.001
        return len(self.trajectories) - 1
    
    def complete_trajectory(self, trajectory_index: int):
        self.trajectories_complete[trajectory_index] = True

    def complete_count(self) -> int:
        return sum(self.trajectories_complete)
 
    def trajectory_count(self) -> int:
        return len(self.trajectories)

    def push(self, transition: Transition, trajectory_index: int):
        current_time = time.time()
        time_since_last_print = current_time - self.last_print_time

        if time_since_last_print >= 10:
            transitions_per_hour = self.transitions_since_last_print * 3600 / time_since_last_print
            current_memory_size = sum([len(trajectory) for trajectory in self.trajectories])
            print(f"Transitions per hour: {transitions_per_hour}")
            print(f"Complete trajectories per hour: {transitions_per_hour / self.config.steps_per_episode}")
            print(f"Trajectory count: {self.trajectory_count()}/{self.config.replay_trajectories}")
            print(f"Complete trajectories count: {self.complete_count()}")
            print(f"Transitions amount: {current_memory_size}")
            
            # Reset count and update last print time
            self.transitions_since_last_print = 0
            self.last_print_time = current_time

        self.trajectories[trajectory_index].append(transition)
        self.trajectories_sum_importance[trajectory_index] += transition.importance
        
        # Increment transitions added since last print
        self.transitions_since_last_print += 1

    def sample_trajectories(self, batch_size: int) -> (list[list[Transition]], list[int]):
        print(f"Trajectory count: {self.trajectory_count()}/{self.config.replay_trajectories}")
        print(f"Complete count: {self.complete_count()}")
        
        complete_trajectories_indices = [i for i, complete in enumerate(self.trajectories_complete) if complete]
        complete_trajectories = [self.trajectories[i] for i in complete_trajectories_indices]
        complete_trajectories_sum_importance = [self.trajectories_sum_importance[i] for i in complete_trajectories_indices]
        if len(complete_trajectories) == 0:
            return [], []
        random_indices = random.choices(
            range(len(complete_trajectories)), 
            weights=[math.pow(complete_trajectories_sum_importance[i]*10, 1.5)+1 for i in range(len(complete_trajectories))], 
            k=min(batch_size, len(complete_trajectories))
        )
        random_trajectories = [complete_trajectories[i] for i in random_indices]

        avg_importance = sum(complete_trajectories_sum_importance) / len(complete_trajectories_sum_importance)
        print(f"Average importance of complete trajectories: {avg_importance}")
        avg_importance_sampled = sum([complete_trajectories_sum_importance[i] for i in random_indices]) / len(random_indices)
        print(f"Average importance of sampled trajectories: {avg_importance_sampled}")

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