import torch


class Model():
    def inference(self, states: list[list[int]], force_greedy: bool = False) -> list[int]:
        raise NotImplementedError("Subclasses must implement inference method.")

    def inference_train(self, states: list[list[int]], force_greedy: bool = False) -> list[int]:
        raise NotImplementedError("Subclasses must implement inference method.")

    def train(self, states: list[list[int]], actions: list[int], rewards: list[float]):
        raise NotImplementedError("Subclasses must implement train method.")

    def get_params(self) -> dict:
        raise NotImplementedError("Subclasses must implement get_state_dict method.")
    
    def set_params(self, params: dict):
        raise NotImplementedError("Subclasses must implement set_state_dict method.")

    def get_state(self) -> dict:
        raise NotImplementedError("Subclasses must implement get_state method.")

    def set_state(self, state: dict):
        raise NotImplementedError("Subclasses must implement set_state method.")
    
    def reset_state(self):
        raise NotImplementedError("Subclasses must implement reset method.")

    def to_device(self, device: torch.device):
        raise NotImplementedError("Subclasses must implement to_device method.")
    
    def get_device(self) -> torch.device:
        raise NotImplementedError("Subclasses must implement get_device method.")
    
    def set_seed(self, seed: int):
        raise NotImplementedError("Subclasses must implement set_seed method.")