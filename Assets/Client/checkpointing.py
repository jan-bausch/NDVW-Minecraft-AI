import datetime
import os
import typing
import ray
import torch

from generating import GenerationInfo
from replaying import ReplayMemory, ReplayingInfo
from all_models import get_model

from config import Config
from model import Model


def load_checkpoint(
    checkpoint_dirname: str,
) -> tuple[
    Config, list[GenerationInfo], list[Model], ReplayingInfo, Model, ReplayMemory
]:
    config: Config = Config.parse(f"{checkpoint_dirname}/config.ini")
    generation_infos = [None for _ in range(config.generation_servers)]
    generation_models = [None for _ in range(config.generation_servers)]
    learner_infos = ReplayingInfo.from_config(config)
    learner_model: Model = None
    replay_memory: ReplayMemory = None

    for filename in os.listdir(checkpoint_dirname):
        if filename.startswith("generation_info_"):
            info = GenerationInfo.parse(f"{checkpoint_dirname}/{filename}")
            generation_infos[info.server_index] = info
        elif filename.startswith("generation_model_") and filename.endswith(
            ".params.pt"
        ):
            model = get_model(config)
            model.set_params(torch.load(f"{checkpoint_dirname}/{filename}", map_location=torch.device(config.device)))
            state_filename = filename.replace(".params.pt", ".state.pt")
            model.set_state(torch.load(f"{checkpoint_dirname}/{state_filename}"))
            server_index = int(
                filename.replace("generation_model_", "").replace(".params.pt", "")
            )
            generation_models[server_index] = model
        elif filename == "learner_info.ini":
            learner_infos = ReplayingInfo.parse(f"{checkpoint_dirname}/{filename}")
        elif filename == "learner_model.params.pt":
            learner_model = get_model(config)
            learner_model.set_params(torch.load(f"{checkpoint_dirname}/{filename}", map_location=torch.device(config.device)))
            learner_model.set_state(
                torch.load(f"{checkpoint_dirname}/learner_model.state.pt")
            )
        elif filename == "replay_memory.pickle":
            with open(f"{checkpoint_dirname}/{filename}", "rb") as f:
                replay_memory = ReplayMemory.deserialize(config, f.read())

    return (
        config,
        generation_infos,
        generation_models,
        learner_infos,
        learner_model,
        replay_memory,
    )


def checkpoint(
    config: Config,
    generation_infos: list[GenerationInfo],
    generation_models_params: list[dict],
    generation_models_state: list[dict],
    learner_infos: ReplayingInfo,
    learner_model_params: dict,
    learner_model_state: dict,
    replay_memory: ReplayMemory,
):
    _config = config.get_parseable_config()
    checkpoint_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    dirname = f"checkpoints/{config.model}_{checkpoint_time_str}"

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    with open(f"{dirname}/config.ini", "w") as f:
        _config.write(f)

    for i, generation_info in enumerate(generation_infos):
        text = generation_info.to_config_text()
        filename = f"generation_info_{generation_info.server_index}.ini"
        with open(f"{dirname}/{filename}", "w") as f:
            f.write(text)

    for i in range(len(generation_models_params)):
        filename = f"generation_model_{i}.params.pt"
        torch.save(generation_models_params[i], f"{dirname}/{filename}")
        state = generation_models_state[i]
        filename = f"generation_model_{i}.state.pt"
        torch.save(state, f"{dirname}/{filename}")

    text = learner_infos.to_config_text()
    filename = f"learner_info.ini"
    with open(f"{dirname}/{filename}", "w") as f:
        f.write(text)

    filename = f"learner_model.params.pt"
    torch.save(learner_model_params, f"{dirname}/{filename}")

    state = learner_model_state
    filename = f"learner_model.state.pt"
    torch.save(state, f"{dirname}/{filename}")

    replay_memory_filename = f"replay_memory.pickle"
    with open(f"{dirname}/{replay_memory_filename}", "wb") as f:
        f.write(ray.get(replay_memory.serialize.remote()))
