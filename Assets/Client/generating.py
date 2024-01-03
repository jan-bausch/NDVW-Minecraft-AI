from attr import dataclass

from config import Config


@dataclass
class GenerationInfo:
    server_index: int
    current_episode: int
    current_step: int
    validation = False

    def to_config_text(self) -> str:
        return f"""[generation]
server_index={self.server_index}
current_episode={self.current_episode}
current_step={self.current_step}
validation={self.validation}"""

    def from_config(config: Config) -> "GenerationInfo":
        _config = config.get_parseable_config()

        return GenerationInfo(
            server_index=_config.getint("generation", "server_index"),
            current_episode=_config.getint("generation", "current_episode"),
            current_step=_config.getint("generation", "current_step"),
            validation=_config.getboolean("generation", "validation"),
        )
