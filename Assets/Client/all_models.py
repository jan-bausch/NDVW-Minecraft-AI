from dqn_lstm_noisy_res import DqnLstmNoisyResModel
from model import Model
from config import Config

def get_model(config: Config) -> Model:
    return DqnLstmNoisyResModel(config)