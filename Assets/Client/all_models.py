from dqn_lstm_noisy_res import DqnLstmNoisyResModel
from dqn_lstm_noisy import DqnLstmNoisyModel
from model import Model
from config import Config

def get_model(config: Config) -> Model:
    if config.model == 'dqn_lstm_noisy_res':
        return DqnLstmNoisyResModel(config)
    
    if config.model == 'dqn_lstm_noisy':
        return DqnLstmNoisyModel(config)