from trainer import train_hardmseg, train_medt
from config import TrainConfig

config = TrainConfig.load_config_class('config/train_config/train_config.yaml')
train_hardmseg(config)