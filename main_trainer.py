from hardmseg_trainer.hardmseg_train import train
from configs import TrainConfig

config = TrainConfig.load_config_class('configs/train_configs/train_config.yaml')
train(config)