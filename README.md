# dqn
Tensorflow implementation of the deep Q-learning based on the DeepMind Nature paper in 2015 using OpenAI Gym as the environment. Environments and various hyper parameters as well as the option to enable Double Q-learning can be set in the configuration file.

## Dependencies
1. Tensorflow
2. OpenCV
3. Gym

## Usage
1. Edit `config.py`
2. Run `python train.py` to start training
3. Run `python play.py` to load the latest checkpoint and play 
