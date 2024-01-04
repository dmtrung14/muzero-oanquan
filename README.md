![supported platforms](https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20%7C%20Windows%20(soon)-929292)
![supported python versions](https://img.shields.io/badge/python-%3E%3D%203.6-306998)
![dependencies status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)
[![style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![license MIT](https://img.shields.io/badge/licence-MIT-green)
[![discord badge](https://img.shields.io/badge/discord-join-6E60EF)](https://discord.gg/GB2vwsF)


# MuZero O An Quan


An implementation of MuZero to play the game O An Quan based on Google DeepMind [paper](https://arxiv.org/abs/1911.08265) (Schrittwieser et al., Nov 2019) and the associated [pseudocode](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py).

Please refer to the [documentation](https://github.com/dmtrung14/muzero-oanquan/wiki/MuZero-O-An-Quan-Documentation). The [example model](https://github.com/dmtrung14/muzero-oanquan/blob/master/results/oanquan/2024-01-03--08-55-53/model.checkpoint) was trained on 100 self-play games, over the duration of 1.5 hours. 
This implementation is primarily for educational purpose.\
[Explanatory video of MuZero](https://youtu.be/We20YSAJZSE)

MuZero is a state of the art Reinforcement Learning algorithm for board games (Chess, Go, ...) and Atari games.
It is the successor to [AlphaZero](https://arxiv.org/abs/1712.01815) but without any knowledge of the environment underlying dynamics. MuZero learns a model of the environment and uses an internal representation that contains only the useful information for predicting the reward, value, policy and transitions. MuZero is also close to [Value prediction networks](https://arxiv.org/abs/1707.03497). See [How it works](https://github.com/werner-duvaud/muzero-general/wiki/How-MuZero-works).

## Features

* [x] Residual Network and Fully connected network in [PyTorch](https://github.com/pytorch/pytorch)
* [x] Multi-Threaded/Asynchronous/[Cluster](https://docs.ray.io/en/latest/cluster-index.html) with [Ray](https://github.com/ray-project/ray)
* [X] Multi GPU support for the training and the selfplay
* [x] TensorBoard real-time monitoring
* [x] Model weights automatically saved at checkpoints
* [x] Single and two player mode
* [x] Commented and [documented](https://github.com/dmtrung14/muzero-oanquan/wiki/MuZero-O-An-Quan-Documentation)
* [x] [Pretrained weights](https://github.com/werner-duvaud/muzero-general/tree/master/results) available

### Further improvements
Here is a list of features which could be interesting to add but which are not in MuZero's paper. We are open to contributions and other ideas.

* [x] [Hyperparameter search](https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization)
* [x] [Continuous action space](https://github.com/werner-duvaud/muzero-general/tree/continuous)
* [x] [Tool to understand the learned model](https://github.com/werner-duvaud/muzero-general/blob/master/diagnose_model.py)

Picking up from the original implementation by Werner Duvaud, we have also implemented
* [x] Batch MCTS
* [ ] Support of more than two player games


Tests are done on Ubuntu with 16 GB RAM / Intel i7 / GTX 1050Ti Max-Q. We make sure to obtain a progression and a level which ensures that it has learned. But we do not systematically reach a human level. For certain environments, we notice a regression after a certain time. The proposed configurations are certainly not optimal and we do not focus for now on the optimization of hyperparameters. Any help is welcome.

## Code structure

![code structure](https://github.com/dmtrung14/muzero-oanquan/blob/master/docs/code-structure-werner-duvaud.png)

Network summary:

<p align="center">
<a href="https://github.com/werner-duvaud/muzero-general/blob/master/docs/muzero-network-werner-duvaud.png">
<img src="https://github.com/werner-duvaud/muzero-general/blob/master/docs/muzero-network-werner-duvaud.png" width="250"/>
</a>
</p>

## Getting started
### Installation

```bash
git clone https://github.com/dmtrung14/muzero-oanquan.git
cd muzero-oanquan

pip install -r requirements.lock
```

### Run

```bash
python muzero.py
```
To visualize the training results, run in a new terminal:
```bash
tensorboard --logdir ./results
```

### Config

You can adapt the configurations of each game by editing the `MuZeroConfig` class of the respective file in the [games folder](https://github.com/werner-duvaud/muzero-general/tree/master/games).

## Related work

* [EfficientZero](https://arxiv.org/abs/2111.00210) (Weirui Ye, Shaohuai Liu, Thanard Kurutach, Pieter Abbeel, Yang Gao)
* [Sampled MuZero](https://arxiv.org/abs/2104.06303) (Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Mohammadamin Barekatain, Simon Schmitt, David Silver)

## Authors
* Trung Dang
* Werner Duvaud, Aurèle Hainaut, and Paul Lenoir
* [Contributors](https://github.com/dmtrung14/muzero-general/graphs/contributors)


## Getting involved

* [GitHub Issues](https://github.com/werner-duvaud/muzero-general/issues): For reporting bugs.
* [Pull Requests](https://github.com/werner-duvaud/muzero-general/pulls): For submitting code contributions.
* [Discord server](https://discord.gg/GB2vwsF): For discussions about development or any general questions.
