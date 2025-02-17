# Tilted Quantile Gradient Updates for Quantile-Constrained Reinforcement Learning

Code for the paper "Tilted Quantile Gradient Updates for Quantile-Constrained Reinforcement Learning", AAAI 2025.

## Paper Abstract

Safe reinforcement learning (RL) is a popular and versatile paradigm to learn reward-maximizing policies with safety guarantees. Previous works tend to express the safety constraints in an expectation form due to the ease of implementation, but this turns out to be ineffective in maintaining safety constraints with high probability. To this end, we move to the quantile-constrained RL that enables a higher level of safety without any expectation-form approximations. We directly estimate the quantile gradients through sampling and provide the theoretical proofs of convergence. Then a tilted update strategy for quantile gradients is implemented to compensate the asymmetric distributional density, with a direct benefit of return performance. Experiments demonstrate that the proposed model fully meets safety requirements (quantile constraints) while outperforming the state-of-the-art benchmarks with higher return.

## Implementation tutorial

The code is implemented on Linux with Python 3.7 and PyTorch 1.5.1, simulation environment requires ``mujoco``, ``gym`` and ``safety_gym``. Conda env is recommended to run the code.

#### 1 pytorch and other packages

first choose a directory, create conda env and activate it, then

```python
conda install pytorch==1.5.1 torchvision==0.6.1 cpuonly opencv==3.4.2 -c pytorch
pip install pyprind psutil
```

#### 2 mujoco

go to safety gym website https://github.com/openai/safety-gym, follow the instructions to install the corresponding version of ``mujoco``

#### 3 rlpyt

```python
git clone https://github.com/astooke/rlpyt.git
cd rlpyt
pip install -e .
```

#### 4 safety gym

```
git clone https://github.com/openai/safety-gym.git
cd safety-gym
pip install -e .
pip uninstall numpy
pip install numpy
```

#### 5 install TQPO

copy tqpo folder in this repository to ``rlpyt/rlpyt/projects``

## Run training code

activate your conda env, then

```python
cd rlpyt/rlpyt/projects/tqpo/experiments
python launch_tqpo.py
```

Code is being sorted out and will be released in one week.
