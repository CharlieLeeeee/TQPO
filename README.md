# Tilted Quantile Gradient Updates for Quantile-Constrained Reinforcement Learning
Code for the paper "Tilted Quantile Gradient Updates for Quantile-Constrained Reinforcement Learning", AAAI 2025.

# Paper Abstract

Safe reinforcement learning (RL) is a popular and versatile paradigm to learn reward-maximizing policies with safety guarantees. Previous works tend to express the safety constraints in an expectation form due to the ease of implementation, but this turns out to be ineffective in maintaining safety constraints with high probability. To this end, we move to the quantile-constrained RL that enables a higher level of safety without any expectation-form approximations. We directly estimate the quantile gradients through sampling and provide the theoretical proofs of convergence. Then a tilted update strategy for quantile gradients is implemented to compensate the asymmetric distributional density, with a direct benefit of return performance. Experiments demonstrate that the proposed model fully meets safety requirements (quantile constraints) while outperforming the state-of-the-art benchmarks with higher return.

# Dependencies

The environment requires ``mujoco``, OpenAI ``gym`` and ``safety_gym``

Code is being sorted out and will be released in the future
