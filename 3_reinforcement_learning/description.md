# Reinforcement Learning — Overview & Practical Guide

Short summary  
Reinforcement Learning (RL) is a branch of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative reward. RL sits between supervised learning (learning from labeled examples) and optimal control (planning under known dynamics). This guide gives a compact, practical, and reference-style overview suitable for engineers and practitioners who want intuition, fundamentals, common algorithms, engineering tips, and short code examples.

Table of Contents
- [Core concepts & vocabulary](#core-concepts--vocabulary)
- [Formalism: Markov Decision Processes (MDPs)](#formalism-markov-decision-processes-mdps)
- [Value functions and Bellman equations](#value-functions-and-bellman-equations)
- [Main algorithm families](#main-algorithm-families)
  - [Value-based](#value-based)
  - [Policy-based](#policy-based)
  - [Actor–Critic and hybrid](#actorcritic-and-hybrid)
  - [Model-based](#model-based)
- [Exploration vs. exploitation](#exploration-vs-exploitation)
- [Practical engineering tips](#practical-engineering-tips)
- [Benchmarks & environments](#benchmarks--environments)
- [Evaluation, reproducibility & compute](#evaluation-reproducibility--compute)
- [Debugging & visualization](#debugging--visualization)
- [Common pitfalls](#common-pitfalls)
- [Short code examples](#short-code-examples)
- [Recommended reading & papers](#recommended-reading--papers)
- [Appendix: notation cheat-sheet, policy-gradient sketch, glossary](#appendix-notation-cheat-sheet-policy-gradient-sketch-glossary)

---

Core concepts & vocabulary
- Agent: the learner/decision-maker.
- Environment: the external system the agent interacts with.
- State (s): a representation of the environment at a time step.
- Action (a): a choice made by the agent.
- Reward (r): scalar feedback from the environment.
- Episode: one sequence of states, actions, rewards until termination.
- Return (G): cumulative future reward (often discounted).
- Policy (π): mapping from states to actions (deterministic or stochastic).
- Value function (V, Q): expected return under a policy.
- Trajectory: sequence (s0, a0, r1, s1, a1, r2, ...).

Formalism: Markov Decision Processes (MDPs)
An MDP is a tuple (S, A, P, R, γ):
- S: state space
- A: action space
- P(s' | s, a): transition probability
- R(s, a) or R(s, a, s'): reward function
- γ ∈ [0,1]: discount factor

Objective: find policy π* maximizing expected (discounted) return:
$$J(π) = E_{τ∼π}\left[\sum_{t=0}^{∞} γ^t r_{t+1}\right]$$
where τ denotes trajectories sampled following π.

Value functions
- State-value: $V^{π}(s)=E[ \sum_{t=0}^{∞} γ^t r_{t+1} \mid s_0=s, π ]$
- Action-value (Q): $Q^{π}(s,a)=E[ \sum_{t=0}^{∞} γ^t r_{t+1} \mid s_0=s, a_0=a, π ]$

Bellman expectation equation:
$$V^{π}(s) = E_{a∼π(·|s), s'∼P(·|s,a)}[ r(s,a,s') + γ V^{π}(s') ]$$

Optimality: Bellman optimality equation (for V* or Q*).

Main algorithm families

Value-based
- Dynamic Programming (policy evaluation / improvement) — requires known model.
- Monte Carlo (MC) methods — learn from complete episodes.
- Temporal Difference (TD) learning — updates bootstrapping from next value.
- Q-Learning (off-policy TD): learn Q* directly; update:  
  Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
- SARSA (on-policy TD): uses chosen action at next state.
- Deep Q-Network (DQN): neural networks as function approximators; practical tricks: replay buffer, target network, gradient clipping, double DQN, dueling DQN, prioritized replay.

Policy-based
- Directly parameterize policy π_θ(a|s) and optimize J(θ).
- REINFORCE (Monte Carlo policy gradient): unbiased but high variance.
- Policy gradient objective and typical estimator:
  $$∇_θ J(θ) ≈ E_{π_θ} \left[ ∑_t ∇_θ \log π_θ(a_t|s_t) \,G_t \right]$$
- Use baselines (e.g., V(s)) to reduce variance: advantage A_t = G_t − b(s_t).

Actor–Critic and hybrid
- Actor: policy network π_θ; Critic: value network V_w to estimate baseline.
- A2C/A3C: synchronous/asynchronous variants that stabilize learning.
- PPO (Proximal Policy Optimization): clipped surrogate objective for stable updates — popular default.
- DDPG: deterministic actor for continuous actions (uses replay buffer).
- TD3: improves DDPG (delay updates, target policy smoothing).
- SAC (Soft Actor-Critic): maximum entropy RL encouraging exploration via stochastic policies.

Model-based
- Learn a model P̂ and R̂, then plan (MPC, value-iteration on model).
- Can be sample-efficient but sensitive to model bias.
- Hybrid approaches combine model learning + model-free correction.

Exploration vs. exploitation
- ε-greedy: simple, effective for discrete actions.
- Boltzmann/softmax action selection.
- Upper Confidence Bound (UCB) — common in bandits.
- Thompson sampling — Bayesian approach.
- Intrinsic motivation / curiosity bonuses — reward for novel states (e.g., prediction error).
- Count-based exploration (or approximations using density models).

Practical engineering tips
- Normalize observations and rewards (running mean/std).
- Clip rewards when dealing with high variance tasks (but be careful—may change optimal policy).
- Frame stacking (Atari), image preprocessing (grayscale, resize).
- Use replay buffers for off-policy learners; consider prioritized replay.
- Target networks in DQN to stabilize bootstrapping.
- Use gradient clipping and learning-rate scheduling.
- Batch size: larger batches stabilize gradient estimates but cost compute.
- Checkpoint models frequently and log seeds/config.
- Reward shaping: add auxiliary rewards sparingly — can create shortcuts.
- Curriculum learning: start with easier tasks, gradually increase difficulty.
- Use vectorized environments for sample collection (e.g., stable-baselines3 VecEnv).
- For continuous control, use action noise (OU noise for DDPG historically, Gaussian for TD3).

Benchmarks & environments
- OpenAI Gym classic control (CartPole, MountainCar), Atari (ALE), MuJoCo (continuous control).
- DeepMind Control Suite, ProcGen (procedural generalization), Roboschool / Isaac Gym for high-throughput simulation.
- Safety and constrained RL benchmarks (SafeRL Benchmarks).

Evaluation, reproducibility & compute
- Evaluation metrics: average episodic return, sample efficiency (return vs steps), success rate, stability across seeds.
- Use multiple random seeds (≥5, often 10+) and report mean ± std or median and interquartile ranges.
- Track wall-clock vs environment steps — for reproducibility note both.
- Reproducibility: fix seeds (env, np, torch), record package versions, and hardware (GPU/CPU).
- Consider mixed precision for speed; distributed rollouts for throughput.

Debugging & visualization
- Plot training curves (mean return, ±std) and learning rate / loss curves.
- Visualize episodes (render environment frames) to catch degenerate behaviors.
- Inspect value estimates and advantage signals for drift or collapse.
- t-SNE or PCA on state embeddings to analyze learned representations.
- Use small deterministic environments and unit tests for algorithm behavior.
- Curriculum / ablation runs to isolate causes of failures.

Common pitfalls
- Reward hacking: agent finds unintended way to maximize reward.
- Sparse rewards: poor learning without shaping or clever exploration.
- Overfitting to environment specifics: poor generalization.
- High variance gradients: need baselines, advantage estimators, or variance reduction.
- Hyperparameter sensitivity: learning rates, entropy weight, discount γ, batch size.
- Bug sources: incorrect discounting, off-by-one trajectory handling, terminal-state bootstrap mistakes.

Short code examples
Note: these are compact templates — adapt for production (logging, seeding, device handling).

1) REINFORCE (PyTorch — minimal, episodic)
```python
# Simple REINFORCE pseudocode (PyTorch-like)
import torch
import torch.nn as nn
import torch.optim as optim
import gym

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

policy = Policy()
opt = optim.Adam(policy.parameters(), lr=1e-3)

for episode in range(1000):
    obs = env.reset()
    rewards = []
    log_probs = []
    done = False
    while not done:
        obs_v = torch.FloatTensor(obs).unsqueeze(0)
        probs = policy(obs_v)
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        log_probs.append(m.log_prob(torch.tensor(action)))
        obs, r, done, _ = env.step(action)
        rewards.append(r)
    # compute returns
    returns = []
    G = 0
    gamma = 0.99
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    loss = 0
    for lp, G in zip(log_probs, returns):
        loss += -lp * G
    opt.zero_grad()
    loss.backward()
    opt.step()
```

2) Using Stable-Baselines3 (PPO) — quick start
```python
from stable_baselines3 import PPO
import gym

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_cartpole")
```

Practical tips for these examples:
- Use vectorized envs (SubprocVecEnv / DummyVecEnv) to speed up data collection.
- For Atari use wrappers: frame skip, gray-scaling, frame stacking, and no-op resets.

Recommended reading & papers
- Sutton & Barto, "Reinforcement Learning: An Introduction" — canonical textbook (start here).
- Richard S. Sutton, "Policy Gradient Methods for Reinforcement Learning with Function Approximation" (1999) — policy gradients.
- Mnih et al., "Human-level control through deep reinforcement learning" (DQN).
- Schulman et al., "Proximal Policy Optimization Algorithms" (PPO).
- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning" (SAC).
- Lillicrap et al., "Continuous control with deep reinforcement learning" (DDPG).
- OpenAI & DeepMind blog posts and implementations for practical tips.

Appendix: notation cheat-sheet, short policy-gradient sketch, glossary

Notation cheat-sheet
- t: time step index
- s_t: state at time t
- a_t: action at time t
- r_{t+1}: reward observed after executing a_t
- γ: discount factor
- π_θ: policy parameterized by θ
- V^π(s), Q^π(s,a): value functions
- G_t: return from time t: ∑_{k=0} γ^k r_{t+k+1}

Policy-gradient sketch
Goal: maximize J(θ) = E_{τ∼π_θ}[∑_t γ^t r_{t+1}]
Using the log-derivative trick:
∇_θ J(θ) = E_{τ∼π_θ} [∑_t ∇_θ log π_θ(a_t|s_t) * G_t]
Replace G_t by advantage estimate A_t for lower variance:
A_t = G_t − b(s_t) (baseline b often = V_w(s_t))

Glossary
- Off-policy: algorithm can learn from data generated by a different policy (e.g., Q-learning).
- On-policy: algorithm requires data from the current policy (e.g., REINFORCE).
- Bootstrapping: using estimates (value function) to update other estimates (e.g., TD).
- Entropy regularization: add entropy bonus to policy loss to encourage exploration.
- Sample efficiency: how many environment interactions needed to reach performance.

Closing notes
This file is intended as a compact reference/starting point. For hands-on learning, combine:
- reading (Sutton & Barto),
- implementing simple algorithms from scratch (start with tabular methods or CartPole),
- inspecting behavior visually and via training curves,
- then move to robust libraries (stable-baselines3, CleanRL) and standard benchmarks.

If you want, I can:
- Expand any section (math-heavy derivation, deeper algorithm pseudocode).
- Produce a shorter cheat-sheet or slide-friendly version.
- Create a polished README.md and commit it to the repository (I can do that next if you want).
