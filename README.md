# Julia Reinforcement Learning Algorithms

## Original Sources
https://github.com/MicrosoftLearning/Reinforcement-Learning-Explained  
https://github.com/udacity/deep-reinforcement-learning  
https://spinningup.openai.com/en/latest/user/algorithms.html  
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/  
https://github.com/dennybritz/reinforcement-learning  

## Algorithms
* Q-Learning
* Sarsa
* Expected-Sarsa
* MC-Control
* MC-Prediction
* Hill Climbing w/ Adaptive Noise Scaling
* CEM
* DQN
* DDQN
* Reinforce
* A2C
* A3C
* VPG
* DDPG
* TD3
* SAC
* PPO (Discrete, Continuous)
* UDRL (Upside-Down RL)
* Bandits

### RLAlgorithms
* Re-export RLHelpers & RLEnvironments
* episode!, episodes!: Take actions/ steps in environment and save values depending on memory type

### RLHelpers
* gather, loop, softupdate!
* Replay-Memories & transformations

### RLEnvironments
* SimpleRooms
* Pendulum
* CartPole (+ Continuous)
* MountainCar (+ Continuous)
* Simple Bandit

```julia
# types: S: State, A: Action, R: Reward
abstract type Environment{S,A,R<:Real} end
abstract type DiscEnv{S,A,R} <: Environment{S,A,R} end
abstract type ContEnv{S,A,R} <: Environment{S,A,R} end
abstract type DiffEnv{S,A,R} <: Environment{S,A,R} end
step!(env::Environment, action)
reset!(env::Environment)

env.observationspace
env.actionspace

# S <: Real: n = numstates
# S <: Vector{<:Real}: n = length(state)
struct ObservationSpace{S}
    n::Int64
end

# DiscEnv: Vector{A} (different actions), 
# ContEnv: Vector{Tuple{A, A}} (action ranges),
# n = length(actionspace)
struct ActionSpace{A}
    actions::Vector{A}
end
```
