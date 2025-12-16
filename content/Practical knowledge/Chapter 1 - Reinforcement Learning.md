---
tags:
  - RL
Date: 2025-12-01
---
# 1. The big picture.

The idea of Reinforcement Learning (RL) is that an agent will learn from the environment by interacting with it (through trial and error) and receiving rewards (negative or positives) as feedback for performing actions.

>[!quote] A formal definition
>RL is a framework for solving control tasks (decision problem) by building an agent that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.

# 2. The Reinforcement Learning Framework

## 2.1 How does Reinforcement Learning work?

![[Pasted image 20251213153344.png]]

The RL loop outputs a sequence of **state, action, reward and next state.**

![[Pasted image 20251213153458.png]]

The agent's goal is to maximize its cumulative reward, called the expected return.

>[!quote] The reward hypothesis
>All goals can be described as the maximization of the expected return (expected cumulative reward).

## 2.2 Observations/States space

Observations/States are the information our agents get from the environment.

- State $s$: a complete description of the state of the world (no hidden information). The environment is fully observed.
	- E.g., In a chess game, we have access to the whole board information.
- Observation $o$: is a partial description of the sstate. The environment is partially observed.
	- E.g., In Super Mario Bros, we only see part of the level close to the player, so we receive an observation.

## 2.3 Action space.

The Action space is a set of all possible actions in an environment.

The actions can come from a discrete or continuous space:

- Discrete space: the number of possible actions is finite.
	- E.g., In Super Mario Bros, we have only 4 possible actions: left, right, up, down.
- Continuous space: the number of possible actions is infinite.
	- E.g., A Self Driving Car agent has an inifinite number of possible actions since it can turn left $20^o, 20.1^o, 20.2^o , \dots$

## 2.4 Rewards and the discounting.

![[Pasted image 20251213154357.png]]

The reward lets the agent know if the action taken was good or not.

Since immediate rewards are more predictable and certain than long-term rewards, future rewards are discounted to account for uncertainty in the distant future.

# 3. Type of tasks.

A task is an instance of Reinforcement Learning problem. 

## 3.1 Episodic tasks.

In this case, we have a starting point and an ending point (terminal state). This creates an episode: a list of States, Actions, Rewards and new States.

E.g., Super Mario Bros: an episode begins at the launch of a new Mario level and ends when you're die or you reached the end of the level.

## 3.2 Continuing tasks.

The tasks that continue forever (no termial state). 

E.g., an agent that does automated stock trading. There is no starting point and no ending point. The agent keeps running until we decide to stop it.

# 4. The Exploration/Exploitation trade-off

- Exploration is exploring the environment by trying random actions in order to find more information about the environment.
- Exploitation is exploiting known information to maximize the reward.

E.g., the choice of picking a restaurant:

- Exploitation: You go to the same one that you know is good everyday and take the risk to miss another better restaurant.
- Exploration: Try restaurants you never went to before, with the risk of having bad experience but the probable oppurtunity of a better experience.

# 5. Two main approaches for solving RL problems

## 5.1 The policy $\pi$ - the agent's brain

It's the function that tells the agent what action to take given the state. It defines the agent's behavior at a given time.

Our goal is to find the optimal policy $\pi^*$, the policy that maximizes the expected cumulative return when the agent act according to it. We find this $\pi^*$ through training.

There are two approaches for our agent to find this optimal policy $\pi^*$.

- Policy-based methods: teach the agent which action to take given a state.
- Value-based methods: teach the agent to learn which state is more valuable then take the action that leads to the more valuable states.

## 5.2 Policy-Based Methods

We learn a policy function directly.

This function will define a mapping from each state to the best corresponding action. Alternatively, it could define a probability distribution over the set of possible actions at that state.

**There are two types of policies:**

- **Deterministic:** a policy that, for a given state will always return the same action.
$$
a = \pi(s)
$$
>action = policy(state).

E.g., In Super Mario Bros, under a deterministic policy, when the agent encounters a creep in a given state, it will always take the same action (e.g., jumping), even though alternative actions (such as fighting back to collect a coin) are possible.

- **Stochastic:** outputs a probability distribution over the set of possible actions given the current state.
$$
\pi(a|s) = P[A|s]
$$
E.g., Given an initial state, our stochastic policy will output probability distribution over the possible action at that state. State $s_{0} \to \pi(A|s_{0}) \to$ {Left: 0.1, Right: 0.7, Jump: 0.2}.

## 5.3 Value-Based Methods.

Instead of learning a policy function, we learn a value function that maps a state to the expected value of being at that state.

The **value of a state** is the expected discounted return the agent can get if it starts in that state, and then acts according to our policy.

>"Act according to our policy" means that our policy is "going to the state with the highest value".

$$
v_{\pi}(s) = E_{\pi}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+2} + \dots |S_{t} = s]
$$

Value function = Expected discounted return given a state.

# 6. Deep Reinforcement Learning.

Using Deep Neural Networks to solve RL problems.

Two value-based algorithms:

- Q-Learning.
- Deep Q-Learning.



