---
tags:
  - RL
Date: 2025-10-27
Relevant:
---

# 1 Reinforcement Learning.

Involve learning what to do - how to map situations to actions to maximize a reward signal.

Learning system's actions influence its later inputs.

Learner isn't told which actions to take, it must discover which actions yield the most reward by trying them out.

Actions may affect the immediate reward and also the next sistuation or all subsequent rewards.

Three most important distinguish features of RL problems.
- Closed-loop in an essential way.
- Not having direct instructions as to what actions to take, and where the consequences of actions, including reward signals.
- Play out over extended time periods.

>[!quote] 
>Full specification of RL problems in terms of optimal control of Markow: Chapter 3.

Basic idea of RL: Capture the most important aspects of real problem - an agent interacting with its environment to achieve a goal.

An agent must be able to:
- Sense the state of the environment to some extent (scope).
- Take actions to affect the state.
- Must have a goal relating to the state of the environment.

RL is designed around 3 core components of intelligent behavior: sensation, action and goal - in their simplest possible form.

## 1.1.1 Challenges.

Trade-off between exploration and exploitation.

To obtain a lot of reward, a RL agent must prefer actions that it has tried in the past and found to be effective in producing reward.

But to discover such actions, it has to try actions that it has not selected before.

So, it has to _exploit_ what it already knows in order to obtain reward, but it also has to _explore_ to find if there are any other better selections in.

>[!important]
>The dilemma is that neither exploration nor exploitation can be pursued exclusively without failing at the task.

### 1.1.2 Key feature.

It explicitly considers **the whole problem** of a goal-directed agent interacting with an uncertain environment.

*Uncertain environment*: Environment that agent doesn't know the pattern; result of actions can be diverse each time.

Other researchers have developed theories of **Planning with general goals** but without considering planning's role in real-time decision making.

>For instance, they plan to find shortest path but in real-time decision making, agent doesn't know the map; it has to learn by trying, making mistakes, and drawing lessons; it has to make real-time decisions like turn right, turn left.

**RL takes the opposite site**. All RL agents have explicitly goals, can sense aspects of their environments, can choose actions to influence their environment.

It's assumed from the beginning that: agent has to operate despite uncertainty about the environment it faces.

When RL involves planning, it has to address the interplay/interaction between planning and real-time decision making (action selection).

> Sometimes RL can use a model of the environment to _plan ahead_ (Model-Based RL, Dyna-Q,...).
> The interplay is the consideration whether thinking (planning) or act immediately based on what it already know.

RL only used Supervised Learning when it has a specific reasons like **function approximation (like NN)**, imitation.

### 1.1.3 General intelligent.

1960-1980's researchers presumed that there are no **general principles** to be discovered, intelligent is just giving huge domain knowledge in a specific field.

RL represents the opposite philosophy: 

>It seeks for **general principles** that can explain and produce intelligent behavior across MANY DOMAINS. RL will learn from experience, optimize rewards and adapt to any environment.

## 1.2 Example.

1. The agent = the chess player  
	1. This agent will interact with the environment (chess board + opponent).
2. The action - the move the agent chooses.
3. Planning part $\to$ anticipate possible replies and counterreplies.
	1. Simulate the future: "If I move here, the opponent might respond like this... then ...". (**Model-based reasoning** - use an internal model).
4. Intuitive judgement part $\to$ intuitive prediction based on past experience (**Model-free RL** - using a learned value function).

## 1.3 Elements of RL.

Four main subelements:
- A policy.
- A value function.
- A reward signal.
- A model (optional).

A _policy_ defines the learning agent's way of behaving at a specific time. It's a mapping from perceived states of the environment to a specific action that should be taken in this environment. A _policy_ can be a function, a lookup table, a search process. A _policy_ alone is sufficient to determine a behavior. It may be stochastic (random).

A _reward signal_ defines a goal in RL problem. On each time step, the environments sends to the agent a _reward_.(For instance in chess game, a right move achieve opponent's queen -> environment's reward). The agent's sole objective is to _maximize the total reward_ over the long run. The reward signal thus defines what are the good and bad events for the agent. The reward sent to the agent at any time based on the agent's current action and the environment's current state. The agent **can not alter the process that does it**. The only way the agent can influence back the reward signal is through its actions. The reward signal is the primary basis for altering the policy. If an action selected by the policy is followed by low reward, the policy may be changed to choose another action in this situation in the future. (Turn right in environment's state A get reward 5, next time in environment's state A, turn left). In general, reward signals may be stochastic functions of the state of the environment and the actions taken. [[Appendix 1.]]

Whereas the reward signal indicates what is good in an immediate sense (intuitive sense), a _value function_ specifies what is good in the long run. The _value of a state_ is the total amount of reward an agent can expect to achieve over the future, starting from that state. Whereas rewards determine the immediate goodness of states, _values_ indicate the long-term goodness of states - the refined judgement. [[Appendix 2.]] TL; DR: _Values_ is just predictions of rewards in the future.

Rewards are the primary part, values - predictions of rewards are just secondary. Without rewards there could be no values, the only purpose of estimating values is to maximize reward. Nevertheless/Therefore, when making decisions we most concern about the _values_. Action choices are made based on _value judgements_. We seeks for actions that can help us attain the highest values, not just highest rewards because those actions obtain the highest rewards over the long run. [[Appendix 2.]] So, we can infer that the most important components among all RL algorithms **are** a method for efficiently estimating values. TL; DR: We are most concerned about the _values_ and all RL algorithms aim to efficiently **estimate** values.

The _model_ of the environment is something that mimics (imitates) the environment's behavior. It will infer how the environment will behave. E.g., given a state and a action, the model will predict the resultant next state and next reward. Use case: _planning_. It means, model will consider possible future situations before they are actually experienced. 

>Methods for solving RL problem using model and planning are called _model-based_ methods.
>Methods explicitly trial-and-error learners-the opposite of planning-are called _model-free_ methods.

## 1.4 Limitations and Scope.

Most considered RL methods in this book are structured around **estimating value function**.

>Methods don't estimate value function: genetic algorithms, genetic programming, simulated annealing - **Evolutionary method**.

**Evolutionary methods** evaluate the life-time behavior of many non-learning agents, each using different _policy_ for interacting with its environment and select those that are able to obtain most reward. [[Appendix 3.]]

Our focus: RL methods that involve learning while interacting with the environment.
Methods can take advantage of the details of individual behavioral **can** be much more efficient than evolutionary methods in **many cases**.

Scope: The term "RL method" doesn't contain "Evolutionary method".

**Policy gradient methods:** like evolutionary methods, do not appeal to value functions. These methods search in spaces of policies (all possible policies) defined by a collection of numerical parameters. They estimate the directions of the parameters should be adjusted to improve a policy's performance the most. They produce these estimates while the agent is interacting with its environment -> they can take advantage of the details of individual behavioral interactions. TL;DR: Adjust parameters to improve the policy's performance the most.

**Optimization & Optimality:** RL agent aims to maximize its rewards but it does not always reach the maximum rewards (policy is too hard to find, the agent doesn't know the environment fully, randomness, limited knowledge, environment complexity). So it's not guaranteed to reach the maximum.








