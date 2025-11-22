---
tags:
  - "#Markov"
  - "#RL"
Date: 2025-11-22
Relevant: "[[Chapter 1 - The RL Problem.]]"
---
In this chapter, the author introduce the problem that they try to solve in the rest of the book. This problem defines the field of Reinforcement Learning: any method that is suited to solving this problem is considered a reinforcement learning method.

# 1. The Agent-Environment Interface

The RL problem is meant to be a straightforward formulation of the problem of learning from interaction to achieve a goal.

*Agent* is the learner and the decision-maker. The things it interacts with, comprising everything outside the agent, is called *the environment*. The agent selecting actions and the environment responding to those actions and presenting new situations to the agent. The environment also generates rewards - a special numerical values that the agent tries to maximize over time. A complete specification of an environment defines a *task*, one instance of the RL problem. This means **changing** the environment is equivalent to **changing** the *task*. 

For instance, in Chess:
 *   Rule 1: Defeating the King yields 100 points $\to$ Task: Playing standard Chess.
 *   Rule 2: Capturing a pawn yields 100 points $\to$ Task: Capturing as many pawns as possible.

>The author uses the term *Agent, Environment, Action* instead of *Controller, Controlled System (plant), Control Signal*.

![[Pasted image 20251122225228.png]]

More specifically, the agent and the environment interact at each of a sequence of discrete time steps, $t = 0, 1, 2, 3\dots$ At each time step $t$, the agent receives some representations of the environment's *state*, $S_{t} \in S$ where $S$ is the set of possible states, and on that basis selects an *action*, $A_{t} \in A(S_{t})$, where $A(S_{t})$ is the set of possible actions in the state $S_{t}$. One time step later, in part as a consequence of its action, the agent receives a numerical *reward*, $R_{t+1} \in \mathcal{R}  \subset \mathbb{R}$ and finds itself at a new state $S_{t+1}$.

>**TL;DR:** At each time step $t$, the agent receives a state representation $S_t \in S$, and selects an action $A_t$ from $A(S_t)$ — the set of all available actions in that state. One time step later, as a consequence, the agent receives a reward $R_{t+1}$ and transitions to a new state $S_{t+1}$.

At each step, the agent implements a mapping from states to probability of selecting each possible action. This mapping is what we call *agent's policy* and is denoted as $\pi_{t}$, where $\pi_{t}(a|s)$ is the probability that $A_{t} = a$ and $S_{t} = s$. RL method specify how the agent changes its policy as a result of its experience. The agent's goal is to maximize its total reward it receives over the long run.
