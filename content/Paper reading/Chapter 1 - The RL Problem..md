---
tags:
  - RL
Date: 2025-10-27
Relevant:
---

# 1 The Essence of Reinforcement Learning.

Reinforcement Learning (RL) involves learning what to do - specifically, how to map situations to actions to maximize a numerical reward signal.

Unlike other forms of learning, the learner is not instructed on which actions to take. Instead, it must discover which actions yield the most reward via **trial-and-error.**

Crucially, RL is distinct because the actions taken by the agent influence not just the immediate reward, but also the next state and, consequently, all subsequent rewards.

Three distinguishing features of RL:
1. Closed-loop interaction: The system learns while interacting; inputs depend on previous outputs.
2. No direct instruction: The agent learns from consequences (Rewards), not from correct examples provided by a teacher.
3. Temporal consequences: It operates over extended time periods, where a decision made now can affect the future state.

>[!quote] 
>We will fully specify the RL problem in terms of Optimal Control of Markov Decision Processes in [[Chapter 3 - Finite Markov Decision Processes]]

**The core concept** 
RL aims to capture the most important aspects of real-world problems: an agent interacting with its environment to achieve a goal. To function effectively, an agent must be able to: 

- **Perceive** the state of the environment to some extent (Sensation).
- **Take actions** to affect the state (Action).
- **Pursue a goal** related to the state of the environment (Goal).

In summary, RL is designed around three core components of intelligent behavior: sensation, action and goal - in their simplest possible forms.

## 1.1 Challenges.

One of the most critical challenges in RL, which distinguishes it from other learning paradigms, is the trade-off between exploration and exploitation.

To maximize reward, an RL agent must prefer actions that it has tried in the past and found to be effective (**Exploitation**) in producing reward. However, to discover such actions in the first place, it has to try actions that it has not selected before (**Exploration**).

The agent must **exploit** known actions to maximize current reward, but must also **explore** new actions to discover potentially superior strategies

>[!important]
>**The dilemma:** Neither exploration nor exploitation can be pursued exclusively without failing at the task. The agent must try a variety of actions and progressively favor those that appear to be the best.

## 1.2 The Holistic Approach.

Another key feature of RL is that it explicitly considers **the whole problem** of a goal-directed agent interacting with an uncertain environment.

>[!info] Concept: *Uncertain environment* 
>An environment is considered "uncertain" (or stochastic) when the agent cannot perfectly predict the outcome of its actions.

**Planning vs. Real-time action.**

Other researchers have developed theories of **Planning with general goals**, but without considering the role of planning in real-time decision making.

- Limitation: They might plan to find the shortest path assuming a perfect map.
- Reality: In RL, the agent often does not know the map; it has to learn by trying, making mistakes, and drawing lessons. It must make real-time decisions (like "turn right" or "turn left") under uncertainty.

**The RL Approach.**

RL takes the opposite approach. It assumes from the beginning that the agent must operate despite uncertainty about the environment it faces. All RL agents have explicitly goals, can sense aspects of their environments, can choose actions to influence their environment.

When RL involves planning (e.g., Model-Based RL, Dyna-Q), it has to address the interplay between planning (thinking ahead) and acting (real-time decision making).

> Sometimes RL can use a model of the environment to _plan ahead_ (Model-Based RL, Dyna-Q,...).
> The interplay is the consideration whether thinking (planning) or act immediately based on what it already know.

RL only utilized Supervised Learning for specific sub-tasks (like Deep Learning for function approximation), but the overarching learning paradigm is distinct.

## 1.3 General intelligence.

Since the late 1960s, many AI researchers presumed that there were no **general principles** of intelligence to be discovered. They believed intelligence was simply a matter of providing massive amounts of domain-specific knowledge

RL represents the opposite philosophy: 

>It seeks **general principles** that can explain and produce intelligent behavior across MANY DOMAINS. RL aims to learn from experience, optimize rewards and adapt to any environment.

## 1.4 Example: The Chess Player

Consider a chess match to visualize these components:

1. The Agent: The chess player.
2. The Environment: The chess board, the pieces, and the opponent.
3. The Mechanisms:
	- **Planning (Model-based):** The agent simulates the future using an internal model ("If I move here, the opponent might reply with X...").
	- **Intuition (Model-free):** The agent makes a fast, intuitive judgment based on past experience on this environment ("This board configuration feels dangerous"), essentially using a learned Value Function.

## 1.5 Elements of Reinforcement Learning

Beyond the agent and the environment, an RL system is defined by four main sub-elements: a policy, a reward signal, a value function, and optionally, a model.

### 1.5.1 Policy ($\pi$).

A _policy_ defines the learning agent's way of behaving at a specific time. Formally, it's a mapping from perceived states of the environment to a specific action that should be taken in this environment. 

- It can be a function, a simple lookup table, a search process. 
- A _policy_ alone is sufficient to determine a behavior. 
- In many cases (especially in Policy Gradients), the *policy* is stochastic (randomized), specifying the probabilities for each action rather than a single deterministic command.

### 1.5.2 Reward Signal ($R_{t}$)

A _reward signal_ defines a goal of the RL problem. On each time step, the environment sends to the agent a _reward_.

- **Objective:** The agent's sole objective is to _maximize the total reward_ over the long run. 
- **Nature:** The reward signal thus defines what are the good and bad events for the agent. For example, in chess, capturing a piece might yield a positive reward, while losing a piece yields a negative one. The reward sent to the agent at any time based on the agent's current action and the environment's current state. 
- **Constraint:** The agent cannot directly change the reward generation mechanism (the rules of the game). The only way the agent can influence the reward signal is through its actions (playing better). 

The reward signal is the primary basis for altering the policy. If an action selected by the policy is followed by low reward, the policy may be changed to choose another action in this situation in the future.

**Key concept: Stochastic Rewards** 
In many real-world problems, the reward function is stochastic (not deterministic). This means the output (reward) is not always the same even if we take the same action in the exact same state - it follows a probability distribution.

$$
R_{t}∼P(r∣S_{t},A_{t})
$$

Analogy: Imagine playing a Slot Machine.

- **State:** Standing in front of the machine.
- **Action:** Pulling the lever.
- **Reward:** The money won.  
    Even if you pull the lever the same way (same action), the reward is random (sometimes $0, sometimes $100). The agent must learn to maximize the expected reward despite this noise.

### 1.5.3 Value function ($V(s)$)

Whereas the reward signal indicates what is good in an immediate sense, a _value function_ specifies what is good in the long run. 
The _value of a state_, denoted as $V(s)$, is the total amount of reward an agent can expect to achieve over the future, starting from that state. 

>[!quote] **Intuition: The "TikTok" vs. "Studying" Analogy**  
> A state can yield a low immediate reward but still have a high value because it leads to better states later.
> - **Studying:** Low Immediate Reward (Tired, boring) $\to$ **High Value** (Leads to knowledge, career stability).
> - **Surfing TikTok:** High Immediate Reward (Fun, dopamine hit) $\to$ **Low Value** (Leads to wasted time, missed deadlines).  
>**Key Takeaway:** We seek actions that bring the highest **Value** (Long-term), not just the highest immediate **Reward** (Short-term).

**Comparison: Reward vs. Value**

| Concept            | Nature                       | Human Equivalent                                                |
| :----------------- | :--------------------------- | :-------------------------------------------------------------- |
| **Reward ($R_t$)** | Short-term, Immediate.       | The feeling of pleasure or pain *right now*.                    |
| **Value ($V(s)$)** | Long-term, Future Aggregate. | Smart assessment; looking ahead ("Endure now, be happy later"). |
>The Value Function formalizes the familiar human idea of "enduring pain now for happiness later.

**The Central Importance of Value**
Without rewards, there could be no values; however, for decision-making, we are most concerned with **values**. Action choices are made based on value judgments. We seek actions that lead to states of highest value, because these actions ultimately maximize the accumulated reward over time.

**The Challenge of Estimation**
Unfortunately, determining values is much harder than determining rewards.
*   **Rewards** are given directly by the environment.
*   **Values** must be **estimated** and continuously re-estimated from sequences of observations.

$\Rightarrow$ Therefore, the most important component of almost all RL algorithms is a method for efficiently **estimating values**.

### 1.5.4 Model.

The _model_ of the environment is something that mimics (imitates) the environment's behavior. It will infer how the environment will behave. 

- **Function:** E.g., given a state and a action, the model will predict the resultant next state and next reward. 
- **Use case:** Models are used for planning. It means, model will consider possible future situations before they are actually experienced. 

>[!info] Categorization
>- **Model-based method:** Methods that use models and planning (E.g., Dyna-Q).
>- **Model-free methods:** Methods that learn explicitly via trial-and-error without a model (e.g., Q-Learning, and crucially, **Standard Policy Gradients**).

## 1.6 Limitations and Scope.

It is crucial to distinguish Reinforcement Learning from **Evolutionary Methods** (e.g., Genetic Algorithms, Simulated Annealing), as they share similar goals but differ fundamentally in execution.

**How Evolutionary Methods Work:**
Instead of learning step-by-step, these methods apply a "survival of the fittest" approach:

1. **Population:** Create a population of agents, each with a different fixed policy or strategy.
2. **Lifetime evaluation:** Let each agent run through an entire episode to measure the total cumulative reward (fitness).
3. **Selection & Mutation:** Select the highest-performing agents and modify/recombine their policies to form the next generation.

**Lifetime behavior** refers to the _entire sequence of actions and outcomes_ an agent experiences during one episode (from start to end).

**The key difference:**
**Evolutionary Methods** ignore the sequential structure of the episode. They assign credit to the entire policy based on the final outcome (Win/Loss), ignoring whether specific actions within the episode were good or bad.
In contrast, **RL methods** leverage the details of individual interactions. Because they learn **during** the episode, RL methods are generally much **sample-efficient** in complex problems where state information is available.

>**Scope:** The term "Reinforcement Learning" refers to methods that learn while interacting, thus excluding pure Evolutionary Methods.

**Policy Gradient methods** occupy a unique middle ground. Like evolutionary methods, they search directly in the policy space (optimizing parameter $\theta$).
However, unlike evolutionary methods, they utilize interaction details to formalize gradient of performance with respect to the policy parameters. This allow them to:

1. Estimate how the parameters should be adjusted to improve performance.
2. Update the policy based on specific interactions (states and actions) rather than just final outcome.

**Optimization & Optimality.**
While an RL agent aims to maximize rewards, it does not always reach the theoretical **global maximum** (Optimality).

- **Reasons:** Determining the perfect policy is often computationally intractable due to partial observability, stochasticity, or limited resources.
- **Practical Goal:** We often settle for finding a policy that is "good enough" (near-optimal) given the constraints.


## 1.7 An Extended Example: Tic-Tac-Toe

To illustrate the mechanics of RL, consider the game of Tic-Tac-Toe.
*   **Goal:** Win by placing three marks in a row.
*   **Assumption:** We are playing against an **imperfect player**.
*   **Objective:** We want to construct an agent that learns to exploit the opponent's imperfections to maximize the probability of winning.

![[Untitled-2025-11-25-1121.png]]

If the board fills up with neither player getting three in a row, the game is a draw. 

### 1.7.1 Why not Classical methods?

- **Minimax:** It assumes the opponent always plays optimally. It would play too defensively against a novice opponent, missing chances to win quickly.
- **Dynamic programming:** Fails because it requires a perfect model of the opponent (specifying exact probabilities of their moves), which we do not have.
- **Evolutionary methods:** These methods would search directly through the space of possible policies. Here, a policy is a rule set that dictates the move for every possible board configurations (Xs and Os). To evaluate just one policy, the agent must play many games to estimate its winning probability. Crucially, this ignores the internal structure of the game - it learns from the final 'Win/Loss' but fails to recognize which move caused the victory or defeat.

### 1.7.2 The Value Function Approach

Instead, we use a **Value Function** approach. We set up a table of numbers, one for each possible state of the game - representing how good that state is.

*   State ($s$): A specific board configuration.
*   Value ($V(s)$):** The estimate of the probability of winning starting from state $s$.

**The strategy:** 
The agent looks at the current state, consider all possible next states, looks up their values and usually chooses the move that leads to the state with the highest value (**Greedy move**). Occasionally, it chooses a random move to explore (**Exploratory move**).

### 1.7.3 The Learning Mechanism (Temporal Difference)

How do we update the values? We use a method called **Temporal Difference (TD) Learning**.

When the agent moves from state $s$ to a new state $s'$ (and finds $s'$ to be in better position), the value of the earlier state is updated to be closer to the value of the later state.

**The Update Rule:**
$$
V(s) \leftarrow V(s) + \alpha[V(s') - V(s)]
$$
- $s$: The state before the move.
- $s'$: The state after the move.
- $V(s)$: The estimated value of the state before the move.
- $V(s')$: The estimated value of the state after the move.
- $\alpha$ is a small positive fraction called *step-size parameter* ($0 \leq \alpha \leq 1$).

This method is called **Temporal Difference (TD) Learning** because it learns from the difference between estimates at two different times.

>[!info] Intuition: Why Update? ("Backing up")
>Imagine moving from state s $\to$ s'.
>- At s, we estimated a 50% chance of winning.
>- Arriving at s', we see a strong position, estimating an 80% chance.
>- **Contradiction:** "I thought $s$ was average (50%) but it led me to a greater spot (80%)"
>- **Update:** The agent adjusts $V(s)$ upwards to be closer to 80%.

**Mathematical Insight: Weighted Average**

We can rewrite the udpate formula to see the role of $\alpha$:
$$
V(s) \leftarrow (1-\alpha)V(s) + \alpha V(s')
$$
1. If $\alpha \to 0$ (Decaying): The method converges to the true probability of winning (assuming the opponent is fixed).
2. If $\alpha$ is constant (e.g., $\alpha =0.1$):
	- New information has 10% weight, old information has 90%.
	- **Benefit:** If the opponent changes their strategy, the agent never fully converges. It keeps adapting, slowly "forgetting" the past to match the new reality.

### 1.7.4 Conclusion

This example highlights the core advantage of RL (Value Functions) over Evolutionary methods.

- **Evolutionary:** Assigns credit to the entire game based on the final result.
- **RL (Value Function):** Evaluates **individual states**. It leverages the sequential structure of the game to learn step-by-step, identifying exactly which moves were critical for the victory.

