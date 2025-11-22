---
tags:
  - RL
  - Markov
Date: 2025-11-22
Relevant: "[[Chapter 3 - Finite Markov Decision Processes]]"
---
# 1. Why using the word "Representations"?

Environment's State (Reality): Is everything in the universe of that environment. For example: The robot is walking, the real state includes wind speed, tire wear, road temperature...
Representation (What the Agent sees): The Agent cannot know everything. It only sees what the sensor captures (Camera, GPS).

# 2. The notation explanation.

$\mathbb{R}$ is the set of all real number. It means the reward can be a number (0, 1, 5.5,...).
$\mathcal{R}$ is set of all possible rewards in this specific task.

For instance:
In chess:
 * $\mathbb{R}$: Infinite numbers.
 * $\mathcal{R}$: Rule of chess: Only {1, 0, 0.5} (Win, Lose, Draw).
 * Then $R_{t+1} \subset$ {1, 0, 0.5} 

1. Discrete Time Steps: $t=0, 1, 2...$ This means the game is played in turns (Turn-based), not a continuous flow like water. After step 0, go to step 1.
2. Constraint Actions ($A(St)$): Actions depend on the state.
	* For example: If $St$ is "Mario is on the ground", then $A(St)$ has "Jump".
	* If $St$ is "Mario is swimming", then $A(St)$ does not have "Jump" but has "Dive".
3. Delayed Consequence $t+1$ trap:
	* You take action at time $t$ ($A_t$).
	* But the result ($R$ and new $S$) you get at a **future** time $t+1$.

$\Rightarrow$ Action is the **Cause**, Reward/Next State is the **Result**.


