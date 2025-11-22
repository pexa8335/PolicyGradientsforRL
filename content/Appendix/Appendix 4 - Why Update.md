---
tags:
  - EvolutionStrategy
  - RL
Date: 2025-10-28
Relevant: "[[Chapter 1 - The RL Problem.]]"
---
# Why Update?

Therefore, upon executing a move from $c \to d$ (Greedy - the best selected move), it is discovered that $d$ holds an actual value (or a newer estimate). This new information ($V(d)$) is utilized to rectify the past error ($V(c)$).

*   At $c$, the prediction is: "Likely a 50% chance of winning".
*   Arriving at $d$, seeing a favorable position, the estimate is revised: "Actually, this indicates an 80% win".
*   **Contradiction:** 50% (past) vs 80% (present).
*   **Action:** Travel back in time (mentally) to signal to the previous state at $c$: "Hey, that spot is promising, rate it higher next time". $\rightarrow$ That is Update $V(c) \leftarrow V(d)$.
