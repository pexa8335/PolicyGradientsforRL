---
tags:
  - RL
  - "#TemporalDifference"
Date: 2025-11-22
Relevant: "[[Chapter 1 - The RL Problem.]]"
---
First, to understand deeply, let's rewrite the formula:
$$
V(s) \leftarrow V(s) +\alpha[V(s') - V(s)] 
$$
$$
V(s) \leftarrow (1-\alpha)V(s) \times \alpha V(s')
$$
## 1. If $\alpha$ converges to zero.

We're converging to the true probability.
**Condition:** This is only true when the opponents won't change their strategy.

##  2. If $\alpha$ is not reduced to zero.

Assumption: We keep $\alpha=0.1$ forever.
This mean: New knowledge has 0.1 weight, old knowledge has 0.9 weight.

*   When the opponent changes strategy, losses begin.
*   $V(s')$ (actual result) drops.
*   Thanks to $\alpha = 0.1$, the $V(s)$ will slowly be dragged down towards the new reality.
*   Gradually "forgetting" the glorious past (when playing against a novice) and "updating" to the harsh reality (now playing against an expert).
*   $\Rightarrow$ Adaptation is achieved.



