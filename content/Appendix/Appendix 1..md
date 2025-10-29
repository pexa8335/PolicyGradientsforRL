---
tags:
  - RL
Date: 2025-10-28
Relevant: "[[Chapter 1 - The RL Problem.]]"
---
1. **"Stochastic function"** → means **not deterministic**.  
The output (reward) is **not always the same** even if we take the same action in the same state — it follows a _probability distribution_.

2. So, mathematically we write:

$$
R_{t}∼P(r∣S_{t},A_{t})
$$

→ meaning:  
At time $t$, the reward $R_{t}$ is randomly chosen from a probability distribution that depends on the current state $S_{t}$ and action taken $A_{t}$.

---
Imagine playing a slot machine
- **State**: the machine chooses.
- **Action**: pulling the lever.
- **Reward**: the money won.

Even if the lever is pulled (same action) on the same machine (same state), the **reward is random** — sometimes $0, sometimes $10, sometimes $100.