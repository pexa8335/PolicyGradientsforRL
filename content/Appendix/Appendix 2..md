---
tags:
  - RL
Date: 2025-10-28
Relevant: "[[Chapter 1 - The RL Problem.]]"
---
Learning -> tired, boring -> low reward but high value.
Surfing tiktok -> happy, chill -> high reward but low value.

So value function should formalize a familiar idea: A state can yield (bring) a low immediate reward but still have a high value because it is followed by other states that yield high rewards.

| Concept            | Comparison with humans                                 | Nature                                                      |
| ------------------ | ------------------------------------------------------ | ----------------------------------------------------------- |
| **Reward**         | Feeling of pleasure / pain _right now_                 | Short-term, immediate                                       |
| **Value**          | Smart assessment, looking ahead                        | Long-term, future aggregate                                 |
| **Value Function** | Brain decides: “endure now, will be happy later”       | Formalizes this idea using expected future rewards          |

It is much harder to determine values than it is to determine rewards. Because rewards are given directly by the environment, but values must be estimated over and over from sequences of observations.