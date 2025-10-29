---
tags:
  - RL
  - EvolutionStrategy
Date: 2025-10-28
Relevant: "[[Chapter 1 - The RL Problem.]]"
---
Evolution strategy evaluate the overall performance of an agent after it completes an entire life.

Evolutionary algorithms don’t try to update *values* or *policies* after each action.

- **Create a population** of agents (each with a different policy or strategy).
- **Let each agent live its full life** — run through an entire episode.
- **Measure the total reward** (the “fitness” or “lifetime performance”) for each agent.
- **Select** the best-performing agents and **mutate/recombine** their policies to form the next generation.

**Lifetime behavior** = the _entire sequence of actions and outcomes_ an agent experiences during one episode (from start to end).

If the space of policies (all possible policies) is small or good policies are easy to find or if a lot of time is available for the search -> evolutionary methods can be effective.

Evolutionary methods have advantages on problem in which agent can't accurately sense the state of its environment.

