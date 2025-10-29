---
tags:
  - EvolutionStrategy
  - RL
Date: 2025-10-28
Relevant: "[[Chapter 1 - The RL Problem.]]"
---
# 1.5 An Extended Example: Tic-Tac-Toe

Approaching the tic-tac-toe problem with a method making use of a value function. First, set up a table of number, one for each possible state of the game.
Each number will be the lastest estimate of our winning probability from that state. This estimate is called state's value, the whole table is value function.

- Each **state** = a board configuration (like Xs and Os in certain positions).
    
- **Value of a state** = current estimate of **probability of winning** if you play optimally from that state.

The table stores one number per state - representing how good that state is.

Look at the current state of the board.
Consider all possible next states, look up the value of each next state.
Choose the action that leads to the best next state. (The state with the highest value).

![[Pasted image 20251028174712.png]]

Solid line - the moves taken.
Dashed line - considered moves.
The second move was an exploratory move - the e* was ranked higher, this move aim to gather more information about its consequences. We only update the **value of the state** that led to an action that we intended to exploit.
