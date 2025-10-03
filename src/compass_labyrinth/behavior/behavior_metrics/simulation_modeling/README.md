# Simulation Modelling

A module designed for simulating agent-based performance in maze environments and benchmarking task performance against behavioral data. This package supports flexible simulation of decision-making agents (random, binary, or multi-option) navigating a complex labyrinth maze with decision points and reward paths.

---

## Objective

This module provides a framework to:

- Simulate randomized decision-making across all decision nodes
- Use observed mouse behavior to define available action sets
- Quantify performance metrics under unbiased decision scenarios
- Compare simulated vs. actual behavior to evaluate strategic advantage

---

## Simulation Logic

- At each decision node, the agent is presented with **the same set of choices the real mouse took** at that node across sessions
- The agent randomly selects one of these observed actions — eliminating bias but preserving biological feasibility
- Simulations are run across all decision nodes per session

This method creates a biologically grounded null model to estimate whether observed mouse performance exceeds chance-level outcomes given the same constraints.

