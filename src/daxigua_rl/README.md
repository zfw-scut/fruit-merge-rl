# daxigua_rl

This package is reserved for future automation and RL work.

Boundary rule:

- `daxigua` contains the playable game and must not import `daxigua_rl`.
- `daxigua_rl` may import stable interfaces from `daxigua`.
- Future environments, adapters, agents, and training code should live here instead of inside the game package.
