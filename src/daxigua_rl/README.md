# daxigua_rl

This package contains future automation and RL work.

Boundary rule:

- `daxigua` contains the playable game and must not import `daxigua_rl`.
- `daxigua_rl` may import stable interfaces from `daxigua`.
- Future environments, adapters, agents, and training code should live here instead of inside the game package.

Current v0 interface:

- `DaxiguaEnv`: gym-like wrapper around `daxigua.core.engine.HeadlessGame`.
- One RL `step(action_index)` means one fruit drop plus headless physics settling, not one rendered frame.
- This package must not import `daxigua.app.Board`, pygame renderers, HUD, audio, or manual input code.
- `Transition`: framework-independent training experience record built from `GraphData`, action offset, reward, next graph, and done flags.
- `ReplayBuffer`: fixed-capacity in-memory buffer for storing and uniformly sampling `Transition` records.
- `RolloutCollector`: single-process collector that plays the headless environment with epsilon-greedy actions and writes `Transition` records into `ReplayBuffer`.
