# daxigua_rl

This package contains future automation and RL work.

Boundary rule:

- `daxigua` contains the playable game and must not import `daxigua_rl`.
- `daxigua_rl` may import stable interfaces from `daxigua`.
- Future environments, adapters, agents, and training code should live here instead of inside the game package.
- Visual playback scripts may lazily import `daxigua.app.Board` to drive the real pygame window, but training and environment code should stay on the headless interface.

Current v0 interface:

- `DaxiguaEnv`: gym-like wrapper around `daxigua.core.engine.HeadlessGame`.
- `RewardConfig`: configurable reward shaping for score, survival, height, danger, and terminal penalty.
- One RL `step(action_index)` means one fruit drop plus headless physics settling, not one rendered frame.
- Training and environment code must not import `daxigua.app.Board`, pygame renderers, HUD, audio, or manual input code.
- `TensorTransition`: current training-path experience record built from CPU `GraphTensor`; replay graph features are stored as float16 to reduce resident memory.
- `ReplayBuffer`: fixed-capacity in-memory buffer for storing and uniformly sampling experience records.
- `RolloutCollector`: single-process collector that plays the headless environment with epsilon-greedy actions and writes `TensorTransition` records into `ReplayBuffer`.
- `DQNTrainer`: standard DQN updater that samples tensor records, builds `GraphBatch`, computes TD loss, and updates the online Q network.
- `daxigua_rl.scripts.train_dqn`: first full DQN training entrypoint with CSV metrics, checkpoints, greedy evaluation, and matplotlib curves.
- `daxigua_rl.scripts.watch_dqn`: visual checkpoint viewer that drives the real pygame `Board` with a trained model.

Run training from the project root:

```bash
PYTHONPATH=src conda run --no-capture-output -n python-torch python -u -m daxigua_rl.scripts.train_dqn
```

Use `--no-capture-output` to see progress output in real time when running through conda.

Watch a trained checkpoint in the real game window:

```bash
PYTHONPATH=src conda run --no-capture-output -n python-torch python -u -m daxigua_rl.scripts.watch_dqn \
  --checkpoint runs/dqn_baseline_h128_l3_10k_eps10k/checkpoints/latest.pt
```
