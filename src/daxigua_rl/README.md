# daxigua_rl

This package contains future automation and RL work.

Boundary rule:

- `daxigua` contains the playable game and must not import `daxigua_rl`.
- `daxigua_rl` may import stable interfaces from `daxigua`.
- Future environments, adapters, agents, and training code should live here instead of inside the game package.
- Visual playback scripts may lazily import `daxigua.app.Board` to drive the real pygame window, but training and environment code should stay on the headless interface.

Current v0 interface:

- `DaxiguaEnv`: gym-like wrapper around `daxigua.core.engine.HeadlessGame`.
- `RewardConfig`: Reward V2 parameters shared by task utility and potential shaping.
- Reward V2 values each real merge as `2**((new_level - 2) / 2)` and shapes
  state changes with `lambda_phi * (gamma * Phi(next) - Phi(previous))`, where
  `Phi = capacity_weight*C + recoverability_weight*R + chain_readiness_weight*K`.
  Survival time, maximum pile height, and continuous danger penalties are not rewards.
- One RL `step(action_index)` means one fruit drop plus headless physics settling, not one rendered frame.
- `FruitState` and `ActionCandidate` expose both display radius and the actual Pymunk collision radius; graph geometry uses the collision radius.
- Training and environment code must not import `daxigua.app.Board`, pygame renderers, HUD, audio, or manual input code.
- `TensorTransition`: current training-path experience record built from CPU `GraphTensor`; replay graph features are stored as float16 to reduce resident memory.
- Only `terminated` transitions disable DQN bootstrap; `truncated` transitions keep their final observation and next graph.
- `ReplayBuffer`: fixed-capacity in-memory buffer for storing and uniformly sampling experience records.
- `RolloutCollector`: single-process collector that assigns `(worker_id, episode_id, step_index)` keys, plays the headless environment with epsilon-greedy actions, and writes `TensorTransition` records into `ReplayBuffer`.
- `DQNTrainer`: standard DQN updater that samples tensor records, builds `GraphBatch`, computes TD loss, and updates the online Q network.
- `daxigua_rl.attribution`: frozen, tuple-only state/event contracts,
  `StateAnalyzer`, and worker-local `AttributionTracker`. The analyzer computes
  15-action reachability, top-connected space, support/partner/motif structure;
  the tracker consumes actual drop/merge transitions, keeps exact fruit
  lineage, enforces one value package per merge, resolves delayed burial
  incidents, and emits realized setup/rescue/terminal-support events.
  Full analyses, lineage, and events remain worker-local and are not written to
  the main replay.
- `DaxiguaEnv.step(..., transition_key=...)` accepts the collector's stable
  `(worker_id, episode_id, step_index)` identity. Direct callers may omit it and
  use the environment-local worker-0 identity.
- Reward computation keeps terminal `next_state_analysis=None`, while
  `post_action_state_analysis` is retained only for terminal causal
  attribution.
- Full attribution requires the canonical 15 environment actions so action
  offsets, analyzed drop columns, and Q-value indices remain identical.
- Training writes attribution metrics to `metrics.csv`, a separate
  `attribution_warmup.json`, and an `attribution_shutdown.json` pending-event
  audit.
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
