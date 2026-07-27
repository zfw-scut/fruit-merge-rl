# daxigua_rl

This package contains the current automation, causal-attribution, and RL
training implementation.

Boundary rule:

- `daxigua` contains the playable game and must not import `daxigua_rl`.
- `daxigua_rl` may import stable interfaces from `daxigua`.
- Environments, adapters, agents, and training code live here instead of
  inside the game package.
- Visual playback scripts may lazily import `daxigua.app.Board` to drive the
  real pygame window, but training and environment code stay on the headless
  interface.

Current interface:

- `DaxiguaEnv`: gym-like wrapper around
  `daxigua.core.engine.HeadlessGame`. One RL `step(action_index)` is one fruit
  drop plus headless physics settling, not one rendered frame.
- `HeadlessGame.capture_snapshot()` / `restore_snapshot()`: versioned,
  checksummed `EngineSnapshot` support for complete stable-boundary Pymunk,
  episode, queue, RNG, and physics-config restoration. `DaxiguaEnv` can also
  be constructed from a compatible live snapshot.
- `RewardConfig`: Reward V2 parameters shared by task utility and potential
  shaping. Each real merge is valued as
  `2**((new_level - 2) / 2)` and state changes are shaped with
  `lambda_phi * (gamma * Phi(next) - Phi(previous))`, where
  `Phi = capacity_weight*C + recoverability_weight*R + chain_readiness_weight*K`.
  Survival time, maximum pile height, and continuous danger penalties are not
  rewards.
- `FruitState` and `ActionCandidate` expose both display radius and the actual
  Pymunk collision radius; graph geometry uses the collision radius.
- `TensorTransition`: the training-path experience record built from CPU
  `GraphTensor`; replay graph features use float16 to reduce resident memory.
  `bootstrap_steps` records the actual 1-to-3-step horizon. Its optional
  `structural_target` stores the selected action's masked six-dimensional,
  one-step structural outcome; it is not accumulated with n-step reward.
- Only `terminated` transitions disable DQN bootstrap; `truncated` transitions
  keep their final observation and next graph.
- `ReplayBuffer`: fixed-capacity in-memory or hot-memory/cold-disk replay. Its
  checkpoint contract restores the full in-memory ring or, for hybrid storage,
  the bounded hot layer and reports omitted cold items as an explicit
  hot-resume.
- `NStepTransitionAccumulator`: worker-local 3-step return builder; terminal
  and truncated tails are flushed with their real shorter horizon.
- `RolloutCollector` / `ParallelRolloutCollector`: assign stable
  `(worker_id, episode_id, step_index)` keys, play the headless environment,
  emit n-step transitions, and build rule-causal samples plus bounded
  counterfactual proposals. Parallel rollout can centralize greedy inference
  in a main-process GPU actor, micro-batching requests while epsilon-random
  actions remain worker-local.
- `StateAnalyzer` and worker-local `AttributionTracker`: analyze canonical
  15-action reachability, top-connected space, support/partner/motif
  structure, keep exact fruit lineage, enforce one value package per merge,
  and resolve delayed setup, rescue, blocking, burial, and terminal-support
  events. `GraphBuilder` projects only generalizable, current-boundary
  structural features and motifs into the main graph; full analysis objects,
  arbitrary IDs, and lineage histories do not enter the main replay.
- `GNNQNetwork`: relation-gated and attention-weighted message passing over
  explicit support/blocker/partner/motif edges, followed by a global-value
  plus centered action-advantage dueling readout. `forward_with_aux()` reuses
  the same encoding for six selected-action structural predictions.
- `CausalReplayBuffer`: separate bounded in-memory replay with
  `positive_setup`, `negative_blocking`, and `counterfactual` strata, carrying
  `rule`, `counterfactual`, or `shapley` pairwise Q supervision. Rule samples
  do not duplicate scalar environment reward.
- `CounterfactualProposalBuilder`: keeps a per-worker 32-boundary snapshot
  ring, joins delayed events to their originating transition, and emits
  compact proposals only for configured high-value or ambiguous cases.
- `CounterfactualCoordinator`: freezes the target policy only on target-network
  sync, schedules physical branch replays asynchronously, enforces the soft
  cost ratio and shared 10% hard token ledger, and inserts only successfully
  reproduced results into `CausalReplayBuffer`.
- `LocalShapleyCoordinator`: selects at most the configured event fraction,
  replays 2-to-4 local historical candidates with paired permutations and
  subset caching, checks grand-coalition reproduction and efficiency, and
  shares the counterfactual hard budget.
- `DQNTrainer`: Double DQN updater using
  `reward_n + gamma**bootstrap_steps * Q_target(s', argmax Q_online(s'))`.
  It combines TD, masked one-step structural, rule-ranking, and
  counterfactual/Shapley Huber losses and fails before `optimizer.step()` on
  non-finite values or gradients. Structural supervision does not change the
  environment reward.
- `daxigua_rl.training.checkpointing`: atomic, versioned checkpoint writer with
  run/config fingerprints, Python/PyTorch/CUDA RNG state, model/target/
  optimizer counters, replay component states, and strict resume validation.
- `daxigua_rl.scripts.train_dqn`: complete causal training entrypoint with CSV
  metrics, budget/reproduction diagnostics, atomic checkpointing, hot-resume,
  greedy evaluation, and matplotlib curves.
- `daxigua_rl.scripts.watch_dqn`: visual checkpoint viewer that drives the
  real pygame `Board` with a trained model.

`DaxiguaEnv.step(..., transition_key=...)` accepts the collector identity.
Direct callers may omit it and use the environment-local worker-0 identity.
Reward computation keeps terminal `next_state_analysis=None`, while
`post_action_state_analysis` is retained only for terminal causal attribution.
Full attribution requires the canonical 15 actions so analysis offsets,
drop columns, and Q-value indices remain identical.

Run the preflight gate from the project root before a long run:

```bash
PYTHONPATH=src conda run --no-capture-output -n python-torch python -u \
  tools/preflight_training.py --config configs/train_dqn_causal_250k.toml
```

The three formal-stage launch configurations and one optional extension inherit
one frozen algorithm/environment structure-aware V2 baseline:

- `train_dqn_causal_smoke_5k.toml`: integration smoke run;
- `train_dqn_causal_calibration_10k.toml`: scale calibration run;
- `train_dqn_causal_250k.toml`: first formal 250000-update large run;
- `train_dqn_causal_500k.toml`: optional future extension, not the current formal target.

Run training from the project root, for example:

```bash
PYTHONPATH=src conda run --no-capture-output -n python-torch python -u \
  -m daxigua_rl.scripts.train_dqn \
  --config configs/train_dqn_causal_smoke_5k.toml
```

Use `--no-capture-output` to see progress in real time through conda.

All first V2 gate runs start at update 0 in new empty directories. Do not
resume an H128/L3 checkpoint or reuse old hot/cold/causal replay: the graph
schema, relation-aware model, dueling readout, auxiliary head, and formal
H256/L4 dimensions are not training-compatible with those artifacts. See
`docs/rl/STRUCTURE_AWARE_GNN_V2.md` for the exact compatibility boundary.

Resume an interrupted run from a trusted checkpoint of that same V2 run:

```bash
PYTHONPATH=src conda run --no-capture-output -n python-torch python -u \
  -m daxigua_rl.scripts.train_dqn \
  --config configs/train_dqn_causal_250k.toml \
  --resume runs/<run>/checkpoints/latest.pt
```

Resume rejects semantic config drift. Output-only cadence, requested total
updates, and resume/run paths are the deliberately mutable fields. Hybrid TD
replay resumes from its saved hot layer; this is recorded in a resume sidecar
rather than presented as an exact cold-replay continuation.

Watch a trained checkpoint in the real game window:

```bash
PYTHONPATH=src conda run --no-capture-output -n python-torch python -u \
  -m daxigua_rl.scripts.watch_dqn \
  --checkpoint runs/<run>/checkpoints/latest.pt
```
