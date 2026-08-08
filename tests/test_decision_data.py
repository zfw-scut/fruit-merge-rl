"""训练期事实采集和未来派生监督框架测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from daxigua.rl.config import (
    AnalysisExportConfig,
    AutoScaleConfig,
    DashboardConfig,
    DecisionDataConfig,
    DqnConfig,
    EvaluationConfig,
    ModelConfig,
    ReplayConfig,
    RewardConfig,
    TrainingConfig,
)
from daxigua.rl.decision_data import (
    ActionSelectionBatch,
    DecisionSelectionBatch,
    DerivedSupervisionBatch,
)
from daxigua.rl.key_decisions import KeyDecisionCollector
from daxigua.rl.observations import TensorState
from daxigua.rl.replay import GpuReplayBuffer
from daxigua.rl.trainer import BaselineTrainer
from daxigua.simulator import SimulatorConfig, TensorVectorSimulator


class _FirstRowSelector:
    active = True

    def __init__(self, capacity):
        self.capacity = int(capacity)

    def prepare(self, context):
        return context.action_selection.q_values[:, :2].clone()

    def select(self, context):
        self.assert_prepared_shape = tuple(context.prepared.shape)
        rows = torch.zeros(
            self.capacity, dtype=torch.int64, device=context.rewards.device
        )
        valid = torch.zeros_like(rows, dtype=torch.bool)
        valid[0] = True
        priorities = torch.zeros_like(rows, dtype=torch.float32)
        priorities[0] = 2.0
        reasons = torch.zeros_like(rows)
        reasons[0] = 4
        return DecisionSelectionBatch(rows, valid, priorities, reasons)


def _state(simulator):
    return TensorState.from_observation(
        simulator.observe(), physics_fps=simulator.config.physics_fps
    )


class ReplayReferenceTest(unittest.TestCase):
    def test_default_replay_does_not_allocate_versioned_references(self):
        simulator = TensorVectorSimulator(
            2,
            config=SimulatorConfig(max_fruits=64, use_cuda_extension=False),
            device='cpu',
        )
        replay = GpuReplayBuffer(2, max_fruits=64, device='cpu')
        ticket = replay.begin_append(_state(simulator))

        self.assertFalse(replay.state_references_enabled)
        self.assertFalse(replay.metadata()['versioned_state_references'])
        with self.assertRaises(RuntimeError):
            _ = ticket.reference

    def test_versioned_reference_detects_overwritten_slots(self):
        simulator = TensorVectorSimulator(
            2,
            config=SimulatorConfig(max_fruits=64, use_cuda_extension=False),
            device='cpu',
        )
        state = _state(simulator)
        replay = GpuReplayBuffer(
            2,
            max_fruits=64,
            device='cpu',
            enable_state_references=True,
        )
        first = replay.begin_append(state)
        replay.finish_append(
            first,
            state,
            torch.tensor((0, 1)),
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
        )
        self.assertTrue(bool(replay.gather_current(first.reference).valid.all()))

        second = replay.begin_append(state)
        replay.finish_append(
            second,
            state,
            torch.tensor((1, 2)),
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
        )
        self.assertFalse(bool(replay.gather_current(first.reference).valid.any()))
        self.assertTrue(bool(replay.gather_current(second.reference).valid.all()))


class DecisionCollectorTest(unittest.TestCase):
    def _pipeline(
            self,
            run_dir,
            *,
            archive=True,
            gpu_capacity=4,
            device='cpu'):
        simulator = TensorVectorSimulator(
            2,
            config=SimulatorConfig(max_fruits=64, use_cuda_extension=False),
            device=device,
        )
        replay = GpuReplayBuffer(
            8,
            max_fruits=64,
            device=device,
            physics_fps=30,
            enable_state_references=True,
        )
        config = DecisionDataConfig(
            enabled=True,
            max_candidates_per_step=2,
            gpu_retention_capacity=gpu_capacity,
            archive_enabled=archive,
            archive_shard_records=1,
            archive_queue_size=2,
        )
        selector = _FirstRowSelector(config.max_candidates_per_step)
        collector = KeyDecisionCollector(
            config,
            replay=replay,
            simulator=simulator,
            run_dir=Path(run_dir),
            selector=selector,
        )
        return simulator, replay, collector, selector

    def test_empty_selector_keeps_configured_framework_inactive(self):
        with TemporaryDirectory() as directory:
            simulator = TensorVectorSimulator(
                1,
                config=SimulatorConfig(
                    max_fruits=64, use_cuda_extension=False
                ),
                device='cpu',
            )
            replay = GpuReplayBuffer(2, max_fruits=64, device='cpu')
            collector = KeyDecisionCollector(
                DecisionDataConfig(enabled=True),
                replay=replay,
                simulator=simulator,
                run_dir=Path(directory),
            )
            self.assertTrue(collector.configured)
            self.assertFalse(collector.active)
            self.assertFalse(
                (Path(directory) / 'analysis' / 'decision_facts').exists()
            )
            collector.close()

    def test_existing_trainer_runs_with_injected_collection_framework(self):
        model = ModelConfig(
            hidden_dim=16,
            edge_hidden_dim=16,
            message_layers=1,
            queue_hidden_dim=8,
            queue_layers=1,
            level_embedding_dim=4,
            max_neighbors=2,
            nearest_neighbors=1,
            motion_neighbors=1,
            vertical_neighbors_per_direction=1,
            action_key_fruits=1,
        )
        with TemporaryDirectory() as directory:
            config = TrainingConfig(
                run_dir=directory,
                device='cpu',
                seed=123,
                max_envs=2,
                active_envs=2,
                total_transitions=4,
                max_episode_drops=4,
                stage_pilot_envs=2,
                stage_pilot_max_drops=2,
                model=model,
                dqn=DqnConfig(
                    use_bfloat16=False,
                    fused_adam=False,
                ),
                reward=RewardConfig(kind='score_v1'),
                replay=ReplayConfig(
                    capacity=4,
                    batch_size=2,
                    warmup_transitions=2,
                    warmup_stage_ratios=(1.0, 0.0, 0.0, 0.0),
                ),
                evaluation=EvaluationConfig(
                    periodic_episodes=1,
                    final_episodes=1,
                    parallel_envs=1,
                    max_episode_drops=1,
                ),
                analysis=AnalysisExportConfig(
                    transition_sample_size=0,
                    trajectory_episodes=0,
                    critical_event_episodes=0,
                ),
                decision_data=DecisionDataConfig(
                    enabled=True,
                    max_candidates_per_step=1,
                    archive_enabled=True,
                    archive_shard_records=1,
                ),
                dashboard=DashboardConfig(enabled=False),
                autoscale=AutoScaleConfig(enabled=False),
            )
            trainer = BaselineTrainer(
                config,
                decision_selector=_FirstRowSelector(1),
            )
            result = trainer.run(final_evaluation=False)
            archive = Path(directory) / 'analysis' / 'decision_facts'

            self.assertEqual(result['transitions'], 4)
            self.assertTrue((Path(directory) / 'checkpoints' / 'final.pt').exists())
            self.assertTrue((archive / 'manifest.json').exists())
            shards = list(archive.glob('decision_facts_*.pt'))
            self.assertEqual(len(shards), 1)
            payload = torch.load(shards[0], weights_only=False)
            self.assertEqual(payload['record_count'], 1)

    def test_collector_keeps_gpu_facts_and_writes_filtered_shard(self):
        with TemporaryDirectory() as directory:
            simulator, replay, collector, selector = self._pipeline(directory)
            current = _state(simulator)
            q_values = torch.arange(42, dtype=torch.float32).view(2, 21)
            action_selection = ActionSelectionBatch(
                actions=torch.tensor((3, 5)),
                greedy_actions=torch.tensor((20, 20)),
                explore_mask=torch.tensor((True, False)),
                q_values=q_values,
            )
            ticket = replay.begin_append(current)
            staged = collector.stage_pre(
                current=current,
                action_selection=action_selection,
                ticket=ticket,
                environment_rows=torch.arange(2),
                transition_start=100,
                policy_version=7,
            )
            actions = action_selection.actions
            result = simulator.step(actions)
            next_state = TensorState.from_observation(
                result.observation, physics_fps=30
            )
            rewards = result.physics.score_delta.to(torch.float32) / 66.0
            stages = torch.tensor((1, 2), dtype=torch.uint8)
            replay.finish_append(
                ticket,
                next_state,
                actions,
                rewards,
                result.physics.done,
                stages,
            )
            facts = collector.observe_post(
                staged,
                result=result,
                next_state=next_state,
                rewards=rewards,
                stages=stages,
                episode_finished=result.physics.done,
            )
            collector.close()

            self.assertEqual(selector.assert_prepared_shape, (2, 2))
            self.assertEqual(facts.batch_size, 2)
            self.assertEqual(facts.valid_mask.tolist(), [True, False])
            self.assertEqual(facts.decision_ids.tolist(), [100, 100])
            self.assertEqual(facts.policy_versions.tolist(), [7, 7])
            self.assertEqual(facts.reason_bits.tolist(), [4, 0])
            self.assertEqual(facts.action_selection.actions.tolist(), [3, 3])
            self.assertTrue(bool(facts.replay_reference.generations.gt(0).all()))
            self.assertIsNotNone(collector.gpu_buffer.pop())

            archive = Path(directory) / 'analysis' / 'decision_facts'
            manifest = archive / 'manifest.json'
            shards = list(archive.glob('decision_facts_*.pt'))
            self.assertTrue(manifest.exists())
            self.assertEqual(len(shards), 1)
            payload = torch.load(shards[0], weights_only=False)
            self.assertEqual(payload['record_count'], 1)
            records = payload['records']
            self.assertEqual(
                records['identity']['decision_ids'].tolist(), [100]
            )
            self.assertEqual(records['action']['q_values'].shape, (1, 21))
            self.assertEqual(
                records['pre_sidecar']['rng_state'].shape, (1,)
            )
            self.assertEqual(
                records['outcome']['physics']['merge_events']['source_ids']
                .shape[0],
                1,
            )

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_archive_completes_nonblocking_copy_before_release(self):
        with TemporaryDirectory() as directory:
            simulator, replay, collector, _ = self._pipeline(
                directory,
                gpu_capacity=0,
                device='cuda',
            )
            current = _state(simulator)
            q_values = torch.arange(
                42, dtype=torch.float32, device='cuda'
            ).view(2, 21)
            action_selection = ActionSelectionBatch(
                actions=torch.tensor((3, 5), device='cuda'),
                greedy_actions=torch.tensor((20, 20), device='cuda'),
                explore_mask=torch.tensor(
                    (True, False), device='cuda'
                ),
                q_values=q_values,
            )
            ticket = replay.begin_append(current)
            staged = collector.stage_pre(
                current=current,
                action_selection=action_selection,
                ticket=ticket,
                environment_rows=torch.arange(2, device='cuda'),
                transition_start=200,
                policy_version=9,
            )
            result = simulator.step(action_selection.actions)
            next_state = TensorState.from_observation(
                result.observation, physics_fps=30
            )
            rewards = result.physics.score_delta.to(torch.float32) / 66.0
            stages = torch.zeros(2, dtype=torch.uint8, device='cuda')
            replay.finish_append(
                ticket,
                next_state,
                action_selection.actions,
                rewards,
                result.physics.done,
                stages,
            )
            collector.observe_post(
                staged,
                result=result,
                next_state=next_state,
                rewards=rewards,
                stages=stages,
                episode_finished=result.physics.done,
            )
            collector.close()

            shard = next(
                (Path(directory) / 'analysis' / 'decision_facts').glob(
                    'decision_facts_*.pt'
                )
            )
            records = torch.load(shard, weights_only=False)['records']
            self.assertEqual(
                records['action']['q_values'][0].tolist(),
                list(map(float, range(21))),
            )

    def test_sidecar_is_not_mutated_by_following_step(self):
        with TemporaryDirectory() as directory:
            simulator, replay, collector, _selector = self._pipeline(
                directory, archive=False
            )
            current = _state(simulator)
            selection = ActionSelectionBatch(
                actions=torch.tensor((0, 1)),
                greedy_actions=torch.tensor((0, 1)),
                explore_mask=torch.zeros(2, dtype=torch.bool),
                q_values=torch.zeros((2, 21)),
            )
            ticket = replay.begin_append(current)
            staged = collector.stage_pre(
                current=current,
                action_selection=selection,
                ticket=ticket,
                environment_rows=torch.arange(2),
                transition_start=0,
                policy_version=0,
            )
            before_rng = staged.pre_sidecar.rng_state.clone()
            simulator.step(selection.actions)
            self.assertTrue(torch.equal(
                staged.pre_sidecar.rng_state, before_rng
            ))
            self.assertFalse(torch.equal(
                staged.pre_sidecar.rng_state, simulator.rng_state
            ))
            collector.close()


class DerivedSupervisionContractTest(unittest.TestCase):
    def test_variable_payload_keeps_identity_and_masks(self):
        batch = DerivedSupervisionBatch(
            task_type='future_auxiliary_target',
            producer_version='test-v1',
            information_scope='agent_observation',
            decision_ids=torch.tensor((10, 11)),
            segment_ids=torch.tensor((-1, 4)),
            plan_ids=torch.tensor((-1, -1)),
            policy_versions=torch.tensor((3, 3)),
            valid_mask=torch.tensor((True, False)),
            confidence=torch.tensor((1.0, 0.5)),
            payload={
                'variable_trajectory': torch.zeros((2, 5, 2)),
                'trajectory_mask': torch.tensor((
                    (True, True, False, False, False),
                    (True, False, False, False, False),
                )),
            },
        )
        self.assertEqual(batch.format_version, 1)
        self.assertEqual(batch.payload['variable_trajectory'].shape, (2, 5, 2))

    def test_payload_rejects_a_different_batch_dimension(self):
        with self.assertRaisesRegex(ValueError, 'first dimension'):
            DerivedSupervisionBatch(
                task_type='bad',
                producer_version='test-v1',
                information_scope='agent_observation',
                decision_ids=torch.tensor((1, 2)),
                segment_ids=torch.tensor((-1, -1)),
                plan_ids=torch.tensor((-1, -1)),
                policy_versions=torch.tensor((0, 0)),
                valid_mask=torch.tensor((True, True)),
                confidence=torch.ones(2),
                payload={'bad': torch.zeros((3, 2))},
            )


if __name__ == '__main__':
    unittest.main()
