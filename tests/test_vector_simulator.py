"""批量 Tensor/CUDA 物理模拟器契约测试。"""

import importlib.util
import unittest


try:
    import torch
except ImportError:  # 核心规则仍可在无 PyTorch 环境中独立测试。
    torch = None


if torch is not None:
    from daxigua.core import merged_fruit_physics_radius
    from daxigua.simulator import (
        GameScoreReward,
        SimulatorConfig,
        SingleEnvAdapter,
        TensorVectorSimulator,
        trace_to_payload,
        VectorEnv,
        ZeroReward,
    )


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class TensorVectorSimulatorTest(unittest.TestCase):
    def _config(self, **overrides):
        values = {
            'board_width': 320,
            'board_height': 420,
            'spawn_y': 80,
            'wall_width': 20,
            'action_count': 7,
            'max_fruits': 8,
            'physics_fps': 60,
            'max_physics_frames': 180,
            'stable_frames': 4,
            'solver_iterations': 2,
            'sync_interval_frames': 4,
            'use_cuda_extension': False,
        }
        values.update(overrides)
        return SimulatorConfig(**values)

    def _install_fruit(
            self,
            simulator,
            env_index,
            slot,
            level,
            x,
            y,
            fruit_id):
        device = simulator.device
        env_ids = torch.tensor([env_index], dtype=torch.int64, device=device)
        slots = torch.tensor([slot], dtype=torch.int64, device=device)
        levels = torch.tensor([level], dtype=torch.int64, device=device)
        radius = simulator._dropped_radii[levels]
        simulator.positions[env_index, slot] = torch.tensor(
            [x, y], dtype=torch.float32, device=device
        )
        simulator.velocities[env_index, slot] = 0
        simulator.levels[env_index, slot] = level
        simulator.physics_radii[env_index, slot] = radius[0]
        simulator.fruit_ids[env_index, slot] = fruit_id
        simulator.age_frames[env_index, slot] = 0
        simulator.active[env_index, slot] = True
        simulator._set_mass_properties(env_ids, slots, levels, radius)
        simulator.next_fruit_id[env_index] = max(
            int(simulator.next_fruit_id[env_index].item()),
            fruit_id + 1,
        )

    def test_reset_is_reproducible_and_partial_reset_is_isolated(self):
        simulator = TensorVectorSimulator(
            4, config=self._config(), device='cpu'
        )
        first_queue = simulator.fruit_queue.clone()
        simulator.reset(seeds=0)
        self.assertTrue(torch.equal(first_queue, simulator.fruit_queue))
        before = simulator.fruit_queue.clone()
        simulator.reset(
            torch.tensor([False, True, False, False]),
            seeds=123,
        )
        self.assertTrue(torch.equal(before[0], simulator.fruit_queue[0]))
        self.assertTrue(torch.equal(before[2:], simulator.fruit_queue[2:]))
        self.assertTrue(
            ((simulator.fruit_queue >= 1) & (simulator.fruit_queue <= 5))
            .all()
            .item()
        )

    def test_action_positions_follow_current_fruit_radius(self):
        simulator = TensorVectorSimulator(
            1, config=self._config(), device='cpu'
        )
        simulator.reset(seeds=1, fruit_queue=[3, 1, 1, 1])
        positions = simulator.action_positions()[0]
        self.assertEqual(float(positions[0]), 64.0)
        self.assertEqual(float(positions[-1]), 256.0)
        self.assertAlmostEqual(float(positions[3]), 160.0)

    def test_batch_step_advances_queues_and_returns_stable_states(self):
        simulator = TensorVectorSimulator(
            4, config=self._config(), device='cpu'
        )
        queue = torch.ones((4, 4), dtype=torch.int64)
        simulator.reset(seeds=torch.arange(4), fruit_queue=queue)
        result = simulator.step(torch.tensor([0, 2, 4, 6]))
        self.assertTrue(result.physics.stable.all().item())
        self.assertFalse(result.physics.done.any().item())
        self.assertFalse(result.physics.truncated.any().item())
        self.assertTrue((result.observation.fruit_count == 1).all().item())
        self.assertTrue((result.observation.step_count == 1).all().item())
        self.assertTrue(
            (result.observation.fruit_queue[:, :3] == 1).all().item()
        )
        final_x = result.observation.positions[..., 0][
            result.observation.active
        ]
        self.assertEqual(len(torch.unique(final_x)), 4)

    def test_kinematic_rest_correction_is_per_fruit(self):
        simulator = TensorVectorSimulator(
            1,
            config=self._config(
                kinematic_rest_frames=4,
                kinematic_rest_displacement_epsilon=0.1,
            ),
            device='cpu',
        )
        self._install_fruit(simulator, 0, 0, 1, 100, 300, 1)
        self._install_fruit(simulator, 0, 1, 2, 200, 300, 2)
        simulator.age_frames[0, :2] = 1
        simulator.velocities[0, 0, 1] = 43.0
        simulator.velocities[0, 1, 1] = 43.0
        running = torch.tensor([True])
        quiet_frames = torch.zeros_like(simulator.levels)

        for _ in range(4):
            frame_start = simulator.positions.clone()
            simulator.positions[0, 1, 0] += 1.0
            simulator._correct_kinematic_rest_velocity(
                running, frame_start, quiet_frames
            )

        self.assertTrue(torch.equal(
            simulator.velocities[0, 0], torch.zeros(2)
        ))
        self.assertEqual(float(simulator.velocities[0, 1, 1]), 43.0)
        self.assertEqual(int(quiet_frames[0, 0]), 4)
        self.assertEqual(int(quiet_frames[0, 1]), 0)

    def test_angular_velocity_uses_same_time_damping_as_linear_velocity(self):
        config = self._config(
            physics_fps=30,
            gravity_y=0.0,
            damping=0.81,
        )
        simulator = TensorVectorSimulator(1, config=config, device='cpu')
        self._install_fruit(simulator, 0, 0, 1, 100, 200, 1)
        simulator.velocities[0, 0, 0] = 10.0
        simulator.angular_velocities[0, 0] = 10.0

        simulator._integrate(torch.tensor([True]))

        expected = 10.0 * config.damping ** config.dt
        self.assertAlmostEqual(
            float(simulator.velocities[0, 0, 0]), expected, places=5
        )
        self.assertAlmostEqual(
            float(simulator.angular_velocities[0, 0]), expected, places=5
        )

    def test_kinematic_rest_threshold_has_frame_rate_independent_speed(self):
        for physics_fps in (30, 120):
            config = self._config(
                physics_fps=physics_fps,
                kinematic_rest_frames=4,
                kinematic_rest_displacement_epsilon=0.1,
            )
            simulator = TensorVectorSimulator(1, config=config, device='cpu')
            self._install_fruit(simulator, 0, 0, 1, 100, 200, 1)
            simulator.age_frames[0, 0] = 1
            simulator.velocities[0, 0, 0] = 10.0
            running = torch.tensor([True])
            quiet_frames = torch.zeros_like(simulator.levels)
            for _ in range(4):
                frame_start = simulator.positions.clone()
                simulator.positions[0, 0, 0] += 10.0 * config.dt
                simulator._correct_kinematic_rest_velocity(
                    running, frame_start, quiet_frames
                )
            self.assertEqual(int(quiet_frames[0, 0]), 4)
            self.assertTrue(torch.equal(
                simulator.velocities[0, 0], torch.zeros(2)
            ))

    def test_newly_merged_fruit_is_not_frozen_in_creation_frame(self):
        config = self._config(kinematic_rest_frames=1)
        simulator = TensorVectorSimulator(1, config=config, device='cpu')
        self._install_fruit(simulator, 0, 0, 1, 100, 200, 1)
        simulator.velocities[0, 0, 0] = 10.0
        frame_start = simulator.positions.clone()
        quiet_frames = torch.zeros_like(simulator.levels)

        simulator._correct_kinematic_rest_velocity(
            torch.tensor([True]), frame_start, quiet_frames
        )

        self.assertEqual(int(quiet_frames[0, 0]), 0)
        self.assertEqual(float(simulator.velocities[0, 0, 0]), 10.0)

    def test_low_speed_contact_does_not_apply_restitution(self):
        config = self._config(restitution_velocity_threshold=35.0)
        simulator = TensorVectorSimulator(2, config=config, device='cpu')
        for env_index, speed in enumerate((10.0, 100.0)):
            self._install_fruit(
                simulator, env_index, 0, 1, 100, 200, env_index + 1
            )
            simulator.velocities[env_index, 0, 1] = speed
        penetration = torch.zeros_like(simulator.physics_radii)
        penetration[:, 0] = 1.0

        simulator._apply_wall_contact(
            penetration, (0.0, -1.0), torch.tensor([True, True])
        )

        self.assertAlmostEqual(float(simulator.velocities[0, 0, 1]), 0.0)
        self.assertAlmostEqual(
            float(simulator.velocities[1, 0, 1]),
            -100.0 * config.fruit_elasticity,
            places=4,
        )

    def test_adaptive_substeps_skip_free_fall_and_refine_deep_contact(self):
        config = self._config(
            physics_fps=30,
            adaptive_collision_substeps=True,
            max_collision_substeps=4,
        )
        simulator = TensorVectorSimulator(2, config=config, device='cpu')
        self._install_fruit(simulator, 0, 0, 1, 100, 100, 1)
        simulator.velocities[0, 0, 1] = 500.0
        self._install_fruit(simulator, 1, 0, 1, 100, 200, 1)
        self._install_fruit(simulator, 1, 1, 2, 120, 200, 2)

        counts = simulator._collision_substep_counts(
            torch.tensor([True, True])
        )

        self.assertEqual(int(counts[0]), 1)
        self.assertEqual(int(counts[1]), 4)

    def test_deterministic_matching_and_chain_merges(self):
        simulator = TensorVectorSimulator(
            1, config=self._config(), device='cpu'
        )
        simulator.reset(seeds=1)
        for slot, fruit_id in enumerate(range(1, 5)):
            self._install_fruit(
                simulator, 0, slot, 1, 160, 220, fruit_id
            )
        running = torch.tensor([True])
        simulator._resolve_merges(running)
        self.assertEqual(int(simulator.score[0]), 2)
        self.assertEqual(int(simulator.active[0].sum()), 2)
        simulator._resolve_merges(running)
        self.assertEqual(int(simulator.score[0]), 5)
        self.assertEqual(int(simulator.active[0].sum()), 1)
        active_level = simulator.levels[0][simulator.active[0]]
        self.assertEqual(int(active_level[0]), 3)
        self.assertEqual(int(simulator._event_count[0]), 3)

    def test_merge_conserves_linear_and_angular_momentum(self):
        simulator = TensorVectorSimulator(
            1, config=self._config(), device='cpu'
        )
        simulator.reset(seeds=1)
        self._install_fruit(simulator, 0, 0, 1, 140, 220, 1)
        self._install_fruit(simulator, 0, 1, 1, 180, 220, 2)
        simulator.velocities[0, 0] = torch.tensor([30.0, 4.0])
        simulator.velocities[0, 1] = torch.tensor([-6.0, 10.0])
        simulator.angular_velocities[0, 0] = 2.0
        simulator.angular_velocities[0, 1] = -1.0

        positions = simulator.positions[0, :2].clone()
        velocities = simulator.velocities[0, :2].clone()
        masses = simulator.masses[0, :2].clone()
        inertias = simulator.inverse_inertias[0, :2].reciprocal()
        midpoint = positions.mean(dim=0)
        source_momenta = masses[:, None] * velocities
        linear_before = source_momenta.sum(dim=0)
        radii = positions - midpoint
        angular_before = (
            inertias * simulator.angular_velocities[0, :2]
        ).sum() + (
            radii[:, 0] * source_momenta[:, 1]
            - radii[:, 1] * source_momenta[:, 0]
        ).sum()

        simulator._resolve_merges(torch.tensor([True]))

        target_slot = int(torch.nonzero(
            simulator.active[0], as_tuple=False
        )[0])
        linear_after = (
            simulator.masses[0, target_slot]
            * simulator.velocities[0, target_slot]
        )
        target_inertia = simulator.inverse_inertias[
            0, target_slot
        ].reciprocal()
        angular_after = (
            target_inertia
            * simulator.angular_velocities[0, target_slot]
        )
        self.assertTrue(torch.allclose(
            linear_after, linear_before, atol=1e-5, rtol=1e-5
        ))
        self.assertTrue(torch.allclose(
            angular_after, angular_before, atol=1e-4, rtol=1e-5
        ))
        self.assertGreater(
            float(simulator.velocities[0, target_slot].norm()), 0.0
        )

    def test_watermelons_score_and_disappear_without_target(self):
        simulator = TensorVectorSimulator(
            1, config=self._config(), device='cpu'
        )
        simulator.reset(seeds=1)
        self._install_fruit(simulator, 0, 0, 11, 160, 220, 1)
        self._install_fruit(simulator, 0, 1, 11, 160, 220, 2)
        simulator._resolve_merges(torch.tensor([True]))
        self.assertEqual(int(simulator.score[0]), 66)
        self.assertEqual(int(simulator.active[0].sum()), 0)
        events = simulator._merge_event_view().to_python(0)
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0].new_level)
        self.assertIsNone(events[0].new_fruit_id)

    def test_settle_timeout_keeps_environment_running(self):
        simulator = TensorVectorSimulator(
            2,
            config=self._config(max_physics_frames=1, stable_frames=2),
            device='cpu',
        )
        result = simulator.step(torch.tensor([0, 1]))
        first_positions = simulator.positions.clone()
        self.assertTrue(result.physics.settle_timeout.all().item())
        self.assertFalse(result.physics.truncated.any().item())
        self.assertFalse(result.physics.done.any().item())
        self.assertFalse(result.observation.done.any().item())
        self.assertFalse(simulator.needs_reset.any().item())

        second = simulator.step(torch.tensor([1, 2]))
        self.assertTrue(second.physics.settle_timeout.all().item())
        self.assertTrue((simulator.step_count == 2).all().item())
        self.assertTrue((second.observation.fruit_count == 2).all().item())
        self.assertTrue(
            (simulator.positions[:, 0] != first_positions[:, 0]).any().item()
        )

    def test_danger_line_is_a_true_termination(self):
        simulator = TensorVectorSimulator(
            1,
            config=self._config(
                max_physics_frames=10,
                stable_frames=2,
                danger_seconds=0.0,
            ),
            device='cpu',
        )
        simulator.reset(seeds=1, fruit_queue=[1, 1, 1, 1])
        self._install_fruit(simulator, 0, 0, 2, 80, 60, 1)
        result = simulator.step(torch.tensor([3]))
        self.assertTrue(result.physics.done[0].item())
        self.assertFalse(result.physics.truncated[0].item())
        self.assertTrue(result.observation.done[0].item())

    def test_capacity_exhaustion_is_never_silent(self):
        simulator = TensorVectorSimulator(
            1, config=self._config(max_fruits=2), device='cpu'
        )
        self._install_fruit(simulator, 0, 0, 1, 80, 300, 1)
        self._install_fruit(simulator, 0, 1, 2, 240, 300, 2)
        with self.assertRaisesRegex(RuntimeError, 'max_fruits'):
            simulator.step(torch.tensor([3]))

    def test_training_fast_profile_preserves_wait_time_scale(self):
        config = SimulatorConfig.training_fast(max_fruits=8)
        self.assertEqual(config.physics_fps, 30)
        self.assertEqual(config.max_physics_frames, 180)
        self.assertEqual(config.stable_frames, 4)
        self.assertTrue(config.drop_fast_forward)
        self.assertTrue(config.adaptive_collision_substeps)
        self.assertEqual(config.max_collision_substeps, 2)
        self.assertEqual(config.kinematic_rest_frames, 1)
        self.assertAlmostEqual(config.position_correction, 0.9)
        self.assertAlmostEqual(
            config.max_physics_frames / config.physics_fps, 6.0
        )
        high_fidelity = SimulatorConfig.high_fidelity_fast(max_fruits=8)
        self.assertEqual(high_fidelity.physics_fps, 120)
        self.assertEqual(high_fidelity.max_physics_frames, 720)
        self.assertTrue(high_fidelity.drop_fast_forward)

    def test_drop_fast_forward_skips_only_stable_free_fall(self):
        config = self._config(
            physics_fps=30,
            max_physics_frames=180,
            stable_frames=4,
            drop_fast_forward=True,
        )
        simulator = TensorVectorSimulator(2, config=config, device='cpu')
        simulator.reset(
            seeds=torch.arange(2),
            fruit_queue=torch.ones((2, 4), dtype=torch.int64),
        )
        self._install_fruit(simulator, 1, 0, 1, 80, 340, 1)
        simulator.velocities[1, 0, 0] = (
            config.stable_velocity_epsilon + 1.0
        )

        result = simulator.step(torch.tensor([3, 6]))

        self.assertGreater(int(result.physics.fast_forwarded_frames[0]), 0)
        self.assertEqual(int(result.physics.fast_forwarded_frames[1]), 0)
        self.assertTrue(torch.all(
            result.physics.fast_forwarded_frames
            <= result.physics.frames_simulated
        ).item())

    def test_single_environment_adapter_preserves_python_contracts(self):
        simulator = TensorVectorSimulator(
            1, config=self._config(), device='cpu'
        )
        adapter = SingleEnvAdapter(VectorEnv(simulator, ZeroReward()))
        state, info = adapter.reset(seed=1, fruit_queue=[1, 1, 1, 1])
        self.assertEqual(state.fruit_count, 0)
        self.assertEqual(len(info['action_candidates']), 7)
        state, reward, terminated, truncated, info = adapter.step(3)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(state.fruit_count, 1)
        self.assertEqual(info['drop_result'].dropped_level, 1)

    def test_game_score_reward_is_explicit_and_vectorized(self):
        simulator = TensorVectorSimulator(
            1, config=self._config(), device='cpu'
        )
        reward = GameScoreReward()
        self.assertFalse(reward.requires_previous_state)
        vector_env = VectorEnv(simulator, reward)
        _obs, rewards, _done, _truncated, _info = vector_env.step(
            torch.tensor([3])
        )
        self.assertEqual(float(rewards[0]), 0.0)


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(),
    'CUDA is not available',
)
class CudaVectorSimulatorTest(unittest.TestCase):
    def _install_fruit(
            self, simulator, slot, level, x, y, fruit_id):
        device = simulator.device
        env_ids = torch.tensor([0], dtype=torch.int64, device=device)
        slots = torch.tensor([slot], dtype=torch.int64, device=device)
        levels = torch.tensor([level], dtype=torch.int64, device=device)
        radius = simulator._dropped_radii[levels]
        simulator.positions[0, slot] = torch.tensor(
            [x, y], dtype=torch.float32, device=device
        )
        simulator.velocities[0, slot] = 0
        simulator.levels[0, slot] = level
        simulator.physics_radii[0, slot] = radius[0]
        simulator.fruit_ids[0, slot] = fruit_id
        simulator.active[0, slot] = True
        simulator._set_mass_properties(env_ids, slots, levels, radius)
        simulator.next_fruit_id[0] = max(
            int(simulator.next_fruit_id[0].item()), fruit_id + 1
        )

    def test_cuda_repeatability_for_fixed_seed_and_actions(self):
        config = SimulatorConfig(
            board_width=320,
            board_height=420,
            spawn_y=80,
            action_count=7,
            max_fruits=8,
            physics_fps=60,
            max_physics_frames=180,
            stable_frames=4,
            solver_iterations=2,
        )
        first = TensorVectorSimulator(32, config=config, device='cuda')
        second = TensorVectorSimulator(32, config=config, device='cuda')
        queue = torch.ones((32, 4), dtype=torch.int64, device='cuda')
        seeds = torch.arange(32, device='cuda')
        first.reset(seeds=seeds, fruit_queue=queue)
        second.reset(seeds=seeds, fruit_queue=queue)
        actions = torch.arange(32, device='cuda').remainder(
            config.action_count
        )
        first_result = first.step(actions)
        second_result = second.step(actions)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(
                first_result.observation.positions,
                second_result.observation.positions,
            )
        )
        self.assertTrue(
            torch.equal(
                first_result.physics.frames_simulated,
                second_result.physics.frames_simulated,
            )
        )

    def test_cuda_drop_fast_forward_reports_skipped_semantic_frames(self):
        config = SimulatorConfig.training_fast(max_fruits=8)
        simulator = TensorVectorSimulator(8, config=config, device='cuda')
        simulator.reset(
            seeds=9,
            fruit_queue=torch.ones((8, 4), dtype=torch.int64, device='cuda'),
        )

        result, trace = simulator.step_with_trace(
            torch.arange(8, device='cuda').remainder(config.action_count),
            [0],
            frame_stride=4,
        )
        torch.cuda.synchronize()

        self.assertTrue(
            (result.physics.fast_forwarded_frames > 0).all().item()
        )
        self.assertTrue(torch.all(
            result.physics.fast_forwarded_frames
            <= result.physics.frames_simulated
        ).item())
        executed = (
            result.physics.frames_simulated
            - result.physics.fast_forwarded_frames
        )
        self.assertTrue(torch.all(
            result.physics.collision_substeps >= executed
        ).item())
        self.assertTrue(torch.any(
            result.physics.collision_substeps > executed
        ).item())
        count = int(trace.record_counts[0])
        expected_capacity = (
            (config.max_physics_frames + 4 - 1) // 4 + 2
        )
        self.assertEqual(trace.frame_numbers.shape[1], expected_capacity)
        self.assertEqual(int(trace.frame_numbers[0, 0]), 0)
        self.assertEqual(
            int(trace.frame_numbers[0, count - 1]),
            int(result.physics.frames_simulated[0]),
        )

    def test_cuda_two_identical_drops_merge(self):
        config = SimulatorConfig(
            board_width=320,
            board_height=420,
            spawn_y=80,
            action_count=7,
            max_fruits=8,
            physics_fps=60,
            max_physics_frames=180,
            stable_frames=4,
            solver_iterations=2,
        )
        simulator = TensorVectorSimulator(16, config=config, device='cuda')
        queue = torch.ones((16, 4), dtype=torch.int64, device='cuda')
        simulator.reset(seeds=1, fruit_queue=queue)
        actions = torch.full((16,), 3, dtype=torch.int64, device='cuda')
        first = simulator.step(actions)
        self.assertTrue(first.physics.stable.all().item())
        second = simulator.step(actions)
        torch.cuda.synchronize()
        self.assertTrue((second.physics.score_delta == 1).all().item())
        self.assertTrue((second.observation.fruit_count == 1).all().item())
        self.assertTrue((second.observation.max_level == 2).all().item())

    def test_cuda_merge_inherits_linear_momentum(self):
        config = SimulatorConfig(
            board_width=560,
            board_height=1120,
            spawn_y=80,
            action_count=21,
            max_fruits=8,
            max_physics_frames=1,
            stable_frames=2,
            solver_iterations=2,
        )
        simulator = TensorVectorSimulator(1, config=config, device='cuda')
        simulator.reset(seeds=1, fruit_queue=[1, 1, 1, 1])
        self._install_fruit(simulator, 0, 1, 240, 500, 1)
        self._install_fruit(simulator, 1, 1, 280, 500, 2)
        source_velocity = torch.tensor(
            [12.0, -3.0], dtype=torch.float32, device='cuda'
        )
        simulator.velocities[0, 0] = source_velocity
        simulator.velocities[0, 1] = source_velocity

        result = simulator.step(torch.tensor([0], device='cuda'))
        torch.cuda.synchronize()

        integrated_velocity = source_velocity.clone()
        integrated_velocity[1] += config.gravity_y * config.dt
        integrated_velocity *= config.damping ** config.dt
        expected_velocity = integrated_velocity * (
            2.0 * float(simulator._mass_table[1])
            / float(simulator._mass_table[2])
        )
        target_mask = result.observation.active[0] & (
            result.observation.levels[0] == 2
        )
        self.assertEqual(int(target_mask.sum()), 1)
        actual_velocity = result.observation.velocities[0][target_mask][0]
        self.assertTrue(torch.allclose(
            actual_velocity, expected_velocity, atol=1e-4, rtol=1e-5
        ))

    def test_cuda_masked_step_leaves_disabled_environments_unchanged(self):
        config = SimulatorConfig(
            board_width=320,
            board_height=420,
            spawn_y=80,
            action_count=7,
            max_fruits=8,
            physics_fps=60,
            max_physics_frames=180,
            stable_frames=4,
            solver_iterations=2,
        )
        simulator = TensorVectorSimulator(8, config=config, device='cuda')
        simulator.reset(seeds=5)
        enabled = torch.tensor(
            [True, True, True, True, False, False, False, False],
            dtype=torch.bool,
            device='cuda',
        )
        disabled_queue = simulator.fruit_queue[~enabled].clone()
        disabled_rng = simulator.rng_state[~enabled].clone()
        simulator.needs_reset[7] = True
        result = simulator.step_masked(
            torch.zeros(8, dtype=torch.int64, device='cuda'), enabled
        )
        self.assertTrue((simulator.step_count[enabled] == 1).all().item())
        self.assertTrue((simulator.step_count[~enabled] == 0).all().item())
        self.assertTrue(
            (result.physics.frames_simulated[~enabled] == 0).all().item()
        )
        self.assertTrue(
            torch.equal(simulator.fruit_queue[~enabled], disabled_queue)
        )
        self.assertTrue(torch.equal(simulator.rng_state[~enabled], disabled_rng))

    def test_cuda_masked_trace_only_advances_enabled_environments(self):
        config = SimulatorConfig(
            board_width=320,
            board_height=420,
            spawn_y=80,
            action_count=7,
            max_fruits=8,
            physics_fps=60,
            max_physics_frames=180,
            stable_frames=4,
            solver_iterations=2,
        )
        simulator = TensorVectorSimulator(8, config=config, device='cuda')
        simulator.reset(
            seeds=11,
            fruit_queue=torch.ones((8, 4), dtype=torch.int64, device='cuda'),
        )
        enabled = torch.tensor(
            [True, False, True, False, True, False, True, False],
            dtype=torch.bool,
            device='cuda',
        )
        result, trace = simulator.step_masked_with_trace(
            torch.arange(8, dtype=torch.int64, device='cuda').remainder(7),
            enabled,
            [2, 6],
            frame_stride=3,
        )
        trace = trace.cpu()

        self.assertEqual(trace.env_indices.tolist(), [2, 6])
        self.assertTrue((simulator.step_count[enabled] == 1).all().item())
        self.assertTrue((simulator.step_count[~enabled] == 0).all().item())
        self.assertTrue(
            (result.physics.frames_simulated[~enabled] == 0).all().item()
        )
        for row, env_index in enumerate((2, 6)):
            count = int(trace.record_counts[row])
            self.assertGreaterEqual(count, 2)
            self.assertEqual(
                int(trace.frame_numbers[row, count - 1]),
                int(result.physics.frames_simulated[env_index]),
            )
            self.assertTrue(
                torch.equal(
                    trace.active[row, count - 1],
                    result.observation.active[env_index].cpu(),
                )
            )

    def test_cuda_trace_records_initial_intervals_and_final_state(self):
        config = SimulatorConfig(
            board_width=320,
            board_height=420,
            spawn_y=80,
            action_count=7,
            max_fruits=8,
            physics_fps=60,
            max_physics_frames=180,
            stable_frames=4,
            solver_iterations=2,
        )
        simulator = TensorVectorSimulator(8, config=config, device='cuda')
        simulator.reset(
            seeds=3,
            fruit_queue=torch.ones((8, 4), dtype=torch.int64, device='cuda'),
        )
        actions = torch.arange(8, dtype=torch.int64, device='cuda').remainder(7)
        result, trace = simulator.step_with_trace(
            actions, [2, 6], frame_stride=4
        )
        trace = trace.cpu()
        payload = trace_to_payload(trace, config)

        self.assertEqual(trace.env_indices.tolist(), [2, 6])
        self.assertEqual(trace.actions.tolist(), [2, 6])
        self.assertEqual(len(payload['clips']), 2)
        self.assertEqual(payload['clips'][0]['records'][0]['frame'], 0)
        for row, env_index in enumerate((2, 6)):
            count = int(trace.record_counts[row])
            self.assertGreaterEqual(count, 2)
            self.assertEqual(int(trace.frame_numbers[row, 0]), 0)
            self.assertEqual(
                int(trace.frame_numbers[row, count - 1]),
                int(result.physics.frames_simulated[env_index]),
            )
            self.assertEqual(
                int(trace.scores[row, count - 1]),
                int(result.observation.score[env_index]),
            )
            self.assertTrue(
                torch.equal(
                    trace.active[row, count - 1],
                    result.observation.active[env_index].cpu(),
                )
            )
            self.assertTrue(
                torch.allclose(
                    trace.positions[row, count - 1],
                    result.observation.positions[env_index].cpu(),
                )
            )

        second_result, second_trace = simulator.step_with_trace(
            torch.zeros(8, dtype=torch.int64, device='cuda'),
            [2, 6],
            frame_stride=4,
        )
        second_trace = second_trace.cpu()
        rollout_payload = trace_to_payload(
            (trace, second_trace), config
        )
        first_clip = rollout_payload['clips'][0]
        self.assertEqual(first_clip['drops'], 2)
        drop_starts = [
            record
            for record in first_clip['records']
            if record['drop_start']
        ]
        self.assertEqual([record['drop'] for record in drop_starts], [1, 2])
        self.assertEqual(drop_starts[1]['action'], 0)
        self.assertEqual(
            first_clip['total_frames'],
            int(result.physics.frames_simulated[2])
            + int(second_result.physics.frames_simulated[2]),
        )

    def test_cuda_watermelons_disappear_without_target(self):
        config = SimulatorConfig(
            max_fruits=8,
            max_physics_frames=20,
            stable_frames=2,
            solver_iterations=2,
        )
        simulator = TensorVectorSimulator(1, config=config, device='cuda')
        simulator.reset(seeds=1, fruit_queue=[1, 1, 1, 1])
        self._install_fruit(simulator, 0, 11, 280, 800, 1)
        self._install_fruit(simulator, 1, 11, 280, 800, 2)
        result = simulator.step(torch.tensor([10], device='cuda'))
        torch.cuda.synchronize()
        self.assertEqual(int(result.physics.score_delta[0]), 66)
        self.assertEqual(int(result.physics.merge_events.count[0]), 1)
        self.assertEqual(
            int(result.physics.merge_events.new_levels[0, 0]), 0
        )
        self.assertEqual(
            int(result.physics.merge_events.new_fruit_ids[0, 0]), 0
        )
        self.assertEqual(int(result.observation.fruit_count[0]), 1)

    def test_cuda_distinguishes_termination_and_settle_timeout(self):
        termination_config = SimulatorConfig(
            board_width=320,
            board_height=420,
            spawn_y=80,
            action_count=7,
            max_fruits=8,
            physics_fps=60,
            max_physics_frames=10,
            stable_frames=2,
            solver_iterations=2,
            danger_seconds=0.0,
        )
        termination_simulator = TensorVectorSimulator(
            1, config=termination_config, device='cuda'
        )
        termination_simulator.reset(
            seeds=1, fruit_queue=[1, 1, 1, 1]
        )
        self._install_fruit(
            termination_simulator, 0, 2, 80, 60, 1
        )
        terminated = termination_simulator.step(
            torch.tensor([3], device='cuda')
        )
        self.assertTrue(terminated.physics.done[0].item())
        self.assertFalse(terminated.physics.truncated[0].item())

        truncation_config = SimulatorConfig(
            max_fruits=8,
            max_physics_frames=1,
            stable_frames=2,
            solver_iterations=2,
        )
        truncation_simulator = TensorVectorSimulator(
            1, config=truncation_config, device='cuda'
        )
        timed_out = truncation_simulator.step(
            torch.tensor([0], device='cuda')
        )
        self.assertFalse(timed_out.physics.done[0].item())
        self.assertFalse(timed_out.physics.truncated[0].item())
        self.assertTrue(timed_out.physics.settle_timeout[0].item())
        self.assertFalse(truncation_simulator.needs_reset[0].item())
        continued = truncation_simulator.step(
            torch.tensor([1], device='cuda')
        )
        self.assertEqual(int(continued.observation.step_count[0]), 2)

    @unittest.skipUnless(
        importlib.util.find_spec('pymunk') is not None,
        'Pymunk is not installed',
    )
    def test_cuda_and_pymunk_agree_on_discrete_two_drop_trace(self):
        from daxigua.simulator.reference import PymunkReferenceGame

        config = SimulatorConfig(
            board_width=320,
            board_height=420,
            spawn_y=80,
            action_count=7,
            max_fruits=8,
            physics_fps=60,
            max_physics_frames=180,
            stable_frames=4,
            solver_iterations=2,
        )
        reference = PymunkReferenceGame(config, seed=1)
        reference.reset(seed=1, fruit_queue=[1, 1, 1, 1])
        simulator = TensorVectorSimulator(1, config=config, device='cuda')
        simulator.reset(seeds=1, fruit_queue=[1, 1, 1, 1])

        for expected_score_delta in (0, 1):
            reference_state, _drop, reference_physics = reference.step(3)
            cuda_result = simulator.step(
                torch.tensor([3], dtype=torch.int64, device='cuda')
            )
            torch.cuda.synchronize()
            cuda_state = simulator.state_at(0)
            self.assertEqual(
                int(cuda_result.physics.score_delta[0]),
                expected_score_delta,
            )
            self.assertEqual(
                reference_physics.score_delta, expected_score_delta
            )
            self.assertEqual(
                cuda_state.fruit_count, reference_state.fruit_count
            )
            self.assertEqual(cuda_state.max_level, reference_state.max_level)
            self.assertEqual(
                bool(cuda_result.physics.stable[0]),
                reference_physics.stable,
            )
            self.assertEqual(
                bool(cuda_result.physics.truncated[0]),
                reference_physics.truncated,
            )
            self.assertEqual(
                bool(cuda_result.physics.settle_timeout[0]),
                reference_physics.settle_timeout,
            )
            self.assertAlmostEqual(
                cuda_state.board_fruits[0].x,
                reference_state.board_fruits[0].x,
                delta=1e-4,
            )
            self.assertAlmostEqual(
                cuda_state.board_fruits[0].y,
                reference_state.board_fruits[0].y,
                delta=6.0,
            )


@unittest.skipUnless(
    torch is not None and importlib.util.find_spec('pymunk') is not None,
    'Pymunk is not installed',
)
class PymunkReferenceGameTest(unittest.TestCase):
    def test_reference_uses_current_merge_score_and_midpoint(self):
        from daxigua.simulator.reference import PymunkReferenceGame

        config = SimulatorConfig(
            max_fruits=8,
            max_physics_frames=20,
            stable_frames=2,
            use_cuda_extension=False,
        )
        game = PymunkReferenceGame(config, seed=1)
        game._create_ball(265, 900, 1)
        game._create_ball(295, 900, 1)
        result = game.advance_physics()
        self.assertEqual(result.score_delta, 1)
        self.assertEqual(len(result.merge_events), 1)
        event = result.merge_events[0]
        self.assertEqual(event.new_level, 2)
        self.assertAlmostEqual(event.x, 280.0, places=4)
        self.assertAlmostEqual(event.y, 900.0, places=4)
        self.assertEqual(
            game.get_state().board_fruits[0].physics_radius,
            merged_fruit_physics_radius(2),
        )

    def test_reference_watermelons_disappear(self):
        from daxigua.simulator.reference import PymunkReferenceGame

        config = SimulatorConfig(
            max_fruits=8,
            max_physics_frames=20,
            stable_frames=2,
            use_cuda_extension=False,
        )
        game = PymunkReferenceGame(config, seed=1)
        game._create_ball(280, 800, 11)
        game._create_ball(280, 800, 11)
        result = game.advance_physics()
        self.assertEqual(result.score_delta, 66)
        self.assertEqual(game.get_state().fruit_count, 0)
        self.assertIsNone(result.merge_events[0].new_level)
        self.assertIsNone(result.merge_events[0].new_fruit_id)

    def test_reference_merge_conserves_total_momentum(self):
        from daxigua.simulator.reference import PymunkReferenceGame

        game = PymunkReferenceGame(
            SimulatorConfig(max_fruits=8, use_cuda_extension=False),
            seed=1,
        )
        shape_a = game._create_ball(140, 220, 1)
        shape_b = game._create_ball(180, 220, 1)
        shape_a.body.velocity = 30.0, 4.0
        shape_b.body.velocity = -6.0, 10.0
        shape_a.body.angular_velocity = 2.0
        shape_b.body.angular_velocity = -1.0
        midpoint = (160.0, 220.0)
        linear_before = (
            shape_a.body.mass * shape_a.body.velocity
            + shape_b.body.mass * shape_b.body.velocity
        )
        angular_before = (
            shape_a.body.moment * shape_a.body.angular_velocity
            + shape_b.body.moment * shape_b.body.angular_velocity
            + (shape_a.body.position.x - midpoint[0])
            * shape_a.body.mass * shape_a.body.velocity.y
            - (shape_a.body.position.y - midpoint[1])
            * shape_a.body.mass * shape_a.body.velocity.x
            + (shape_b.body.position.x - midpoint[0])
            * shape_b.body.mass * shape_b.body.velocity.y
            - (shape_b.body.position.y - midpoint[1])
            * shape_b.body.mass * shape_b.body.velocity.x
        )

        game._handle_merge(type('Arbiter', (), {
            'shapes': (shape_a, shape_b)
        })())

        self.assertEqual(len(game.balls), 1)
        new_body = game.balls[0].body
        self.assertAlmostEqual(
            new_body.mass * new_body.velocity.x,
            linear_before.x,
            places=5,
        )
        self.assertAlmostEqual(
            new_body.mass * new_body.velocity.y,
            linear_before.y,
            places=5,
        )
        self.assertAlmostEqual(
            new_body.moment * new_body.angular_velocity,
            angular_before,
            places=4,
        )


if __name__ == '__main__':
    unittest.main()
