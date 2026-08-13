"""批量 Tensor/CUDA 物理模拟器契约测试。"""

import unittest


try:
    import torch
except ImportError:  # 核心规则仍可在无 PyTorch 环境中独立测试。
    torch = None


if torch is not None:
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

    def test_decision_state_rows_clone_without_modifying_source(self):
        config = self._config(track_action_effects=True)
        source = TensorVectorSimulator(2, config=config, device='cpu')
        destination = TensorVectorSimulator(4, config=config, device='cpu')
        source.reset(
            seeds=torch.tensor((11, 22)),
            fruit_queue=torch.ones((2, 4), dtype=torch.int64),
        )
        source.step(torch.tensor((1, 5)))
        before = source.observe().clone()
        source_rows = torch.tensor((1, 0, 1, 0))

        destination.copy_rows_from(source, source_rows)

        cloned = destination.observe()
        self.assertTrue(torch.equal(
            cloned.positions, before.positions.index_select(0, source_rows)
        ))
        self.assertTrue(torch.equal(
            cloned.fruit_queue,
            before.fruit_queue.index_select(0, source_rows),
        ))
        cloned_step_count = cloned.step_count.clone()
        self.assertTrue(torch.equal(source.observe().positions, before.positions))
        result = destination.step(torch.tensor((0, 2, 4, 6)))
        self.assertTrue(bool(result.physics.stable.all()))
        self.assertTrue(torch.equal(
            result.observation.step_count,
            cloned_step_count + 1,
        ))
        self.assertTrue(torch.equal(source.observe().positions, before.positions))

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

    def test_unsupported_fruit_keeps_accelerating_under_gravity(self):
        config = self._config(
            physics_fps=120,
            stable_frames=15,
            gravity_y=1800.0,
        )
        simulator = TensorVectorSimulator(1, config=config, device='cpu')
        self._install_fruit(simulator, 0, 0, 1, 100, 200, 1)
        start_y = float(simulator.positions[0, 0, 1])
        velocities = []

        for _ in range(3):
            physics = simulator.advance_incremental_frame()
            velocities.append(float(simulator.velocities[0, 0, 1]))

        self.assertGreater(velocities[0], 0.0)
        self.assertGreater(velocities[1], velocities[0])
        self.assertGreater(velocities[2], velocities[1])
        self.assertGreater(float(simulator.positions[0, 0, 1]), start_y)
        self.assertFalse(bool(physics.stable[0]))

    def test_fruit_accelerates_after_support_disappears_mid_action(self):
        config = self._config(
            physics_fps=120,
            stable_frames=15,
            gravity_y=1800.0,
        )
        simulator = TensorVectorSimulator(1, config=config, device='cpu')
        self._install_fruit(simulator, 0, 0, 3, 100, 360, 1)
        self._install_fruit(simulator, 0, 1, 2, 100, 290, 2)
        for _ in range(12):
            simulator.advance_incremental_frame()
        simulator.reset_incremental_progress(stable=True)
        simulator.active[0, 0] = False
        start_y = float(simulator.positions[0, 1, 1])
        velocities = []

        for _ in range(3):
            physics = simulator.advance_incremental_frame()
            velocities.append(float(simulator.velocities[0, 1, 1]))

        self.assertGreater(velocities[0], 0.0)
        self.assertGreater(velocities[1], velocities[0])
        self.assertGreater(velocities[2], velocities[1])
        self.assertGreater(float(simulator.positions[0, 1, 1]), start_y)
        self.assertFalse(bool(physics.stable[0]))

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

    def test_merge_initializes_zero_linear_and_angular_velocity(self):
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

        simulator._resolve_merges(torch.tensor([True]))

        target_slot = int(torch.nonzero(
            simulator.active[0], as_tuple=False
        )[0])
        self.assertTrue(torch.equal(
            simulator.velocities[0, target_slot], torch.zeros(2)
        ))
        self.assertEqual(
            float(simulator.angular_velocities[0, target_slot]), 0.0
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
        self.assertFalse(config.drop_fast_forward)
        self.assertTrue(config.adaptive_collision_substeps)
        self.assertEqual(config.max_collision_substeps, 2)
        self.assertAlmostEqual(config.position_correction, 0.9)
        self.assertAlmostEqual(
            config.max_physics_frames / config.physics_fps, 6.0
        )
        high_fidelity = SimulatorConfig.high_fidelity_fast(max_fruits=8)
        self.assertEqual(high_fidelity.physics_fps, 120)
        self.assertEqual(high_fidelity.max_physics_frames, 720)
        self.assertFalse(high_fidelity.drop_fast_forward)

    def test_legacy_drop_fast_forward_flag_cannot_enable_skipping(self):
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

        self.assertFalse(config.drop_fast_forward)
        self.assertTrue(
            torch.equal(
                result.physics.fast_forwarded_frames,
                torch.zeros_like(result.physics.fast_forwarded_frames),
            )
        )

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

    def test_cuda_unsupported_fruit_keeps_accelerating_under_gravity(self):
        config = SimulatorConfig.high_fidelity_fast(
            max_fruits=8,
            use_cuda_extension=True,
            drop_fast_forward=False,
        )
        simulator = TensorVectorSimulator(1, config=config, device='cuda')
        simulator.reset(seeds=1)
        self._install_fruit(simulator, 0, 1, 280, 500, 1)
        start_y = float(simulator.positions[0, 0, 1])
        velocities = []

        for _ in range(3):
            physics = simulator.advance_incremental_frame()
            velocities.append(float(simulator.velocities[0, 0, 1]))
        torch.cuda.synchronize()

        self.assertGreater(velocities[0], 0.0)
        self.assertGreater(velocities[1], velocities[0])
        self.assertGreater(velocities[2], velocities[1])
        self.assertGreater(float(simulator.positions[0, 0, 1]), start_y)
        self.assertFalse(bool(physics.stable[0]))

    def test_cuda_fruit_accelerates_after_support_disappears_mid_action(self):
        config = SimulatorConfig.high_fidelity_fast(
            max_fruits=8,
            use_cuda_extension=True,
            drop_fast_forward=False,
        )
        simulator = TensorVectorSimulator(1, config=config, device='cuda')
        simulator.reset(seeds=7, fruit_queue=[1, 2, 3, 4])
        self._install_fruit(simulator, 0, 3, 120, 1058, 1)
        self._install_fruit(simulator, 1, 2, 120, 986, 2)
        simulator.begin_incremental_action(torch.tensor([20], device='cuda'))
        for _ in range(12):
            simulator.advance_incremental_frame()
        simulator.reset_incremental_progress(stable=True)
        simulator.active[0, 0] = False
        start_y = float(simulator.positions[0, 1, 1])
        velocities = []

        for _ in range(3):
            physics = simulator.advance_incremental_frame()
            velocities.append(float(simulator.velocities[0, 1, 1]))
        torch.cuda.synchronize()

        self.assertGreater(velocities[0], 0.0)
        self.assertGreater(velocities[1], velocities[0])
        self.assertGreater(velocities[2], velocities[1])
        self.assertGreater(float(simulator.positions[0, 1, 1]), start_y)
        self.assertFalse(bool(physics.stable[0]))

    def test_cuda_training_profile_simulates_every_free_fall_frame(self):
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

        self.assertFalse(config.drop_fast_forward)
        self.assertTrue(
            (result.physics.fast_forwarded_frames == 0).all().item()
        )
        executed = result.physics.frames_simulated
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

    def test_cuda_merge_initializes_zero_velocity(self):
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

        target_mask = result.observation.active[0] & (
            result.observation.levels[0] == 2
        )
        self.assertEqual(int(target_mask.sum()), 1)
        actual_velocity = result.observation.velocities[0][target_mask][0]
        actual_angular_velocity = result.observation.angular_velocities[
            0
        ][target_mask][0]
        self.assertTrue(torch.equal(
            actual_velocity, torch.zeros_like(actual_velocity)
        ))
        self.assertEqual(float(actual_angular_velocity), 0.0)


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

    def test_cuda_incremental_frames_match_full_training_step(self):
        config = SimulatorConfig.high_fidelity_fast(
            max_fruits=64,
            use_cuda_extension=True,
            drop_fast_forward=False,
        )
        full = TensorVectorSimulator(1, config=config, device='cuda')
        incremental = TensorVectorSimulator(1, config=config, device='cuda')
        queue = [1, 1, 4, 5]
        full.reset(seeds=7, fruit_queue=queue)
        incremental.reset(seeds=7, fruit_queue=queue)
        action = torch.tensor([10], dtype=torch.int64, device='cuda')

        for _ in range(2):
            expected = full.step(action)
            incremental.begin_incremental_action(action)
            frames = 0
            stable = False
            done = False
            while (
                    frames < config.max_physics_frames
                    and not stable
                    and not done):
                physics = incremental.advance_incremental_frame()
                frames += 1
                stable = bool(physics.stable[0].item())
                done = bool(physics.done[0].item())
            torch.cuda.synchronize()

            self.assertEqual(
                int(expected.physics.frames_simulated[0].item()), frames
            )
            for name in (
                    'positions', 'velocities', 'angles', 'angular_velocities',
                    'levels', 'physics_radii', 'fruit_ids', 'age_frames',
                    'active', 'fruit_queue', 'score', 'step_count',
                    'physics_frame', 'fail_frames', 'next_fruit_id', 'rng_state'):
                self.assertTrue(
                    torch.equal(getattr(full, name), getattr(incremental, name)),
                    name,
                )
        self.assertEqual(1, int(full.score[0].item()))

    def test_cuda_training_step_without_fast_forward_matches_lab_path(self):
        config = SimulatorConfig.training_fast(
            max_fruits=64,
            use_cuda_extension=True,
            drop_fast_forward=False,
        )
        full = TensorVectorSimulator(1, config=config, device='cuda')
        incremental = TensorVectorSimulator(1, config=config, device='cuda')
        full.reset(seeds=20260813)
        incremental.reset(seeds=20260813)
        actions = (2, 2, 12, 1, 14, 8, 6, 11, 18, 13)

        for action_index in actions:
            action = torch.tensor(
                [action_index], dtype=torch.int64, device='cuda'
            )
            expected = full.step(action)
            incremental.begin_incremental_action(action)
            frames = 0
            stable = False
            done = False
            while (
                    frames < config.max_physics_frames
                    and not stable
                    and not done):
                physics = incremental.advance_incremental_frame()
                frames += 1
                stable = bool(physics.stable[0].item())
                done = bool(physics.done[0].item())

            self.assertEqual(
                int(expected.physics.frames_simulated[0].item()), frames
            )
            for name in (
                    'positions', 'velocities', 'angles', 'angular_velocities',
                    'levels', 'physics_radii', 'fruit_ids', 'age_frames',
                    'active', 'fruit_queue', 'score', 'step_count',
                    'physics_frame', 'fail_frames', 'next_fruit_id',
                    'rng_state'):
                self.assertTrue(
                    torch.equal(getattr(full, name), getattr(incremental, name)),
                    f'{name} after action {action_index}',
                )


if __name__ == '__main__':
    unittest.main()
