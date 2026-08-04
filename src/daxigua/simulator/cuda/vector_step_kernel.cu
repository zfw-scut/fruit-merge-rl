#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kMaxFruits = 256;
constexpr long long kRngMultiplier = 1103515245LL;
constexpr long long kRngIncrement = 12345LL;
constexpr long long kRngMask = 0x7fffffffLL;

struct Vec2 {
  float x;
  float y;
};

__device__ inline Vec2 add(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
__device__ inline Vec2 sub(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
__device__ inline Vec2 mul(Vec2 a, float value) { return {a.x * value, a.y * value}; }
__device__ inline float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
__device__ inline float cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }
__device__ inline Vec2 angular_cross(float omega, Vec2 radius) {
  return {-omega * radius.y, omega * radius.x};
}
__device__ inline float clamp_value(float value, float minimum, float maximum) {
  return fmaxf(minimum, fminf(maximum, value));
}

struct KernelState {
  float* positions;
  float* velocities;
  float* angles;
  float* angular_velocities;
  long long* levels;
  float* physics_radii;
  float* masses;
  float* inverse_masses;
  float* inverse_inertias;
  long long* fruit_ids;
  long long* age_frames;
  bool* active;
  int env;
  int capacity;

  __device__ inline int slot_index(int slot) const {
    return env * capacity + slot;
  }
  __device__ inline int vector_index(int slot) const {
    return (env * capacity + slot) * 2;
  }
  __device__ inline Vec2 position(int slot) const {
    int index = vector_index(slot);
    return {positions[index], positions[index + 1]};
  }
  __device__ inline void set_position(int slot, Vec2 value) {
    int index = vector_index(slot);
    positions[index] = value.x;
    positions[index + 1] = value.y;
  }
  __device__ inline Vec2 velocity(int slot) const {
    int index = vector_index(slot);
    return {velocities[index], velocities[index + 1]};
  }
  __device__ inline void set_velocity(int slot, Vec2 value) {
    int index = vector_index(slot);
    velocities[index] = value.x;
    velocities[index + 1] = value.y;
  }
};

__device__ inline void apply_wall(
    KernelState& state,
    int slot,
    Vec2 normal,
    float penetration,
    float elasticity,
    float friction) {
  if (penetration <= 0.0f) return;
  int index = state.slot_index(slot);
  float radius = state.physics_radii[index];
  float inverse_mass = state.inverse_masses[index];
  float inverse_inertia = state.inverse_inertias[index];
  Vec2 position = state.position(slot);
  position = add(position, mul(normal, penetration));
  state.set_position(slot, position);

  Vec2 velocity = state.velocity(slot);
  float normal_velocity = dot(velocity, normal);
  if (normal_velocity >= 0.0f) return;
  float normal_impulse = -(1.0f + elasticity) * normal_velocity / inverse_mass;
  Vec2 tangent = {-normal.y, normal.x};
  Vec2 radius_vector = mul(normal, -radius);
  Vec2 contact_velocity = add(
      velocity,
      angular_cross(state.angular_velocities[index], radius_vector));
  float tangent_velocity = dot(contact_velocity, tangent);
  float cross_radius_tangent = cross(radius_vector, tangent);
  float tangent_denominator = inverse_mass
      + cross_radius_tangent * cross_radius_tangent * inverse_inertia;
  float tangent_impulse = -tangent_velocity / fmaxf(tangent_denominator, 1e-12f);
  float friction_limit = friction * normal_impulse;
  tangent_impulse = clamp_value(tangent_impulse, -friction_limit, friction_limit);
  Vec2 impulse = add(mul(normal, normal_impulse), mul(tangent, tangent_impulse));
  state.set_velocity(slot, add(velocity, mul(impulse, inverse_mass)));
  state.angular_velocities[index] += cross(radius_vector, impulse) * inverse_inertia;
}

__device__ inline void resolve_walls(
    KernelState& state,
    int slot,
    int board_width,
    int board_height,
    int wall_width,
    float elasticity,
    float wall_friction) {
  int index = state.slot_index(slot);
  if (!state.active[index]) return;
  float radius = state.physics_radii[index];
  Vec2 position = state.position(slot);
  apply_wall(state, slot, {1.0f, 0.0f}, wall_width + radius - position.x,
             elasticity, wall_friction);
  position = state.position(slot);
  apply_wall(state, slot, {-1.0f, 0.0f},
              position.x - (board_width - wall_width - radius),
             elasticity, wall_friction);
  position = state.position(slot);
  apply_wall(state, slot, {0.0f, -1.0f},
             position.y - (board_height - wall_width - radius),
             elasticity, wall_friction);
}

__device__ inline void resolve_pair(
    KernelState& state,
    int slot_i,
    int slot_j,
    float elasticity,
    float friction,
    float contact_slop,
    float position_correction) {
  int index_i = state.slot_index(slot_i);
  int index_j = state.slot_index(slot_j);
  if (!state.active[index_i] || !state.active[index_j]) return;
  Vec2 position_i = state.position(slot_i);
  Vec2 position_j = state.position(slot_j);
  Vec2 delta = sub(position_j, position_i);
  float radius_i = state.physics_radii[index_i];
  float radius_j = state.physics_radii[index_j];
  float radius_sum = radius_i + radius_j;
  // 大部分槽位对相距很远，先做廉价 AABB 判定，避免进入圆距离和冲量计算。
  if (fabsf(delta.x) > radius_sum || fabsf(delta.y) > radius_sum) return;
  float distance_squared = dot(delta, delta);
  if (distance_squared > radius_sum * radius_sum) return;
  float distance = sqrtf(fmaxf(distance_squared, 1e-12f));
  Vec2 normal = distance_squared < 1e-12f
      ? Vec2{1.0f, 0.0f}
      : mul(delta, 1.0f / distance);
  float inverse_mass_i = state.inverse_masses[index_i];
  float inverse_mass_j = state.inverse_masses[index_j];
  float inverse_mass_sum = fmaxf(inverse_mass_i + inverse_mass_j, 1e-12f);
  float penetration = fmaxf(radius_sum - distance - contact_slop, 0.0f);
  float correction = position_correction * penetration / inverse_mass_sum;
  state.set_position(slot_i, sub(position_i, mul(normal, correction * inverse_mass_i)));
  state.set_position(slot_j, add(position_j, mul(normal, correction * inverse_mass_j)));

  Vec2 radius_vector_i = mul(normal, radius_i);
  Vec2 radius_vector_j = mul(normal, -radius_j);
  Vec2 contact_velocity_i = add(
      state.velocity(slot_i),
      angular_cross(state.angular_velocities[index_i], radius_vector_i));
  Vec2 contact_velocity_j = add(
      state.velocity(slot_j),
      angular_cross(state.angular_velocities[index_j], radius_vector_j));
  Vec2 relative_velocity = sub(contact_velocity_j, contact_velocity_i);
  float normal_velocity = dot(relative_velocity, normal);
  if (normal_velocity >= 0.0f) return;
  float normal_impulse = -(1.0f + elasticity) * normal_velocity / inverse_mass_sum;
  Vec2 tangent = {-normal.y, normal.x};
  float tangent_velocity = dot(relative_velocity, tangent);
  float cross_i = cross(radius_vector_i, tangent);
  float cross_j = cross(radius_vector_j, tangent);
  float tangent_denominator = inverse_mass_sum
      + cross_i * cross_i * state.inverse_inertias[index_i]
      + cross_j * cross_j * state.inverse_inertias[index_j];
  float tangent_impulse = -tangent_velocity / fmaxf(tangent_denominator, 1e-12f);
  float friction_limit = friction * normal_impulse;
  tangent_impulse = clamp_value(tangent_impulse, -friction_limit, friction_limit);
  Vec2 impulse = add(mul(normal, normal_impulse), mul(tangent, tangent_impulse));
  state.set_velocity(
      slot_i,
      sub(state.velocity(slot_i), mul(impulse, inverse_mass_i)));
  state.set_velocity(
      slot_j,
      add(state.velocity(slot_j), mul(impulse, inverse_mass_j)));
  state.angular_velocities[index_i] -=
      cross(radius_vector_i, impulse) * state.inverse_inertias[index_i];
  state.angular_velocities[index_j] +=
      cross(radius_vector_j, impulse) * state.inverse_inertias[index_j];
}

__device__ inline bool touching(
    KernelState& state, int slot_i, int slot_j, float tolerance) {
  int index_i = state.slot_index(slot_i);
  int index_j = state.slot_index(slot_j);
  if (!state.active[index_i] || !state.active[index_j]) return false;
  if (state.levels[index_i] != state.levels[index_j]) return false;
  Vec2 delta = sub(state.position(slot_j), state.position(slot_i));
  float radius_sum = state.physics_radii[index_i]
      + state.physics_radii[index_j] + tolerance;
  if (fabsf(delta.x) > radius_sum || fabsf(delta.y) > radius_sum) return false;
  return dot(delta, delta) <= radius_sum * radius_sum;
}

__device__ inline void record_trace_frame(
    KernelState& state,
    int trace_row,
    int record_index,
    int frame_number,
    int trace_capacity,
    float* trace_positions,
    float* trace_velocities,
    float* trace_angles,
    float* trace_angular_velocities,
    long long* trace_levels,
    float* trace_physics_radii,
    long long* trace_fruit_ids,
    bool* trace_active,
    long long* trace_scores,
    long long* trace_merge_counts,
    long long* trace_frame_numbers,
    long long* trace_record_counts,
    long long score,
    long long merge_count) {
  int frame_index = trace_row * trace_capacity + record_index;
  int fruit_base = frame_index * state.capacity;
  for (int slot = 0; slot < state.capacity; ++slot) {
    int source_index = state.slot_index(slot);
    int target_index = fruit_base + slot;
    Vec2 position = state.position(slot);
    Vec2 velocity = state.velocity(slot);
    trace_positions[target_index * 2] = position.x;
    trace_positions[target_index * 2 + 1] = position.y;
    trace_velocities[target_index * 2] = velocity.x;
    trace_velocities[target_index * 2 + 1] = velocity.y;
    trace_angles[target_index] = state.angles[source_index];
    trace_angular_velocities[target_index] =
        state.angular_velocities[source_index];
    trace_levels[target_index] = state.levels[source_index];
    trace_physics_radii[target_index] = state.physics_radii[source_index];
    trace_fruit_ids[target_index] = state.fruit_ids[source_index];
    trace_active[target_index] = state.active[source_index];
  }
  trace_scores[frame_index] = score;
  trace_merge_counts[frame_index] = merge_count;
  trace_frame_numbers[frame_index] = frame_number;
  trace_record_counts[trace_row] = record_index + 1;
}

__global__ void vector_step_kernel(
    const long long* actions,
    const bool* enabled,
    float* positions,
    float* velocities,
    float* frame_start_positions,
    float* angles,
    float* angular_velocities,
    long long* levels,
    float* physics_radii,
    float* masses,
    float* inverse_masses,
    float* inverse_inertias,
    long long* fruit_ids,
    long long* age_frames,
    bool* active,
    long long* fruit_queue,
    long long* score,
    long long* last_score,
    long long* step_count,
    long long* physics_frame,
    long long* fail_frames,
    long long* next_fruit_id,
    long long* rng_state,
    bool* terminated,
    bool* needs_reset,
    long long* last_drop_level,
    float* last_drop_x,
    long long* last_drop_id,
    long long* last_queue_before,
    long long* last_queue_after,
    long long* event_count,
    long long* event_source_levels,
    long long* event_new_levels,
    float* event_positions,
    long long* event_score_deltas,
    long long* event_source_ids,
    long long* event_new_fruit_ids,
    const float* display_radii,
    const float* dropped_radii,
    const float* merged_radii,
    const float* mass_table,
    const long long* merge_scores,
    long long* frames_simulated,
    long long* fast_forwarded_frames,
    bool* stable_result,
    bool* done_result,
    bool* truncated_result,
    const long long* trace_rows,
    float* trace_positions,
    float* trace_velocities,
    float* trace_angles,
    float* trace_angular_velocities,
    long long* trace_levels,
    float* trace_physics_radii,
    long long* trace_fruit_ids,
    bool* trace_active,
    long long* trace_scores,
    long long* trace_merge_counts,
    long long* trace_frame_numbers,
    long long* trace_record_counts,
    int trace_count,
    int trace_capacity,
    int trace_stride,
    int num_envs,
    int board_width,
    int board_height,
    int spawn_y,
    int wall_width,
    int action_count,
    int max_fruits,
    int queue_length,
    int physics_fps,
    int max_physics_frames,
    int stable_frames,
    int solver_iterations,
    bool drop_fast_forward,
    int kinematic_rest_frames,
    float kinematic_rest_displacement_epsilon,
    float gravity_y,
    float damping,
    float fruit_elasticity,
    float fruit_friction,
    float wall_friction,
    float stable_velocity_epsilon,
    float stable_angular_velocity_epsilon,
    int danger_frame_limit,
    float contact_slop,
    float position_correction,
    float merge_tolerance) {
  int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs || !enabled[env]) return;

  KernelState state{
      positions, velocities, angles, angular_velocities, levels,
      physics_radii, masses, inverse_masses, inverse_inertias,
      fruit_ids, age_frames, active, env, max_fruits};

  unsigned char claimed[kMaxFruits];
  unsigned char kinematic_quiet_frames[kMaxFruits];
  for (int slot = 0; slot < max_fruits; ++slot) {
    kinematic_quiet_frames[slot] = 0;
  }
  int trace_row = -1;
  int queue_base = env * queue_length;
  int event_base = env * max_fruits;
  long long score_before = score[env];
  event_count[env] = 0;
  frames_simulated[env] = 0;
  fast_forwarded_frames[env] = 0;
  stable_result[env] = false;
  done_result[env] = false;
  truncated_result[env] = false;

  int free_slot = -1;
  bool fast_forward_eligible = drop_fast_forward;
  int active_slot_upper_bound = 0;
  for (int slot = 0; slot < max_fruits; ++slot) {
    int index = state.slot_index(slot);
    if (!active[index]) {
      if (free_slot < 0) {
        free_slot = slot;
        if (!drop_fast_forward) break;
      }
      continue;
    }
    if (drop_fast_forward) {
      Vec2 velocity = state.velocity(slot);
      if (dot(velocity, velocity)
              > stable_velocity_epsilon * stable_velocity_epsilon
          || fabsf(angular_velocities[index])
              > stable_angular_velocity_epsilon
          || state.position(slot).y < static_cast<float>(spawn_y)) {
        fast_forward_eligible = false;
      }
    }
  }
  if (free_slot < 0) {
    truncated_result[env] = true;
    needs_reset[env] = true;
    return;
  }

  for (int queue_index = 0; queue_index < queue_length; ++queue_index) {
    last_queue_before[queue_base + queue_index] =
        fruit_queue[queue_base + queue_index];
  }
  long long level = fruit_queue[queue_base];
  float display_radius = display_radii[level];
  float left = wall_width + display_radius + 2.0f;
  float right = board_width - wall_width - display_radius - 2.0f;
  float normalized = static_cast<float>(actions[env]) /
      static_cast<float>(action_count - 1);
  float drop_x = left + (right - left) * normalized;
  int drop_index = state.slot_index(free_slot);
  state.set_position(free_slot, {drop_x, static_cast<float>(spawn_y)});
  state.set_velocity(free_slot, {0.0f, 80.0f});
  angles[drop_index] = 0.0f;
  angular_velocities[drop_index] = 0.0f;
  levels[drop_index] = level;
  float drop_radius = dropped_radii[level];
  physics_radii[drop_index] = drop_radius;
  float mass = mass_table[level];
  masses[drop_index] = mass;
  inverse_masses[drop_index] = 1.0f / mass;
  inverse_inertias[drop_index] = 1.0f / (0.5f * mass * drop_radius * drop_radius);
  long long drop_id = next_fruit_id[env]++;
  fruit_ids[drop_index] = drop_id;
  age_frames[drop_index] = 0;
  active[drop_index] = true;

  for (int queue_index = 0; queue_index + 1 < queue_length; ++queue_index) {
    fruit_queue[queue_base + queue_index] =
        fruit_queue[queue_base + queue_index + 1];
  }
  long long next_rng =
      (rng_state[env] * kRngMultiplier + kRngIncrement) & kRngMask;
  rng_state[env] = next_rng;
  fruit_queue[queue_base + queue_length - 1] = next_rng % 5 + 1;
  for (int queue_index = 0; queue_index < queue_length; ++queue_index) {
    last_queue_after[queue_base + queue_index] =
        fruit_queue[queue_base + queue_index];
  }
  last_drop_level[env] = level;
  last_drop_x[env] = drop_x;
  last_drop_id[env] = drop_id;
  step_count[env] += 1;

  active_slot_upper_bound = free_slot + 1;
  for (int slot = free_slot + 1; slot < max_fruits; ++slot) {
    if (active[state.slot_index(slot)]) active_slot_upper_bound = slot + 1;
  }

  if (trace_count > 0) trace_row = static_cast<int>(trace_rows[env]);
  if (trace_row >= 0 && trace_row < trace_count) {
    record_trace_frame(
        state, trace_row, 0, 0, trace_capacity,
        trace_positions, trace_velocities, trace_angles,
        trace_angular_velocities, trace_levels, trace_physics_radii,
        trace_fruit_ids, trace_active, trace_scores, trace_merge_counts,
        trace_frame_numbers, trace_record_counts, score[env], 0);
  }

  const float dt = 1.0f / static_cast<float>(physics_fps);
  const float frame_damping = powf(damping, dt);
  const float stable_velocity_squared =
      stable_velocity_epsilon * stable_velocity_epsilon;
  int consecutive_stable = 0;
  int skipped_frames = 0;
  if (fast_forward_eligible) {
    float contact_y = static_cast<float>(board_height - wall_width)
        - drop_radius;
    for (int slot = 0; slot < active_slot_upper_bound; ++slot) {
      int index = state.slot_index(slot);
      if (!active[index] || slot == free_slot) continue;
      Vec2 other_position = state.position(slot);
      float dx = drop_x - other_position.x;
      float radius_sum = drop_radius + physics_radii[index];
      if (fabsf(dx) >= radius_sum) continue;
      float vertical_offset = sqrtf(fmaxf(
          radius_sum * radius_sum - dx * dx, 0.0f));
      float candidate_y = other_position.y - vertical_offset;
      if (candidate_y > static_cast<float>(spawn_y)
          && candidate_y < contact_y) {
        contact_y = candidate_y;
      }
    }

    Vec2 drop_position = state.position(free_slot);
    Vec2 drop_velocity = state.velocity(free_slot);
    while (skipped_frames < max_physics_frames) {
      float next_velocity_y =
          (drop_velocity.y + gravity_y * dt) * frame_damping;
      float next_y = drop_position.y + next_velocity_y * dt;
      if (next_y >= contact_y) break;
      drop_velocity.y = next_velocity_y;
      drop_position.y = next_y;
      ++skipped_frames;
    }
    state.set_position(free_slot, drop_position);
    state.set_velocity(free_slot, drop_velocity);
    for (int slot = 0; slot < active_slot_upper_bound; ++slot) {
      int index = state.slot_index(slot);
      if (active[index]) age_frames[index] += skipped_frames;
    }
    physics_frame[env] += skipped_frames;
    if (skipped_frames > 0) fail_frames[env] = 0;
    frames_simulated[env] = skipped_frames;
    fast_forwarded_frames[env] = skipped_frames;
  }

  int trace_record_index = 1;
  if (skipped_frames > 0 && trace_row >= 0 && trace_row < trace_count) {
    record_trace_frame(
        state, trace_row, trace_record_index++, skipped_frames, trace_capacity,
        trace_positions, trace_velocities, trace_angles,
        trace_angular_velocities, trace_levels, trace_physics_radii,
        trace_fruit_ids, trace_active, trace_scores, trace_merge_counts,
        trace_frame_numbers, trace_record_counts, score[env], 0);
  }
  bool running = skipped_frames < max_physics_frames;

  for (int frame = skipped_frames;
       frame < max_physics_frames && running;
       ++frame) {
    for (int slot = 0; slot < active_slot_upper_bound; ++slot) {
      int index = state.slot_index(slot);
      if (!active[index]) continue;
      int vector_index = state.vector_index(slot);
      Vec2 frame_start = state.position(slot);
      frame_start_positions[vector_index] = frame_start.x;
      frame_start_positions[vector_index + 1] = frame_start.y;
      Vec2 velocity = state.velocity(slot);
      velocity.y += gravity_y * dt;
      velocity = mul(velocity, frame_damping);
      state.set_velocity(slot, velocity);
      state.set_position(slot, add(state.position(slot), mul(velocity, dt)));
      angles[index] += angular_velocities[index] * dt;
      age_frames[index] += 1;
    }

    for (int iteration = 0; iteration < solver_iterations; ++iteration) {
      for (int slot = 0; slot < active_slot_upper_bound; ++slot) {
        resolve_walls(state, slot, board_width, board_height, wall_width,
                      fruit_elasticity, wall_friction);
      }
      for (int slot_i = 0; slot_i < active_slot_upper_bound; ++slot_i) {
        for (int slot_j = slot_i + 1; slot_j < active_slot_upper_bound; ++slot_j) {
          resolve_pair(state, slot_i, slot_j, fruit_elasticity,
                       fruit_friction, contact_slop, position_correction);
        }
      }
    }
    for (int slot = 0; slot < active_slot_upper_bound; ++slot) {
      claimed[slot] = 0;
      resolve_walls(state, slot, board_width, board_height, wall_width,
                    fruit_elasticity, wall_friction);
    }

    for (int slot_i = 0; slot_i < active_slot_upper_bound; ++slot_i) {
      if (claimed[slot_i]) continue;
      for (int slot_j = slot_i + 1; slot_j < active_slot_upper_bound; ++slot_j) {
        if (claimed[slot_j]) continue;
        if (!touching(state, slot_i, slot_j, merge_tolerance)) continue;
        int index_i = state.slot_index(slot_i);
        int index_j = state.slot_index(slot_j);
        claimed[slot_i] = 1;
        claimed[slot_j] = 1;
        long long source_level = levels[index_i];
        long long source_id_i = fruit_ids[index_i];
        long long source_id_j = fruit_ids[index_j];
        Vec2 midpoint = mul(add(state.position(slot_i), state.position(slot_j)), 0.5f);
        long long delta_score = merge_scores[source_level];
        long long target_level = source_level < 11 ? source_level + 1 : 0;
        long long new_id = target_level > 0 ? next_fruit_id[env]++ : 0;
        Vec2 inherited_velocity = {0.0f, 0.0f};
        float inherited_angular_velocity = 0.0f;
        float new_radius = 0.0f;
        float new_mass = 1.0f;
        if (target_level > 0) {
          Vec2 position_i = state.position(slot_i);
          Vec2 position_j = state.position(slot_j);
          Vec2 velocity_i = state.velocity(slot_i);
          Vec2 velocity_j = state.velocity(slot_j);
          float mass_i = masses[index_i];
          float mass_j = masses[index_j];
          new_radius = merged_radii[target_level];
          new_mass = mass_table[target_level];
          Vec2 momentum_i = mul(velocity_i, mass_i);
          Vec2 momentum_j = mul(velocity_j, mass_j);
          inherited_velocity = mul(
              add(momentum_i, momentum_j), 1.0f / new_mass);
          float inertia_i =
              1.0f / fmaxf(inverse_inertias[index_i], 1e-12f);
          float inertia_j =
              1.0f / fmaxf(inverse_inertias[index_j], 1e-12f);
          float angular_momentum =
              inertia_i * angular_velocities[index_i]
              + inertia_j * angular_velocities[index_j]
              + cross(sub(position_i, midpoint), momentum_i)
              + cross(sub(position_j, midpoint), momentum_j);
          float new_inertia = 0.5f * new_mass * new_radius * new_radius;
          inherited_angular_velocity = angular_momentum / new_inertia;
        }

        int event_index = event_base + static_cast<int>(event_count[env]);
        event_source_levels[event_index] = source_level;
        event_new_levels[event_index] = target_level;
        event_positions[event_index * 2] = midpoint.x;
        event_positions[event_index * 2 + 1] = midpoint.y;
        event_score_deltas[event_index] = delta_score;
        event_source_ids[event_index * 2] = source_id_i;
        event_source_ids[event_index * 2 + 1] = source_id_j;
        event_new_fruit_ids[event_index] = new_id;
        event_count[env] += 1;

        last_score[env] = score[env];
        score[env] += delta_score;
        active[index_j] = false;
        kinematic_quiet_frames[slot_j] = 0;
        levels[index_j] = 0;
        fruit_ids[index_j] = 0;
        physics_radii[index_j] = 0.0f;
        state.set_velocity(slot_j, {0.0f, 0.0f});

        if (target_level == 0) {
          active[index_i] = false;
          kinematic_quiet_frames[slot_i] = 0;
          levels[index_i] = 0;
          fruit_ids[index_i] = 0;
          physics_radii[index_i] = 0.0f;
          state.set_velocity(slot_i, {0.0f, 0.0f});
        } else {
          kinematic_quiet_frames[slot_i] = 0;
          levels[index_i] = target_level;
          fruit_ids[index_i] = new_id;
          age_frames[index_i] = 0;
          state.set_position(slot_i, midpoint);
          state.set_velocity(slot_i, inherited_velocity);
          angles[index_i] = 0.0f;
          angular_velocities[index_i] = inherited_angular_velocity;
          physics_radii[index_i] = new_radius;
          masses[index_i] = new_mass;
          inverse_masses[index_i] = 1.0f / new_mass;
          inverse_inertias[index_i] =
              1.0f / (0.5f * new_mass * new_radius * new_radius);
        }
        break;
      }
    }

    while (active_slot_upper_bound > 0
           && !active[state.slot_index(active_slot_upper_bound - 1)]) {
      --active_slot_upper_bound;
    }

    if (kinematic_rest_frames > 0) {
      for (int slot = 0; slot < active_slot_upper_bound; ++slot) {
        int index = state.slot_index(slot);
        if (!active[index]) {
          kinematic_quiet_frames[slot] = 0;
          continue;
        }
        int vector_index = state.vector_index(slot);
        Vec2 position = state.position(slot);
        Vec2 displacement = {
            position.x - frame_start_positions[vector_index],
            position.y - frame_start_positions[vector_index + 1]};
        bool already_resting =
            kinematic_quiet_frames[slot] >= kinematic_rest_frames;
        float displacement_epsilon = already_resting
            ? stable_velocity_epsilon * dt
            : kinematic_rest_displacement_epsilon;
        if (dot(displacement, displacement)
            <= displacement_epsilon * displacement_epsilon) {
          if (kinematic_quiet_frames[slot] < 255) {
            ++kinematic_quiet_frames[slot];
          }
        } else {
          kinematic_quiet_frames[slot] = 0;
        }
        if (kinematic_quiet_frames[slot] >= kinematic_rest_frames) {
          state.set_velocity(slot, {0.0f, 0.0f});
        }
      }
    }

    frames_simulated[env] += 1;
    physics_frame[env] += 1;

    long long newest_id = 0;
    for (int slot = 0; slot < active_slot_upper_bound; ++slot) {
      int index = state.slot_index(slot);
      if (active[index]) newest_id = max(newest_id, fruit_ids[index]);
    }
    bool over_line = false;
    for (int slot = 0; slot < active_slot_upper_bound; ++slot) {
      int index = state.slot_index(slot);
      if (!active[index] || fruit_ids[index] == newest_id) continue;
      if (static_cast<int>(state.position(slot).y) < spawn_y) {
        over_line = true;
        break;
      }
    }
    fail_frames[env] = over_line ? fail_frames[env] + 1 : 0;
    if (fail_frames[env] > danger_frame_limit) {
      done_result[env] = true;
      running = false;
    }

    bool all_stable = running;
    for (int slot = 0;
         all_stable && slot < active_slot_upper_bound;
         ++slot) {
      int index = state.slot_index(slot);
      if (!active[index]) continue;
      Vec2 velocity = state.velocity(slot);
      if (dot(velocity, velocity) > stable_velocity_squared
          || fabsf(angular_velocities[index]) > stable_angular_velocity_epsilon) {
        all_stable = false;
        break;
      }
    }
    consecutive_stable = all_stable ? consecutive_stable + 1 : 0;
    if (consecutive_stable >= stable_frames) {
      stable_result[env] = true;
      running = false;
    }

    if (trace_row >= 0 && trace_row < trace_count) {
      int completed_frames = frame + 1;
      bool trace_interval = completed_frames % trace_stride == 0;
      if (trace_interval || !running
          || completed_frames == max_physics_frames) {
        record_trace_frame(
            state, trace_row, trace_record_index++, completed_frames,
            trace_capacity,
            trace_positions, trace_velocities, trace_angles,
            trace_angular_velocities, trace_levels, trace_physics_radii,
            trace_fruit_ids, trace_active, trace_scores, trace_merge_counts,
            trace_frame_numbers, trace_record_counts, score[env],
            event_count[env]);
      }
    }
  }

  // max_physics_frames is only the wait budget between decisions. Preserve
  // motion so the next drop can continue; only technical boundaries truncate.
  terminated[env] = done_result[env];
  needs_reset[env] = done_result[env] || truncated_result[env];
  (void)score_before;
}

}  // namespace

void vector_step_cuda(
    torch::Tensor actions,
    torch::Tensor enabled,
    torch::Tensor positions,
    torch::Tensor velocities,
    torch::Tensor frame_start_positions,
    torch::Tensor angles,
    torch::Tensor angular_velocities,
    torch::Tensor levels,
    torch::Tensor physics_radii,
    torch::Tensor masses,
    torch::Tensor inverse_masses,
    torch::Tensor inverse_inertias,
    torch::Tensor fruit_ids,
    torch::Tensor age_frames,
    torch::Tensor active,
    torch::Tensor fruit_queue,
    torch::Tensor score,
    torch::Tensor last_score,
    torch::Tensor step_count,
    torch::Tensor physics_frame,
    torch::Tensor fail_frames,
    torch::Tensor next_fruit_id,
    torch::Tensor rng_state,
    torch::Tensor terminated,
    torch::Tensor needs_reset,
    torch::Tensor last_drop_level,
    torch::Tensor last_drop_x,
    torch::Tensor last_drop_id,
    torch::Tensor last_queue_before,
    torch::Tensor last_queue_after,
    torch::Tensor event_count,
    torch::Tensor event_source_levels,
    torch::Tensor event_new_levels,
    torch::Tensor event_positions,
    torch::Tensor event_score_deltas,
    torch::Tensor event_source_ids,
    torch::Tensor event_new_fruit_ids,
    torch::Tensor display_radii,
    torch::Tensor dropped_radii,
    torch::Tensor merged_radii,
    torch::Tensor mass_table,
    torch::Tensor merge_scores,
    torch::Tensor frames_simulated,
    torch::Tensor fast_forwarded_frames,
    torch::Tensor stable_result,
    torch::Tensor done_result,
    torch::Tensor truncated_result,
    torch::Tensor trace_rows,
    torch::Tensor trace_positions,
    torch::Tensor trace_velocities,
    torch::Tensor trace_angles,
    torch::Tensor trace_angular_velocities,
    torch::Tensor trace_levels,
    torch::Tensor trace_physics_radii,
    torch::Tensor trace_fruit_ids,
    torch::Tensor trace_active,
    torch::Tensor trace_scores,
    torch::Tensor trace_merge_counts,
    torch::Tensor trace_frame_numbers,
    torch::Tensor trace_record_counts,
    int64_t trace_count,
    int64_t trace_capacity,
    int64_t trace_stride,
    int64_t board_width,
    int64_t board_height,
    int64_t spawn_y,
    int64_t wall_width,
    int64_t action_count,
    int64_t max_fruits,
    int64_t queue_length,
    int64_t physics_fps,
    int64_t max_physics_frames,
    int64_t stable_frames,
    int64_t solver_iterations,
    bool drop_fast_forward,
    int64_t kinematic_rest_frames,
    double kinematic_rest_displacement_epsilon,
    double gravity_y,
    double damping,
    double fruit_elasticity,
    double fruit_friction,
    double wall_friction,
    double stable_velocity_epsilon,
    double stable_angular_velocity_epsilon,
    int64_t danger_frame_limit,
    double contact_slop,
    double position_correction,
    double merge_tolerance,
    int64_t threads_per_block) {
  const c10::cuda::CUDAGuard device_guard(actions.device());
  int num_envs = static_cast<int>(actions.numel());
  int threads = static_cast<int>(threads_per_block);
  int blocks = (num_envs + threads - 1) / threads;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vector_step_kernel<<<blocks, threads, 0, stream>>>(
      actions.data_ptr<long long>(), enabled.data_ptr<bool>(),
      positions.data_ptr<float>(),
      velocities.data_ptr<float>(), frame_start_positions.data_ptr<float>(),
      angles.data_ptr<float>(),
      angular_velocities.data_ptr<float>(), levels.data_ptr<long long>(),
      physics_radii.data_ptr<float>(), masses.data_ptr<float>(),
      inverse_masses.data_ptr<float>(), inverse_inertias.data_ptr<float>(),
      fruit_ids.data_ptr<long long>(), age_frames.data_ptr<long long>(),
      active.data_ptr<bool>(), fruit_queue.data_ptr<long long>(),
      score.data_ptr<long long>(), last_score.data_ptr<long long>(),
      step_count.data_ptr<long long>(), physics_frame.data_ptr<long long>(),
      fail_frames.data_ptr<long long>(), next_fruit_id.data_ptr<long long>(),
      rng_state.data_ptr<long long>(), terminated.data_ptr<bool>(),
      needs_reset.data_ptr<bool>(), last_drop_level.data_ptr<long long>(),
      last_drop_x.data_ptr<float>(), last_drop_id.data_ptr<long long>(),
      last_queue_before.data_ptr<long long>(),
      last_queue_after.data_ptr<long long>(), event_count.data_ptr<long long>(),
      event_source_levels.data_ptr<long long>(),
      event_new_levels.data_ptr<long long>(), event_positions.data_ptr<float>(),
      event_score_deltas.data_ptr<long long>(),
      event_source_ids.data_ptr<long long>(),
      event_new_fruit_ids.data_ptr<long long>(), display_radii.data_ptr<float>(),
      dropped_radii.data_ptr<float>(), merged_radii.data_ptr<float>(),
      mass_table.data_ptr<float>(), merge_scores.data_ptr<long long>(),
      frames_simulated.data_ptr<long long>(),
      fast_forwarded_frames.data_ptr<long long>(), stable_result.data_ptr<bool>(),
      done_result.data_ptr<bool>(), truncated_result.data_ptr<bool>(),
      trace_rows.data_ptr<long long>(), trace_positions.data_ptr<float>(),
      trace_velocities.data_ptr<float>(), trace_angles.data_ptr<float>(),
      trace_angular_velocities.data_ptr<float>(),
      trace_levels.data_ptr<long long>(),
      trace_physics_radii.data_ptr<float>(),
      trace_fruit_ids.data_ptr<long long>(), trace_active.data_ptr<bool>(),
      trace_scores.data_ptr<long long>(),
      trace_merge_counts.data_ptr<long long>(),
      trace_frame_numbers.data_ptr<long long>(),
      trace_record_counts.data_ptr<long long>(),
      static_cast<int>(trace_count), static_cast<int>(trace_capacity),
      static_cast<int>(trace_stride),
      num_envs, static_cast<int>(board_width), static_cast<int>(board_height),
      static_cast<int>(spawn_y), static_cast<int>(wall_width),
      static_cast<int>(action_count), static_cast<int>(max_fruits),
      static_cast<int>(queue_length), static_cast<int>(physics_fps),
      static_cast<int>(max_physics_frames), static_cast<int>(stable_frames),
      static_cast<int>(solver_iterations),
      drop_fast_forward,
      static_cast<int>(kinematic_rest_frames),
      static_cast<float>(kinematic_rest_displacement_epsilon),
      static_cast<float>(gravity_y),
      static_cast<float>(damping), static_cast<float>(fruit_elasticity),
      static_cast<float>(fruit_friction), static_cast<float>(wall_friction),
      static_cast<float>(stable_velocity_epsilon),
      static_cast<float>(stable_angular_velocity_epsilon),
      static_cast<int>(danger_frame_limit), static_cast<float>(contact_slop),
      static_cast<float>(position_correction),
      static_cast<float>(merge_tolerance));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
