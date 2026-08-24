export type SceneGeometry = {
  board_width: number;
  board_height: number;
  wall_width: number;
  spawn_y: number;
  action_count?: number;
  queue_length?: number;
  max_fruits?: number;
};

export type SceneFruit = {
  id: number;
  level: number;
  x: number;
  y: number;
  physics_radius: number;
  angle?: number;
  vx?: number;
  vy?: number;
};

export type SceneSnapshot = {
  name?: string;
  score?: number;
  step_count?: number;
  physics_frame?: number;
  queue?: number[];
  fruits: SceneFruit[];
  geometry?: Partial<SceneGeometry>;
  metadata?: Record<string, unknown>;
};

export type SceneFruitSpec = {
  level: number;
  radius: number;
};

export const DEFAULT_SCENE_GEOMETRY: SceneGeometry = {
  board_width: 560,
  board_height: 1120,
  wall_width: 8,
  spawn_y: 156,
  action_count: 21,
  queue_length: 4,
  max_fruits: 64,
};
