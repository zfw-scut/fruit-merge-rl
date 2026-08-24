import type { ReactNode } from "react";
import {
  DEFAULT_SCENE_GEOMETRY,
  type SceneFruit,
  type SceneFruitSpec,
  type SceneGeometry,
} from "./types";

const LEVEL_COLORS = [
  "#8f55ad", "#e46c0a", "#ef8423", "#f8d64a", "#76c86f", "#df544e",
  "#f09288", "#dca923", "#9b8175", "#31a65f", "#258875",
];

type Props = {
  fruits: SceneFruit[];
  geometry?: Partial<SceneGeometry>;
  specs?: SceneFruitSpec[];
  textures?: string[];
  className?: string;
  ariaLabel?: string;
  showGrid?: boolean;
  showDanger?: boolean;
  showLabels?: boolean;
  beforeFruits?: ReactNode;
  afterFruits?: ReactNode;
};

export function SceneCanvas({
  fruits,
  geometry: partialGeometry,
  specs = [],
  textures = [],
  className = "",
  ariaLabel = "只读水果场景",
  showGrid = true,
  showDanger = true,
  showLabels = true,
  beforeFruits,
  afterFruits,
}: Props) {
  const geometry = { ...DEFAULT_SCENE_GEOMETRY, ...partialGeometry };
  const radii = new Map(specs.map((spec) => [spec.level, spec.radius]));
  return (
    <svg
      className={`scene-canvas lab-board ${className}`.trim()}
      viewBox={`0 0 ${geometry.board_width} ${geometry.board_height}`}
      aria-label={ariaLabel}
    >
      <rect width={geometry.board_width} height={geometry.board_height} className="lab-board-bg" />
      {showGrid && Array.from({ length: 10 }, (_, index) => (
        <line key={`v${index}`} x1={(index + 1) * geometry.board_width / 11} x2={(index + 1) * geometry.board_width / 11} y1="0" y2={geometry.board_height} className="lab-grid-line" />
      ))}
      {showGrid && Array.from({ length: 11 }, (_, index) => (
        <line key={`h${index}`} y1={(index + 1) * geometry.board_height / 12} y2={(index + 1) * geometry.board_height / 12} x1="0" x2={geometry.board_width} className="lab-grid-line" />
      ))}
      {showDanger && <line x1={geometry.wall_width} x2={geometry.board_width - geometry.wall_width} y1={geometry.spawn_y} y2={geometry.spawn_y} className="lab-danger-line" />}
      {beforeFruits}
      {fruits.map((fruit) => {
        const radius = radii.get(fruit.level) ?? fruit.physics_radius;
        const texture = textures[fruit.level];
        return (
          <g key={fruit.id} data-fruit-id={fruit.id} transform={`translate(${fruit.x} ${fruit.y}) rotate(${(fruit.angle ?? 0) * 180 / Math.PI})`} className="lab-fruit">
            <title>{`L${fruit.level} #${fruit.id} · (${fruit.x.toFixed(1)}, ${fruit.y.toFixed(1)})`}</title>
            <circle r={radius + 4} />
            {texture
              ? <image href={texture} x={-radius} y={-radius} width={radius * 2} height={radius * 2} />
              : <circle r={radius} fill={LEVEL_COLORS[(fruit.level - 1) % LEVEL_COLORS.length]} />}
            {showLabels && <text y={4}>L{fruit.level}</text>}
          </g>
        );
      })}
      {afterFruits}
      <rect x="0" y={geometry.board_height - geometry.wall_width} width={geometry.board_width} height={geometry.wall_width} className="lab-wall" />
      <rect x="0" y="0" width={geometry.wall_width} height={geometry.board_height} className="lab-wall" />
      <rect x={geometry.board_width - geometry.wall_width} y="0" width={geometry.wall_width} height={geometry.board_height} className="lab-wall" />
    </svg>
  );
}
