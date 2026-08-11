export type ModelRecord = {
  id: string;
  name: string;
  shortName: string;
  role: string;
  score30: number;
  score120: number;
  parameters: number;
  transitions: number;
  trainingHours: number;
  report: string;
  commit: string;
  physics: "inherited-momentum" | "zero-velocity";
  accent: string;
  evidence: string;
};

// 数值来自 docs/model_evaluations/COMPARISON_MATRIX.md；训练小时为基于
// transition/稳定吞吐的近似计算，128M使用已冻结墙钟预算，不伪装成精确实测值。
export const MODELS: ModelRecord[] = [
  {
    id: "baseline-r1",
    name: "三层历史基线",
    shortName: "3L 基线",
    role: "容量下限",
    score30: 3280.29,
    score120: 3053.43,
    parameters: 725423,
    transitions: 14.481,
    trainingHours: 0.67,
    report: "docs/model_evaluations/model-baseline-r1.md",
    commit: "0836ffd",
    physics: "inherited-momentum",
    accent: "#9aa7b9",
    evidence: "1024局/FPS",
  },
  {
    id: "baseline-l4",
    name: "四层扩容基线",
    shortName: "4L 扩容",
    role: "容量对照",
    score30: 3742.69,
    score120: 3471.08,
    parameters: 876734,
    transitions: 16.001,
    trainingHours: 0.83,
    report: "docs/model_evaluations/model-baseline-scale-v1-l4-r1.md",
    commit: "f6edba2",
    physics: "inherited-momentum",
    accent: "#78a8ef",
    evidence: "4096局/FPS",
  },
  {
    id: "baseline-l5-fast",
    name: "五层 Fast 24M",
    shortName: "5L Fast",
    role: "简洁基线",
    score30: 4268.14,
    score120: 3897.29,
    parameters: 1028045,
    transitions: 24.001,
    trainingHours: 1.45,
    report: "docs/model_evaluations/model-baseline-scale-v1-l5-epsilon-r1.md",
    commit: "56c2e3d",
    physics: "inherited-momentum",
    accent: "#4c8de8",
    evidence: "4096局/FPS",
  },
  {
    id: "rank-active-34m",
    name: "排名主动续训 34M",
    shortName: "排名主动",
    role: "主动学习实验",
    score30: 4493.78,
    score120: 4183.56,
    parameters: 1210451,
    transitions: 34,
    trainingHours: 3.08,
    report: "docs/model_evaluations/model-auxiliary-action-rank-active-r3.md",
    commit: "c01cedb",
    physics: "inherited-momentum",
    accent: "#8879e4",
    evidence: "4096局/FPS",
  },
  {
    id: "single-step-branch",
    name: "单步隔离旁路 24M+4M",
    shortName: "单步旁路",
    role: "反事实框架",
    score30: 4668.89,
    score120: 4347.91,
    parameters: 1210451,
    transitions: 28.001,
    trainingHours: 2.74,
    report: "docs/model_evaluations/model-auxiliary-action-single-step-branch-r4.md",
    commit: "bdf04ae",
    physics: "inherited-momentum",
    accent: "#e5a444",
    evidence: "4096局/FPS",
  },
  {
    id: "auxiliary-r1",
    name: "辅助动作首轮 24M",
    shortName: "辅助首轮",
    role: "24M预算最佳",
    score30: 4849.65,
    score120: 4476.35,
    parameters: 1210451,
    transitions: 24.001,
    trainingHours: 1.91,
    report: "docs/model_evaluations/model-auxiliary-action-r1.md",
    commit: "feb10ec",
    physics: "inherited-momentum",
    accent: "#46b97b",
    evidence: "4096局/FPS",
  },
  {
    id: "baseline-128m",
    name: "五层 Fast 128M",
    shortName: "5L 128M",
    role: "绝对均分基准",
    score30: 5050.52,
    score120: 4794.81,
    parameters: 1028045,
    transitions: 128.001,
    trainingHours: 12,
    report: "docs/model_evaluations/model-baseline-scale-v1-l5-fast-128m-r2.md",
    commit: "fd65767",
    physics: "inherited-momentum",
    accent: "#3478e5",
    evidence: "4096局/FPS",
  },
];

export const CURRENT_TRAINING = {
  name: "结构化辅助旁路 128M",
  status: "训练中",
  physics: "合成水果速度归零",
  budget: "128M 父轨迹 + 12M 旁路",
  report: "docs/codex/55_启动128M结构化辅助旁路训练_2026_08_11.md",
};
