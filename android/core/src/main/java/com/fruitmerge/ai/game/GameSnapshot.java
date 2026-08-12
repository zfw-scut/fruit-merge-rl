package com.fruitmerge.ai.game;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 一次稳定边界上的移动端只读局面。
 *
 * <p>该对象只包含普通 Java 数值，Android 层会直接构造 SAB-T120 的定长 ONNX
 * 输入；物理对象和 libGDX 类型都不会越过推理边界。</p>
 */
public final class GameSnapshot {
    public final int score;
    public final int lastScore;
    public final int stepCount;
    public final float dangerProgress;
    public final boolean overDangerLine;
    public final int[] queue;
    public final List<FruitSnapshot> fruits;

    public GameSnapshot(
            int score,
            int lastScore,
            int stepCount,
            float dangerProgress,
            boolean overDangerLine,
            int[] queue,
            List<FruitSnapshot> fruits) {
        this.score = score;
        this.lastScore = lastScore;
        this.stepCount = stepCount;
        this.dangerProgress = dangerProgress;
        this.overDangerLine = overDangerLine;
        this.queue = queue.clone();
        this.fruits = Collections.unmodifiableList(new ArrayList<>(fruits));
    }

    /** 单颗水果的静态和运动字段，命名与 Python {@code FruitState} 对齐。 */
    public static final class FruitSnapshot {
        public final int slot;
        public final int id;
        public final int level;
        public final float displayRadius;
        public final float physicsRadius;
        public final float x;
        public final float y;
        public final float vx;
        public final float vy;
        public final float angle;
        public final float angularVelocity;
        public final int ageFrames;
        public final boolean stable;

        public FruitSnapshot(
                int slot,
                int id,
                int level,
                float displayRadius,
                float physicsRadius,
                float x,
                float y,
                float vx,
                float vy,
                float angle,
                float angularVelocity,
                int ageFrames,
                boolean stable) {
            this.slot = slot;
            this.id = id;
            this.level = level;
            this.displayRadius = displayRadius;
            this.physicsRadius = physicsRadius;
            this.x = x;
            this.y = y;
            this.vx = vx;
            this.vy = vy;
            this.angle = angle;
            this.angularVelocity = angularVelocity;
            this.ageFrames = ageFrames;
            this.stable = stable;
        }
    }
}
