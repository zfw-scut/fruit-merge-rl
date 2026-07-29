package com.fruitmerge.ai.game;

/**
 * Android 游戏与训练环境共享的稳定规则常量。
 *
 * <p>数组内容逐项对应 {@code src/daxigua/core/rules.py}。移动端不依赖桌面 Python
 * 物理层，因此必须把极少量运行期规则冻结在这里；黄金状态测试会继续校验几何和
 * 模型输入，避免两端悄悄漂移。</p>
 */
public final class FruitRules {
    public static final int MIN_LEVEL = 1;
    public static final int MAX_LEVEL = 11;
    public static final int SPAWN_MIN_LEVEL = 1;
    public static final int SPAWN_MAX_LEVEL = 4;
    public static final int QUEUE_LENGTH = 4;
    public static final int ACTION_COUNT = 21;

    public static final float BOARD_WIDTH = 560f;
    public static final float BOARD_HEIGHT = 1120f;
    public static final float SPAWN_Y = 252f;
    public static final float WALL_WIDTH = 20f;
    public static final float FLOOR_Y = BOARD_HEIGHT - WALL_WIDTH;

    public static final float GRAVITY_PIXELS_PER_SECOND_SQUARED = 1800f;
    public static final float STABLE_VELOCITY_PIXELS_PER_SECOND = 35f;
    public static final float STABLE_ANGULAR_VELOCITY = 4f;
    public static final float STABLE_WINDOW_SECONDS = 0.2f;

    private static final int[] DISPLAY_RADII = {
            0,
            20, 30, 42, 46, 58, 70, 74, 100, 118, 120, 156
    };

    private FruitRules() {
        // 规则类只提供静态入口，禁止创建没有意义的实例。
    }

    public static float displayRadius(int level) {
        requireLevel(level);
        return DISPLAY_RADII[level];
    }

    public static float droppedPhysicsRadius(int level) {
        float radius = displayRadius(level);
        return radius - radius % 5f;
    }

    public static float mergedPhysicsRadius(int level) {
        return displayRadius(level) - 1f;
    }

    public static float mass(int level) {
        return Math.max(1f, (float) Math.floor(displayRadius(level) / 10f));
    }

    public static int mergeScore(int resultingLevel) {
        requireLevel(resultingLevel);
        if (resultingLevel < MAX_LEVEL) {
            return resultingLevel;
        }
        return 100;
    }

    public static float clampDropX(float x, int level) {
        float radius = displayRadius(level);
        float left = WALL_WIDTH + radius + 2f;
        float right = BOARD_WIDTH - WALL_WIDTH - radius - 2f;
        return Math.max(left, Math.min(right, x));
    }

    public static float actionDropX(int actionIndex, int level) {
        if (actionIndex < 0 || actionIndex >= ACTION_COUNT) {
            throw new IllegalArgumentException("action index must be in [0, 20]");
        }
        float radius = displayRadius(level);
        float left = WALL_WIDTH + radius + 2f;
        float right = BOARD_WIDTH - WALL_WIDTH - radius - 2f;
        return left + (right - left) * actionIndex / (ACTION_COUNT - 1f);
    }

    private static void requireLevel(int level) {
        if (level < MIN_LEVEL || level > MAX_LEVEL) {
            throw new IllegalArgumentException("fruit level must be in [1, 11]");
        }
    }
}
