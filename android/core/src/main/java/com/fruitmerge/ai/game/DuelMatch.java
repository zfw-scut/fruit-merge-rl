package com.fruitmerge.ai.game;

import com.badlogic.gdx.utils.Array;
import com.badlogic.gdx.utils.IntArray;

import java.util.Objects;
import java.util.Random;

/**
 * AI 对战模式的双场景规则控制器。
 *
 * <p>玩家与 AI 各自拥有完全独立的 {@link FruitPhysicsWorld} 和局内统计，但每轮
 * 共用同一颗待投水果。控制器只管理规则、物理与计分，不负责界面、AI 推理、音效
 * 或结算页；调用方可以安全地在经典模式之外逐步接入对战表现。</p>
 *
 * <p>{@link #update(float, float)} 始终先推进双方物理、消费双方合成，再在同一个
 * 帧边界统一判定胜负。这样双方同帧越线时不会因 player/AI 的更新顺序获得优势。</p>
 */
public final class DuelMatch {
    public static final float DEFAULT_ROUND_SECONDS = 5f;
    public static final float DEFAULT_NEXT_ROUND_DELAY_SECONDS = 0.24f;
    public static final float DANGER_SECONDS = 2f;

    private static final float MAX_PHYSICS_DELTA_SECONDS = 0.1f;

    private final Random queueRandom;
    private final float roundDurationSeconds;
    private final float nextRoundDelaySeconds;
    private final IntArray queue = new IntArray(FruitRules.QUEUE_LENGTH);
    private final Array<MergeVisualEvent> mergeVisualEvents = new Array<>();
    private final Lane player;
    private final Lane ai;

    private Outcome outcome = Outcome.IN_PROGRESS;
    private int currentLevel;
    private int roundIndex;
    private long matchGeneration;
    private float roundRemainingSeconds;
    private float nextRoundRemainingSeconds;
    private boolean roundOpen;
    private boolean disposed;

    /** 使用非确定种子的默认对局。 */
    public DuelMatch() {
        this(
                new Random(),
                DEFAULT_ROUND_SECONDS,
                DEFAULT_NEXT_ROUND_DELAY_SECONDS
        );
    }

    /** 使用固定种子的默认对局，主要用于可复现的本地演示和测试。 */
    public DuelMatch(long seed) {
        this(
                new Random(seed),
                DEFAULT_ROUND_SECONDS,
                DEFAULT_NEXT_ROUND_DELAY_SECONDS
        );
    }

    /**
     * 使用固定种子和自定义回合时间创建对局。
     *
     * @param seed 共享水果序列随机种子
     * @param roundDurationSeconds 每轮真实时间限制
     * @param nextRoundDelaySeconds 双方提交后的短暂停顿
     */
    public DuelMatch(
            long seed,
            float roundDurationSeconds,
            float nextRoundDelaySeconds) {
        this(
                new Random(seed),
                roundDurationSeconds,
                nextRoundDelaySeconds
        );
    }

    DuelMatch(
            Random queueRandom,
            float roundDurationSeconds,
            float nextRoundDelaySeconds) {
        this.queueRandom = Objects.requireNonNull(
                queueRandom,
                "queueRandom"
        );
        requireFinitePositive(roundDurationSeconds, "roundDurationSeconds");
        requireFiniteNonNegative(
                nextRoundDelaySeconds,
                "nextRoundDelaySeconds"
        );
        this.roundDurationSeconds = roundDurationSeconds;
        this.nextRoundDelaySeconds = nextRoundDelaySeconds;
        player = new Lane(Side.PLAYER, new FruitPhysicsWorld());
        ai = new Lane(Side.AI, new FruitPhysicsWorld());
        reset();
    }

    /**
     * 推进一帧对战。
     *
     * <p>回合倒计时使用未缩放的 {@code realDeltaSeconds}，因此改变游戏速度不会
     * 给任意一方更多或更少的思考时间。Box2D 与危险线持续时间使用缩放后的 delta，
     * 使物理、失败判定和游戏速度保持一致。</p>
     */
    public void update(float realDeltaSeconds, float gameSpeed) {
        requireUsable();
        requireFiniteNonNegative(realDeltaSeconds, "realDeltaSeconds");
        requireFinitePositive(gameSpeed, "gameSpeed");
        if (outcome != Outcome.IN_PROGRESS) {
            return;
        }

        /*
         * FruitPhysicsWorld 内部同样会限制超大 delta。这里显式限制一次，确保危险线
         * 累计时间与本帧实际交给物理世界的时间一致。
         */
        float physicsDelta = Math.min(
                realDeltaSeconds * gameSpeed,
                MAX_PHYSICS_DELTA_SECONDS
        );

        // 公平性不变量：先推进两场，再处理任何一方的输赢。
        player.physics.step(physicsDelta);
        ai.physics.step(physicsDelta);
        consumeMergeEvents(player);
        consumeMergeEvents(ai);

        boolean playerEliminatedThisFrame =
                updateDanger(player, physicsDelta);
        boolean aiEliminatedThisFrame = updateDanger(ai, physicsDelta);
        Outcome resolved = resolveOutcome(
                playerEliminatedThisFrame,
                player.score,
                aiEliminatedThisFrame,
                ai.score
        );
        if (resolved != Outcome.IN_PROGRESS) {
            player.alive = !playerEliminatedThisFrame;
            ai.alive = !aiEliminatedThisFrame;
            outcome = resolved;
            roundOpen = false;
            return;
        }

        if (roundOpen) {
            roundRemainingSeconds = Math.max(
                    0f,
                    roundRemainingSeconds - realDeltaSeconds
            );
        } else {
            nextRoundRemainingSeconds = Math.max(
                    0f,
                    nextRoundRemainingSeconds - realDeltaSeconds
            );
            if (nextRoundRemainingSeconds <= 0f) {
                advanceRound();
            }
        }
    }

    /** 玩家在本轮提交当前水果。 */
    public boolean dropPlayer(float x) {
        return drop(player, x, false);
    }

    /** AI 在本轮提交当前水果。 */
    public boolean dropAi(float x) {
        return drop(ai, x, false);
    }

    /**
     * 回合超时后，以玩家当前预览位置自动投放。
     *
     * <p>控制器只提供确定性触发点，不自行决定调用时机；界面层看到
     * {@link #roundRemainingSeconds()} 为零后调用即可。</p>
     */
    public boolean timeoutPlayer() {
        return drop(player, player.previewX, true);
    }

    /** 回合超时后，以 AI 当前预览位置自动投放。 */
    public boolean timeoutAi() {
        return drop(ai, ai.previewX, true);
    }

    /** 回合超时后，以指定位置替代玩家投放。 */
    public boolean timeoutPlayer(float x) {
        return drop(player, x, true);
    }

    /** 回合超时后，以指定位置替代 AI 投放。 */
    public boolean timeoutAi(float x) {
        return drop(ai, x, true);
    }

    /** 更新玩家尚未提交水果的预览位置。 */
    public boolean setPlayerPreviewX(float x) {
        return setPreviewX(player, x);
    }

    /** 更新 AI 尚未提交水果的预览位置。 */
    public boolean setAiPreviewX(float x) {
        return setPreviewX(ai, x);
    }

    /** 按 side 更新尚未提交水果的预览位置。 */
    public boolean setPreviewX(Side side, float x) {
        return setPreviewX(lane(side), x);
    }

    /**
     * 取走自上次 drain 后产生的双方合成表现事件。
     *
     * <p>逻辑分数已在事件进入此队列前完成结算；返回值只供爆浆、浮分和音效使用。</p>
     */
    public Array<MergeVisualEvent> drainMergeVisualEvents() {
        requireUsable();
        Array<MergeVisualEvent> drained =
                new Array<>(mergeVisualEvents);
        mergeVisualEvents.clear();
        return drained;
    }

    /**
     * 重置双方场景和局内统计。
     *
     * <p>共享随机源会继续向前，而不是回退到同一个初始种子，避免玩家重开后得到
     * 完全相同的四果序列。{@link #matchGeneration()} 会递增，调用方可据此拒绝
     * 上一局迟到的异步 AI 决策。</p>
     */
    public void reset() {
        requireUsable();
        player.physics.clear();
        ai.physics.clear();
        queue.clear();
        fillQueue();
        currentLevel = queue.first();
        player.reset(currentLevel);
        ai.reset(currentLevel);
        mergeVisualEvents.clear();
        outcome = Outcome.IN_PROGRESS;
        roundIndex = 0;
        matchGeneration += 1L;
        roundRemainingSeconds = roundDurationSeconds;
        nextRoundRemainingSeconds = 0f;
        roundOpen = true;
    }

    /** 释放双方 Box2D 世界；重复调用安全。 */
    public void dispose() {
        if (disposed) {
            return;
        }
        disposed = true;
        mergeVisualEvents.clear();
        player.physics.dispose();
        ai.physics.dispose();
    }

    public Lane playerLane() {
        return player;
    }

    public Lane aiLane() {
        return ai;
    }

    public Lane lane(Side side) {
        Objects.requireNonNull(side, "side");
        return side == Side.PLAYER ? player : ai;
    }

    public Outcome outcome() {
        return outcome;
    }

    /** 胜方；进行中或平局返回 null。 */
    public Side winner() {
        if (outcome == Outcome.PLAYER_WIN) {
            return Side.PLAYER;
        }
        if (outcome == Outcome.AI_WIN) {
            return Side.AI;
        }
        return null;
    }

    public int currentLevel() {
        return currentLevel;
    }

    public int roundIndex() {
        return roundIndex;
    }

    public long matchGeneration() {
        return matchGeneration;
    }

    public boolean roundOpen() {
        return roundOpen && outcome == Outcome.IN_PROGRESS;
    }

    public boolean awaitingNextRound() {
        return !roundOpen && outcome == Outcome.IN_PROGRESS;
    }

    public float roundRemainingSeconds() {
        return roundRemainingSeconds;
    }

    public float nextRoundRemainingSeconds() {
        return nextRoundRemainingSeconds;
    }

    public int queuedLevel(int index) {
        if (index < 0 || index >= queue.size) {
            throw new IndexOutOfBoundsException(
                    "queue index must be in [0, " + (queue.size - 1) + "]"
            );
        }
        return queue.get(index);
    }

    /** 返回防御性副本，调用方不能修改正式共享序列。 */
    public IntArray queueSnapshot() {
        return new IntArray(queue);
    }

    public float roundDurationSeconds() {
        return roundDurationSeconds;
    }

    public float nextRoundDelaySeconds() {
        return nextRoundDelaySeconds;
    }

    private boolean drop(Lane lane, float x, boolean timeout) {
        requireUsable();
        if (outcome != Outcome.IN_PROGRESS
                || !roundOpen
                || !lane.alive
                || lane.submittedThisRound) {
            return false;
        }
        if (timeout) {
            if (roundRemainingSeconds > 0f) {
                return false;
            }
        } else if (roundRemainingSeconds <= 0f) {
            return false;
        }

        lane.previewX = FruitRules.clampDropX(x, currentLevel);
        lane.physics.addDroppedFruit(
                currentLevel,
                lane.previewX,
                previewY(currentLevel)
        );
        lane.stepCount += 1;
        lane.submittedThisRound = true;

        if (player.submittedThisRound && ai.submittedThisRound) {
            roundOpen = false;
            nextRoundRemainingSeconds = nextRoundDelaySeconds;
            if (nextRoundRemainingSeconds <= 0f) {
                advanceRound();
            }
        }
        return true;
    }

    private boolean setPreviewX(Lane lane, float x) {
        requireUsable();
        if (outcome != Outcome.IN_PROGRESS
                || !roundOpen
                || !lane.alive
                || lane.submittedThisRound) {
            return false;
        }
        lane.previewX = FruitRules.clampDropX(x, currentLevel);
        return true;
    }

    private void advanceRound() {
        queue.removeIndex(0);
        fillQueue();
        currentLevel = queue.first();
        roundIndex += 1;
        roundRemainingSeconds = roundDurationSeconds;
        nextRoundRemainingSeconds = 0f;
        roundOpen = true;
        player.submittedThisRound = false;
        ai.submittedThisRound = false;
        player.previewX = FruitRules.clampDropX(
                player.previewX,
                currentLevel
        );
        ai.previewX = FruitRules.clampDropX(ai.previewX, currentLevel);
    }

    private void fillQueue() {
        while (queue.size < FruitRules.QUEUE_LENGTH) {
            queue.add(
                    FruitRules.SPAWN_MIN_LEVEL
                            + queueRandom.nextInt(
                            FruitRules.SPAWN_MAX_LEVEL
                                    - FruitRules.SPAWN_MIN_LEVEL + 1)
            );
        }
    }

    private void consumeMergeEvents(Lane lane) {
        Array<FruitPhysicsWorld.MergeEvent> events =
                lane.physics.drainMergeEvents();
        for (FruitPhysicsWorld.MergeEvent event : events) {
            lane.lastScore = lane.score;
            lane.score += event.scoreDelta;
            if (event.level == FruitRules.MAX_LEVEL) {
                lane.watermelonCount += 1;
            }
            mergeVisualEvents.add(new MergeVisualEvent(lane.side, event));
        }
    }

    private boolean updateDanger(Lane lane, float scaledDeltaSeconds) {
        if (!lane.alive) {
            return false;
        }
        boolean danger = false;
        Array<FruitPhysicsWorld.FruitBody> fruits = lane.physics.fruits();
        /*
         * 与经典移动端规则一致：跳过列表最后一颗新果，防止水果刚投下或刚合成就在
         * 生成线附近瞬间判负。双方使用各自列表，互不共享危险状态。
         */
        for (int index = 0;
                index < Math.max(0, fruits.size - 1);
                index++) {
            if (fruits.get(index).y() < FruitRules.SPAWN_Y) {
                danger = true;
                break;
            }
        }
        lane.dangerSeconds = danger
                ? lane.dangerSeconds + scaledDeltaSeconds
                : 0f;
        return lane.dangerSeconds >= DANGER_SECONDS;
    }

    static Outcome resolveOutcome(
            boolean playerEliminated,
            int playerScore,
            boolean aiEliminated,
            int aiScore) {
        if (playerEliminated && aiEliminated) {
            if (playerScore > aiScore) {
                return Outcome.PLAYER_WIN;
            }
            if (aiScore > playerScore) {
                return Outcome.AI_WIN;
            }
            return Outcome.DRAW;
        }
        if (playerEliminated) {
            return Outcome.AI_WIN;
        }
        if (aiEliminated) {
            return Outcome.PLAYER_WIN;
        }
        return Outcome.IN_PROGRESS;
    }

    private float previewY(int level) {
        return FruitRules.SPAWN_Y
                - FruitRules.displayRadius(level)
                - 12f;
    }

    private void requireUsable() {
        if (disposed) {
            throw new IllegalStateException("duel match is disposed");
        }
    }

    private static void requireFinitePositive(float value, String name) {
        if (!Float.isFinite(value) || value <= 0f) {
            throw new IllegalArgumentException(
                    name + " must be finite and > 0"
            );
        }
    }

    private static void requireFiniteNonNegative(
            float value,
            String name) {
        if (!Float.isFinite(value) || value < 0f) {
            throw new IllegalArgumentException(
                    name + " must be finite and >= 0"
            );
        }
    }

    public enum Side {
        PLAYER,
        AI
    }

    public enum Outcome {
        IN_PROGRESS,
        PLAYER_WIN,
        AI_WIN,
        DRAW
    }

    /** 单方场景的只读局内状态与物理入口。 */
    public static final class Lane {
        private final Side side;
        private final FruitPhysicsWorld physics;
        private int score;
        private int lastScore;
        private int stepCount;
        private int watermelonCount;
        private float dangerSeconds;
        private float previewX;
        private boolean alive;
        private boolean submittedThisRound;

        private Lane(Side side, FruitPhysicsWorld physics) {
            this.side = side;
            this.physics = physics;
        }

        private void reset(int level) {
            score = 0;
            lastScore = 0;
            stepCount = 0;
            watermelonCount = 0;
            dangerSeconds = 0f;
            previewX = FruitRules.clampDropX(
                    FruitRules.BOARD_WIDTH / 2f,
                    level
            );
            alive = true;
            submittedThisRound = false;
        }

        public Side side() {
            return side;
        }

        /** 渲染与 AI 快照可只读访问该方自己的物理世界。 */
        public FruitPhysicsWorld physics() {
            return physics;
        }

        public int score() {
            return score;
        }

        public int lastScore() {
            return lastScore;
        }

        public int stepCount() {
            return stepCount;
        }

        public int watermelonCount() {
            return watermelonCount;
        }

        public float dangerSeconds() {
            return dangerSeconds;
        }

        public float previewX() {
            return previewX;
        }

        public boolean alive() {
            return alive;
        }

        public boolean submittedThisRound() {
            return submittedThisRound;
        }
    }

    /** 带场景归属的不可变合成表现事件。 */
    public static final class MergeVisualEvent {
        private final Side side;
        private final int fruitId;
        private final int sourceFruitIdA;
        private final int sourceFruitIdB;
        private final int level;
        private final float x;
        private final float y;
        private final int scoreDelta;

        private MergeVisualEvent(
                Side side,
                FruitPhysicsWorld.MergeEvent event) {
            this.side = side;
            fruitId = event.fruitId;
            sourceFruitIdA = event.sourceFruitIdA;
            sourceFruitIdB = event.sourceFruitIdB;
            level = event.level;
            x = event.x;
            y = event.y;
            scoreDelta = event.scoreDelta;
        }

        public Side side() {
            return side;
        }

        public int fruitId() {
            return fruitId;
        }

        public int sourceFruitIdA() {
            return sourceFruitIdA;
        }

        public int sourceFruitIdB() {
            return sourceFruitIdB;
        }

        public int level() {
            return level;
        }

        public float x() {
            return x;
        }

        public float y() {
            return y;
        }

        public int scoreDelta() {
            return scoreDelta;
        }
    }
}
