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

    private final StatefulRandom queueRandom;
    private float roundDurationSeconds;
    private float nextRoundDelaySeconds;
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
                new StatefulRandom(new Random().nextLong()),
                DEFAULT_ROUND_SECONDS,
                DEFAULT_NEXT_ROUND_DELAY_SECONDS
        );
    }

    /** 使用固定种子的默认对局，主要用于可复现的本地演示和测试。 */
    public DuelMatch(long seed) {
        this(
                new StatefulRandom(seed),
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
                new StatefulRandom(seed),
                roundDurationSeconds,
                nextRoundDelaySeconds
        );
    }

    DuelMatch(
            Random queueRandom,
            float roundDurationSeconds,
            float nextRoundDelaySeconds) {
        this(
                new StatefulRandom(Objects.requireNonNull(
                        queueRandom,
                        "queueRandom"
                ).nextLong()),
                roundDurationSeconds,
                nextRoundDelaySeconds
        );
    }

    private DuelMatch(
            StatefulRandom queueRandom,
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
     * 给任意一方更多或更少的思考时间。物理与危险线持续时间使用缩放后的 delta，
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
     * 在同一个规则边界原子提交玩家与 AI 的当前水果。
     *
     * <p>该入口用于 AI 已提前完成决策、但为了表现同步而只移动预览水果等待玩家的
     * 情况。只有双方都仍可提交时才会修改任一场景；不会出现玩家成功而 AI 因状态
     * 变化失败的半提交。本方法也只推进一次回合关闭逻辑，因此即使下一轮延迟为零，
     * 两颗水果仍会使用完全相同的当前等级。</p>
     */
    public boolean dropBoth(float playerX, float aiX) {
        return dropBoth(playerX, aiX, false);
    }

    /**
     * 倒计时归零时原子提交双方的预览位置。
     *
     * <p>与分别调用两个 timeout 方法相比，这个入口能保证损坏或重复回调不会只给
     * 一方投下水果。只要任一方已经提交，本方法就保持双方原状态并返回 false。</p>
     */
    public boolean timeoutBoth() {
        return dropBoth(player.previewX, ai.previewX, true);
    }

    /** 倒计时归零时，以指定位置原子提交双方。 */
    public boolean timeoutBoth(float playerX, float aiX) {
        return dropBoth(playerX, aiX, true);
    }

    private boolean dropBoth(float playerX, float aiX, boolean timeout) {
        requireUsable();
        if (!canDrop(player, timeout) || !canDrop(ai, timeout)) {
            return false;
        }

        addDrop(player, playerX);
        addDrop(ai, aiX);
        closeRoundIfComplete();
        return true;
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
     * 导出可持久化的完整规则快照。
     *
     * <p>快照包含共享随机序列的内部状态，因此恢复后的第五轮及更远期水果也会与
     * 原对局一致，而不只是复制当前可见的四颗队列。合成爆浆等表现层事件不属于
     * 规则快照；恢复时会清空，避免把保存前已经播放过的反馈重复播放。</p>
     */
    public Snapshot snapshot() {
        requireUsable();
        int[] queueLevels = new int[queue.size];
        for (int index = 0; index < queue.size; index++) {
            queueLevels[index] = queue.get(index);
        }
        return new Snapshot(
                queueRandom.state(),
                roundDurationSeconds,
                nextRoundDelaySeconds,
                queueLevels,
                currentLevel,
                roundIndex,
                matchGeneration,
                roundRemainingSeconds,
                nextRoundRemainingSeconds,
                roundOpen,
                outcome,
                laneSnapshot(player),
                laneSnapshot(ai)
        );
    }

    /**
     * 从完整规则快照恢复双方场景。
     *
     * <p>恢复会使 {@link #matchGeneration()} 至少递增一次，以拒绝保存或切换场景前
     * 尚未返回的异步 AI 决策。除这个纯粹用于验票的代数外，队列、计时、统计与
     * 物理状态均按快照精确恢复。</p>
     */
    public void restore(Snapshot snapshot) {
        requireUsable();
        Objects.requireNonNull(snapshot, "snapshot");

        /*
         * Snapshot 构造器已完成不变量校验，两个物理快照也都是不可变值对象。
         * 先恢复物理世界，再发布规则字段，调用方不会在方法返回前观察到半恢复。
         */
        player.physics.restore(snapshot.player.physics);
        ai.physics.restore(snapshot.ai.physics);

        queueRandom.restoreState(snapshot.queueRandomState);
        roundDurationSeconds = snapshot.roundDurationSeconds;
        nextRoundDelaySeconds = snapshot.nextRoundDelaySeconds;
        queue.clear();
        for (int level : snapshot.queueLevels) {
            queue.add(level);
        }
        currentLevel = snapshot.currentLevel;
        roundIndex = snapshot.roundIndex;
        matchGeneration = Math.max(
                matchGeneration,
                snapshot.matchGeneration
        );
        if (matchGeneration < Long.MAX_VALUE) {
            matchGeneration += 1L;
        }
        roundRemainingSeconds = snapshot.roundRemainingSeconds;
        nextRoundRemainingSeconds = snapshot.nextRoundRemainingSeconds;
        roundOpen = snapshot.roundOpen;
        outcome = snapshot.outcome;
        restoreLane(player, snapshot.player);
        restoreLane(ai, snapshot.ai);
        mergeVisualEvents.clear();
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

    /** 释放双方物理世界；重复调用安全。 */
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
        if (!canDrop(lane, timeout)) {
            return false;
        }

        addDrop(lane, x);
        closeRoundIfComplete();
        return true;
    }

    private boolean canDrop(Lane lane, boolean timeout) {
        if (outcome != Outcome.IN_PROGRESS
                || !roundOpen
                || !lane.alive
                || lane.submittedThisRound) {
            return false;
        }
        return timeout
                ? roundRemainingSeconds <= 0f
                : roundRemainingSeconds > 0f;
    }

    private void addDrop(Lane lane, float x) {
        lane.previewX = FruitRules.clampDropX(x, currentLevel);
        lane.physics.addDroppedFruit(
                currentLevel,
                lane.previewX,
                FruitRules.SPAWN_Y
        );
        lane.stepCount += 1;
        lane.submittedThisRound = true;
    }

    private void closeRoundIfComplete() {
        if (player.submittedThisRound && ai.submittedThisRound) {
            roundOpen = false;
            nextRoundRemainingSeconds = nextRoundDelaySeconds;
            if (nextRoundRemainingSeconds <= 0f) {
                advanceRound();
            }
        }
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

    private LaneSnapshot laneSnapshot(Lane lane) {
        return new LaneSnapshot(
                lane.score,
                lane.lastScore,
                lane.stepCount,
                lane.watermelonCount,
                lane.dangerSeconds,
                lane.previewX,
                lane.alive,
                lane.submittedThisRound,
                lane.physics.snapshot()
        );
    }

    private void restoreLane(Lane lane, LaneSnapshot snapshot) {
        lane.score = snapshot.score;
        lane.lastScore = snapshot.lastScore;
        lane.stepCount = snapshot.stepCount;
        lane.watermelonCount = snapshot.watermelonCount;
        lane.dangerSeconds = snapshot.dangerSeconds;
        lane.previewX = snapshot.previewX;
        lane.alive = snapshot.alive;
        lane.submittedThisRound = snapshot.submittedThisRound;
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
        boolean danger = lane.physics.isOverDangerLine();
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

    private static void requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(
                    name + " must be >= 0"
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

    /**
     * 可序列化为 JSON/Preferences 的对战规则快照。
     *
     * <p>所有数组都会防御性复制。构造器公开是为了让移动端持久化层可以从磁盘字段
     * 重建值对象，而不需要访问 DuelMatch 的私有状态。</p>
     */
    public static final class Snapshot {
        private final long queueRandomState;
        private final float roundDurationSeconds;
        private final float nextRoundDelaySeconds;
        private final int[] queueLevels;
        private final int currentLevel;
        private final int roundIndex;
        private final long matchGeneration;
        private final float roundRemainingSeconds;
        private final float nextRoundRemainingSeconds;
        private final boolean roundOpen;
        private final Outcome outcome;
        private final LaneSnapshot player;
        private final LaneSnapshot ai;

        public Snapshot(
                long queueRandomState,
                float roundDurationSeconds,
                float nextRoundDelaySeconds,
                int[] queueLevels,
                int currentLevel,
                int roundIndex,
                long matchGeneration,
                float roundRemainingSeconds,
                float nextRoundRemainingSeconds,
                boolean roundOpen,
                Outcome outcome,
                LaneSnapshot player,
                LaneSnapshot ai) {
            requireFinitePositive(
                    roundDurationSeconds,
                    "roundDurationSeconds"
            );
            requireFiniteNonNegative(
                    nextRoundDelaySeconds,
                    "nextRoundDelaySeconds"
            );
            requireNonNegative(roundIndex, "roundIndex");
            if (matchGeneration < 0L) {
                throw new IllegalArgumentException(
                        "matchGeneration must be >= 0"
                );
            }
            if ((queueRandomState & ~StatefulRandom.MASK) != 0L) {
                throw new IllegalArgumentException(
                        "queueRandomState must fit in 48 bits"
                );
            }
            requireFiniteNonNegative(
                    roundRemainingSeconds,
                    "roundRemainingSeconds"
            );
            requireFiniteNonNegative(
                    nextRoundRemainingSeconds,
                    "nextRoundRemainingSeconds"
            );
            this.outcome = Objects.requireNonNull(outcome, "outcome");
            this.player = Objects.requireNonNull(player, "player");
            this.ai = Objects.requireNonNull(ai, "ai");
            Objects.requireNonNull(queueLevels, "queueLevels");
            if (queueLevels.length != FruitRules.QUEUE_LENGTH) {
                throw new IllegalArgumentException(
                        "queueLevels must contain exactly "
                                + FruitRules.QUEUE_LENGTH + " levels"
                );
            }
            for (int level : queueLevels) {
                if (level < FruitRules.SPAWN_MIN_LEVEL
                        || level > FruitRules.SPAWN_MAX_LEVEL) {
                    throw new IllegalArgumentException(
                            "queue level is outside spawn range: " + level
                    );
                }
            }
            if (currentLevel != queueLevels[0]) {
                throw new IllegalArgumentException(
                        "currentLevel must equal the queue head"
                );
            }
            if (outcome != Outcome.IN_PROGRESS && roundOpen) {
                throw new IllegalArgumentException(
                        "a finished match cannot have an open round"
                );
            }
            if (roundOpen && nextRoundRemainingSeconds > 0f) {
                throw new IllegalArgumentException(
                        "an open round cannot have a next-round delay"
                );
            }

            this.queueRandomState = queueRandomState;
            this.roundDurationSeconds = roundDurationSeconds;
            this.nextRoundDelaySeconds = nextRoundDelaySeconds;
            this.queueLevels = queueLevels.clone();
            this.currentLevel = currentLevel;
            this.roundIndex = roundIndex;
            this.matchGeneration = matchGeneration;
            this.roundRemainingSeconds = roundRemainingSeconds;
            this.nextRoundRemainingSeconds = nextRoundRemainingSeconds;
            this.roundOpen = roundOpen;
        }

        public long queueRandomState() {
            return queueRandomState;
        }

        public float roundDurationSeconds() {
            return roundDurationSeconds;
        }

        public float nextRoundDelaySeconds() {
            return nextRoundDelaySeconds;
        }

        public int[] queueLevels() {
            return queueLevels.clone();
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

        public float roundRemainingSeconds() {
            return roundRemainingSeconds;
        }

        public float nextRoundRemainingSeconds() {
            return nextRoundRemainingSeconds;
        }

        public boolean roundOpen() {
            return roundOpen;
        }

        public Outcome outcome() {
            return outcome;
        }

        public LaneSnapshot player() {
            return player;
        }

        public LaneSnapshot ai() {
            return ai;
        }
    }

    /** 单方对战状态的不可变快照。 */
    public static final class LaneSnapshot {
        private final int score;
        private final int lastScore;
        private final int stepCount;
        private final int watermelonCount;
        private final float dangerSeconds;
        private final float previewX;
        private final boolean alive;
        private final boolean submittedThisRound;
        private final FruitPhysicsWorld.Snapshot physics;

        public LaneSnapshot(
                int score,
                int lastScore,
                int stepCount,
                int watermelonCount,
                float dangerSeconds,
                float previewX,
                boolean alive,
                boolean submittedThisRound,
                FruitPhysicsWorld.Snapshot physics) {
            requireNonNegative(score, "score");
            requireNonNegative(lastScore, "lastScore");
            requireNonNegative(stepCount, "stepCount");
            requireNonNegative(watermelonCount, "watermelonCount");
            requireFiniteNonNegative(dangerSeconds, "dangerSeconds");
            if (!Float.isFinite(previewX)) {
                throw new IllegalArgumentException(
                        "previewX must be finite"
                );
            }
            if (lastScore > score) {
                throw new IllegalArgumentException(
                        "lastScore cannot exceed score"
                );
            }
            this.score = score;
            this.lastScore = lastScore;
            this.stepCount = stepCount;
            this.watermelonCount = watermelonCount;
            this.dangerSeconds = dangerSeconds;
            this.previewX = previewX;
            this.alive = alive;
            this.submittedThisRound = submittedThisRound;
            this.physics = Objects.requireNonNull(physics, "physics");
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

        public FruitPhysicsWorld.Snapshot physics() {
            return physics;
        }
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
            level = event.level > 0 ? event.level : event.sourceLevel;
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

    /**
     * 与 {@link java.util.Random} 相同的 48 位 LCG，但显式暴露内部状态给规则快照。
     *
     * <p>这样固定 seed 的既有水果序列保持不变，同时不依赖反射访问 JDK 私有字段。</p>
     */
    private static final class StatefulRandom {
        private static final long MULTIPLIER = 0x5DEECE66DL;
        private static final long ADDEND = 0xBL;
        private static final long MASK = (1L << 48) - 1L;

        private long state;

        private StatefulRandom(long seed) {
            state = (seed ^ MULTIPLIER) & MASK;
        }

        private long state() {
            return state;
        }

        private void restoreState(long state) {
            if ((state & ~MASK) != 0L) {
                throw new IllegalArgumentException(
                        "queue random state must fit in 48 bits"
                );
            }
            this.state = state;
        }

        private int next(int bits) {
            state = (state * MULTIPLIER + ADDEND) & MASK;
            return (int) (state >>> (48 - bits));
        }

        private int nextInt(int bound) {
            if (bound <= 0) {
                throw new IllegalArgumentException("bound must be positive");
            }
            if ((bound & -bound) == bound) {
                return (int) ((bound * (long) next(31)) >> 31);
            }
            int bits;
            int value;
            do {
                bits = next(31);
                value = bits % bound;
            } while (bits - value + (bound - 1) < 0);
            return value;
        }
    }
}
