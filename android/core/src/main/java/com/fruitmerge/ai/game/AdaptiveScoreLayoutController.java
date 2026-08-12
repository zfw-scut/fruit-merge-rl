package com.fruitmerge.ai.game;

/**
 * 分数卡顶部/底部停靠的纯表现状态机。
 *
 * <p>控制器只接收已经归一化的“稳定水果堆高度”，不读取 Box2D，也不持有任何
 * libGDX 绘制对象。主界面因此可以独立决定哪些水果算稳定、什么时候因为拖动或
 * 飞分动画暂缓换位；这里仅负责迟滞阈值、连续确认、最短驻留和无过冲插值。</p>
 *
 * <p>{@link #bottomProgress()} 的语义固定为 {@code 1=底部停靠}、
 * {@code 0=顶部停靠}。渲染层可用它同时插值棋盘纵向偏移、顶部栏高度以及两个
 * 分数卡的透明度，而不会把物理坐标或模型输入一起移动。</p>
 */
public final class AdaptiveScoreLayoutController {
    static final float TOP_ENTER_RATIO = 0.56f;
    static final float BOTTOM_ENTER_RATIO = 0.44f;
    static final float TOP_CONFIRM_SECONDS = 0.65f;
    static final float BOTTOM_CONFIRM_SECONDS = 1.10f;
    static final float MIN_RESIDENCE_SECONDS = 3.0f;
    static final float TRANSITION_SECONDS = 0.62f;

    private static final float SNAP_SPLIT_RATIO = 0.50f;
    private static final float PROGRESS_EPSILON = 0.0001f;

    /** 分数卡的两个稳定停靠位。 */
    public enum Dock {
        TOP,
        BOTTOM
    }

    private Dock targetDock = Dock.BOTTOM;
    private Dock pendingDock;
    private float bottomProgress = 1f;
    private float candidateSeconds;
    private float residenceSeconds = MIN_RESIDENCE_SECONDS;
    private float transitionStartProgress = 1f;
    private float transitionElapsed;
    private boolean transitioning;

    public AdaptiveScoreLayoutController() {
        resetBottom();
    }

    /**
     * 新局固定从底部停靠开始，并允许首次高度确认后立即发起切换。
     *
     * <p>“最短驻留三秒”只约束已经发生过的换位。新局并没有刚完成一次动画，
     * 因而这里把驻留计时初始化为已满足，避免载入快速堆高的测试局时还要额外
     * 等待三秒。</p>
     */
    public void resetBottom() {
        snapTo(Dock.BOTTOM);
    }

    /**
     * 读档或直接进入预览场景时，根据当前堆高无动画地选择最近停靠位。
     *
     * <p>正常运行使用 44%/56% 迟滞带；首次展示没有“上一停靠位”可供迟滞，
     * 因此用用户可理解的 50% 中线决定初始位置。非有限输入按空局处理。</p>
     */
    public void snapForPileRatio(float pileRatio) {
        float safeRatio = sanitizePileRatio(pileRatio);
        snapTo(safeRatio >= SNAP_SPLIT_RATIO ? Dock.TOP : Dock.BOTTOM);
    }

    /**
     * 推进一次状态机。
     *
     * @param pileRatio 已排除下落水果后的归一化堆高，通常位于 {@code [0, 1]}
     * @param deltaSeconds 真实时间增量；NaN、无穷或负数按零处理
     * @param mayStartTransition 当前是否允许开始新换位。已经开始的动画不会被暂停
     */
    public void update(
            float pileRatio,
            float deltaSeconds,
            boolean mayStartTransition) {
        float delta = sanitizeDelta(deltaSeconds);
        boolean wasTransitioning = transitioning;
        advanceMotion(delta);

        /*
         * 当前帧若用于推进一段已经开始的动画，就不再把同一份 delta 重复计入
         * 反向阈值或驻留时间。特别是一个 0.62s 的大帧不能既完成上移，又立刻
         * 为下移累计 0.62s 候选时间。
         */
        if (wasTransitioning) {
            return;
        }

        if (!Float.isFinite(pileRatio)) {
            // 没有可靠高度样本时，动画和驻留照常走，但不制造或取消布局意图。
            return;
        }
        float safeRatio = clamp01(pileRatio);

        if (pendingDock != null) {
            if (isExplicitReverseEvidence(pendingDock, safeRatio)) {
                /*
                 * 已确认意图在飞分/拖动阻塞期间会保留；只有越过另一侧阈值才算
                 * 明确反证。落回 44%～56% 中性区不会让下一帧重新等待一遍。
                 */
                pendingDock = null;
                candidateSeconds = 0f;
                return;
            }
            if (mayStartTransition
                    && residenceSeconds >= MIN_RESIDENCE_SECONDS) {
                startTransition(pendingDock);
            }
            return;
        }

        Dock candidate = opposite(targetDock);
        if (!qualifiesFor(candidate, safeRatio)) {
            candidateSeconds = 0f;
            return;
        }

        candidateSeconds += delta;
        float required = candidate == Dock.TOP
                ? TOP_CONFIRM_SECONDS
                : BOTTOM_CONFIRM_SECONDS;
        if (candidateSeconds < required) {
            return;
        }

        pendingDock = candidate;
        candidateSeconds = 0f;
        if (mayStartTransition
                && residenceSeconds >= MIN_RESIDENCE_SECONDS) {
            startTransition(candidate);
        }
    }

    /**
     * 当前底部布局权重：1 表示完全停靠底部，0 表示完全停靠顶部。
     */
    public float bottomProgress() {
        return bottomProgress;
    }

    /**
     * 当前正在显示或已经获准开始动画的目标停靠位。
     *
     * <p>被阻塞的已确认 pending 不会提前改变本值，避免渲染层在飞分尚未结束时
     * 就把吸入目标或卡片颜色切到另一端。</p>
     */
    public Dock targetDock() {
        return targetDock;
    }

    public boolean isTransitioning() {
        return transitioning;
    }

    private void snapTo(Dock dock) {
        targetDock = dock;
        pendingDock = null;
        candidateSeconds = 0f;
        bottomProgress = progressFor(dock);
        transitionStartProgress = bottomProgress;
        transitionElapsed = 0f;
        transitioning = false;
        residenceSeconds = MIN_RESIDENCE_SECONDS;
    }

    private void advanceMotion(float delta) {
        if (!transitioning) {
            residenceSeconds = saturatingAdd(residenceSeconds, delta);
            return;
        }

        transitionElapsed = saturatingAdd(transitionElapsed, delta);
        float linear = Math.min(
                1f,
                transitionElapsed / TRANSITION_SECONDS
        );
        float eased = smootherStep(linear);
        float destination = progressFor(targetDock);
        bottomProgress = lerp(
                transitionStartProgress,
                destination,
                eased
        );
        /*
         * 精确钉住端点既避免累计误差，也让外层能可靠地用 isTransitioning()
         * 判断何时允许新的飞分序列或反向候选。
         */
        if (linear >= 1f) {
            bottomProgress = destination;
            transitioning = false;
            transitionElapsed = 0f;
            transitionStartProgress = destination;
            residenceSeconds = 0f;
        }
    }

    private void startTransition(Dock destination) {
        if (destination == null || destination == targetDock) {
            pendingDock = null;
            candidateSeconds = 0f;
            return;
        }
        targetDock = destination;
        pendingDock = null;
        candidateSeconds = 0f;
        transitionStartProgress = clamp01(bottomProgress);
        transitionElapsed = 0f;
        float destinationProgress = progressFor(destination);
        if (Math.abs(
                transitionStartProgress - destinationProgress
        ) <= PROGRESS_EPSILON) {
            bottomProgress = destinationProgress;
            transitioning = false;
            residenceSeconds = 0f;
            return;
        }
        transitioning = true;
    }

    private static boolean qualifiesFor(Dock dock, float ratio) {
        return dock == Dock.TOP
                ? ratio >= TOP_ENTER_RATIO
                : ratio <= BOTTOM_ENTER_RATIO;
    }

    private static boolean isExplicitReverseEvidence(
            Dock pending,
            float ratio) {
        return pending == Dock.TOP
                ? ratio <= BOTTOM_ENTER_RATIO
                : ratio >= TOP_ENTER_RATIO;
    }

    private static Dock opposite(Dock dock) {
        return dock == Dock.TOP ? Dock.BOTTOM : Dock.TOP;
    }

    private static float progressFor(Dock dock) {
        return dock == Dock.BOTTOM ? 1f : 0f;
    }

    /**
     * Quintic smootherstep 在两端的一阶、二阶导数均为零，卡片启停不会突然抽动。
     */
    private static float smootherStep(float value) {
        float x = clamp01(value);
        return x * x * x * (x * (x * 6f - 15f) + 10f);
    }

    private static float lerp(float from, float to, float amount) {
        return from + (to - from) * amount;
    }

    private static float sanitizePileRatio(float value) {
        return Float.isFinite(value) ? clamp01(value) : 0f;
    }

    private static float sanitizeDelta(float value) {
        return Float.isFinite(value) && value > 0f ? value : 0f;
    }

    private static float saturatingAdd(float value, float increment) {
        if (increment <= 0f) {
            return value;
        }
        float result = value + increment;
        return Float.isFinite(result) ? result : Float.MAX_VALUE;
    }

    private static float clamp01(float value) {
        return Math.max(0f, Math.min(1f, value));
    }
}
