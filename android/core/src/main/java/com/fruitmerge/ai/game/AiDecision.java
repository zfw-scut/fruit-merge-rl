package com.fruitmerge.ai.game;

/** Android 后台推理返回给渲染线程的一次动作选择。 */
public final class AiDecision {
    public final int actionIndex;
    public final int alternativeActionIndex;
    public final float selectedQ;
    public final float alternativeQ;
    public final float[] qValues;
    public final String source;

    public AiDecision(
            int actionIndex,
            int alternativeActionIndex,
            float selectedQ,
            float alternativeQ,
            float[] qValues,
            String source) {
        if (actionIndex < 0 || actionIndex >= FruitRules.ACTION_COUNT) {
            throw new IllegalArgumentException("invalid selected action");
        }
        this.actionIndex = actionIndex;
        this.alternativeActionIndex = alternativeActionIndex;
        this.selectedQ = selectedQ;
        this.alternativeQ = alternativeQ;
        this.qValues = qValues == null ? new float[0] : qValues.clone();
        this.source = source == null ? "unknown" : source;
    }

    /**
     * 返回前两名价值差的相对尺度。
     *
     * <p>它只用来控制“犹豫”动画，不能解释为概率或统计置信度。</p>
     */
    public float normalizedChoiceMargin() {
        if (!Float.isFinite(selectedQ) || !Float.isFinite(alternativeQ)) {
            return 0f;
        }
        float scale = Math.max(1f, Math.max(Math.abs(selectedQ), Math.abs(alternativeQ)));
        return Math.max(0f, (selectedQ - alternativeQ) / scale);
    }
}
