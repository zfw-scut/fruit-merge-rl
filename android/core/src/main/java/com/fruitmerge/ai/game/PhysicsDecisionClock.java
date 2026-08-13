package com.fruitmerge.ai.game;

/** 以固定物理帧维护训练环境的稳定、危险和单次投放等待边界。 */
public final class PhysicsDecisionClock {
    private int stableFrames;
    private int dangerFrames;
    private int framesSinceDrop;

    public void resetGame() {
        stableFrames = 0;
        dangerFrames = 0;
        framesSinceDrop = 0;
    }

    public void resetForDrop() {
        stableFrames = 0;
        framesSinceDrop = 0;
    }

    public void resetStable() {
        stableFrames = 0;
    }

    public void restore(
            float stableSeconds,
            float dangerSeconds,
            int restoredFramesSinceDrop) {
        stableFrames = secondsToFrames(stableSeconds);
        dangerFrames = secondsToFrames(dangerSeconds);
        framesSinceDrop = Math.max(
                0,
                Math.min(
                        FruitRules.MAX_PHYSICS_FRAMES_PER_DROP,
                        restoredFramesSinceDrop
                )
        );
    }

    /** 返回 false 表示已到达失败或下一次模型决策边界，应停止继续推进物理。 */
    public boolean afterFrame(boolean stable, boolean overDangerLine) {
        framesSinceDrop = Math.min(
                FruitRules.MAX_PHYSICS_FRAMES_PER_DROP,
                framesSinceDrop + 1
        );
        stableFrames = stable
                ? Math.min(FruitRules.STABLE_FRAMES, stableFrames + 1)
                : 0;
        dangerFrames = overDangerLine
                ? Math.min(FruitRules.DANGER_FRAMES + 1, dangerFrames + 1)
                : 0;
        return !failed() && !decisionReady();
    }

    public boolean decisionReady() {
        return stableFrames >= FruitRules.STABLE_FRAMES
                || framesSinceDrop >= FruitRules.MAX_PHYSICS_FRAMES_PER_DROP;
    }

    public boolean failed() {
        // CUDA/Tensor 契约在 fail_frames > danger_frame_limit 时结束。
        return dangerFrames > FruitRules.DANGER_FRAMES;
    }

    public int stableFrames() {
        return stableFrames;
    }

    public int dangerFrames() {
        return dangerFrames;
    }

    public int framesSinceDrop() {
        return framesSinceDrop;
    }

    public float stableSeconds() {
        return stableFrames / (float) FruitRules.PHYSICS_FPS;
    }

    public float dangerSeconds() {
        return dangerFrames / (float) FruitRules.PHYSICS_FPS;
    }

    private static int secondsToFrames(float seconds) {
        if (!Float.isFinite(seconds) || seconds <= 0f) {
            return 0;
        }
        return Math.max(0, Math.round(seconds * FruitRules.PHYSICS_FPS));
    }
}
