package com.fruitmerge.ai.game;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class AdaptiveScoreLayoutControllerTest {
    private static final float EPSILON = 0.0001f;

    @Test
    public void resetAndSnapshotChooseExpectedDockWithoutAnimation() {
        AdaptiveScoreLayoutController controller =
                new AdaptiveScoreLayoutController();

        assertEquals(
                AdaptiveScoreLayoutController.Dock.BOTTOM,
                controller.targetDock()
        );
        assertEquals(1f, controller.bottomProgress(), EPSILON);
        assertFalse(controller.isTransitioning());

        controller.snapForPileRatio(0.50f);
        assertEquals(
                AdaptiveScoreLayoutController.Dock.TOP,
                controller.targetDock()
        );
        assertEquals(0f, controller.bottomProgress(), EPSILON);

        controller.snapForPileRatio(Float.NaN);
        assertEquals(
                AdaptiveScoreLayoutController.Dock.BOTTOM,
                controller.targetDock()
        );
        assertEquals(1f, controller.bottomProgress(), EPSILON);
    }

    @Test
    public void topRequestNeedsContinuousConfirmationAboveUpperThreshold() {
        AdaptiveScoreLayoutController controller =
                new AdaptiveScoreLayoutController();

        controller.update(0.60f, 0.64f, true);
        assertFalse(controller.isTransitioning());

        // 回到迟滞带会打断尚未确认的连续计时。
        controller.update(0.55f, 0.20f, true);
        controller.update(0.60f, 0.64f, true);
        assertFalse(controller.isTransitioning());

        controller.update(0.60f, 0.01f, true);
        assertTrue(controller.isTransitioning());
        assertEquals(
                AdaptiveScoreLayoutController.Dock.TOP,
                controller.targetDock()
        );
        // 确认阈值的这一帧只发起动画，不会瞬间吞掉同一份 delta。
        assertEquals(1f, controller.bottomProgress(), EPSILON);
    }

    @Test
    public void transitionUsesSymmetricSmootherStepAndNeverOvershoots() {
        AdaptiveScoreLayoutController controller =
                new AdaptiveScoreLayoutController();
        beginTopTransition(controller);

        float previous = controller.bottomProgress();
        for (int index = 0; index < 5; index++) {
            controller.update(0.60f, 0.10f, true);
            float current = controller.bottomProgress();
            assertTrue(current <= previous + EPSILON);
            assertTrue(current >= -EPSILON);
            assertTrue(current <= 1f + EPSILON);
            previous = current;
        }

        controller.update(0.60f, 0.12f, true);
        assertFalse(controller.isTransitioning());
        assertEquals(0f, controller.bottomProgress(), EPSILON);

        /*
         * 从顶部返回底部时仍从当前 progress 起步；在 0.62s 的正中点，
         * quintic smootherstep 恰好为 0.5。
         */
        controller.update(0.40f, 1.10f, true);
        controller.update(0.50f, 1.90f, true);
        assertTrue(controller.isTransitioning());
        assertEquals(
                AdaptiveScoreLayoutController.Dock.BOTTOM,
                controller.targetDock()
        );
        controller.update(0.40f, 0.31f, true);
        assertEquals(0.5f, controller.bottomProgress(), EPSILON);
        controller.update(0.40f, 0.31f, true);
        assertFalse(controller.isTransitioning());
        assertEquals(1f, controller.bottomProgress(), EPSILON);
    }

    @Test
    public void minimumResidenceDelaysReverseButKeepsConfirmedIntent() {
        AdaptiveScoreLayoutController controller =
                new AdaptiveScoreLayoutController();
        beginTopTransition(controller);
        controller.update(0.60f, 0.62f, true);
        assertFalse(controller.isTransitioning());

        // 下移已经确认，但刚结束上移动画，尚未满足三秒驻留。
        controller.update(0.40f, 1.10f, true);
        assertFalse(controller.isTransitioning());
        assertEquals(
                AdaptiveScoreLayoutController.Dock.TOP,
                controller.targetDock()
        );

        // 中性区不会丢掉已经确认的 pending。
        controller.update(0.50f, 1.89f, true);
        assertFalse(controller.isTransitioning());
        controller.update(0.50f, 0.02f, true);
        assertTrue(controller.isTransitioning());
        assertEquals(
                AdaptiveScoreLayoutController.Dock.BOTTOM,
                controller.targetDock()
        );
    }

    @Test
    public void blockedConfirmedIntentStartsLaterFromNeutralBand() {
        AdaptiveScoreLayoutController controller =
                new AdaptiveScoreLayoutController();

        controller.update(0.70f, 0.65f, false);
        assertFalse(controller.isTransitioning());
        assertEquals(
                AdaptiveScoreLayoutController.Dock.BOTTOM,
                controller.targetDock()
        );

        controller.update(0.50f, 4f, false);
        assertFalse(controller.isTransitioning());
        controller.update(0.50f, 0f, true);
        assertTrue(controller.isTransitioning());
        assertEquals(
                AdaptiveScoreLayoutController.Dock.TOP,
                controller.targetDock()
        );
    }

    @Test
    public void explicitReverseThresholdCancelsBlockedIntent() {
        AdaptiveScoreLayoutController controller =
                new AdaptiveScoreLayoutController();

        controller.update(0.70f, 0.65f, false);
        controller.update(0.43f, 0.01f, false);
        controller.update(0.50f, 5f, true);

        assertFalse(controller.isTransitioning());
        assertEquals(
                AdaptiveScoreLayoutController.Dock.BOTTOM,
                controller.targetDock()
        );

        // 被取消后必须重新连续满足完整的 0.65s，而不是沿用旧证据。
        controller.update(0.70f, 0.64f, true);
        assertFalse(controller.isTransitioning());
        controller.update(0.70f, 0.01f, true);
        assertTrue(controller.isTransitioning());
    }

    @Test
    public void hysteresisBandDoesNotCauseRepeatedMovement() {
        AdaptiveScoreLayoutController controller =
                new AdaptiveScoreLayoutController();

        for (int index = 0; index < 40; index++) {
            float ratio = index % 2 == 0 ? 0.45f : 0.55f;
            controller.update(ratio, 0.20f, true);
        }

        assertFalse(controller.isTransitioning());
        assertEquals(
                AdaptiveScoreLayoutController.Dock.BOTTOM,
                controller.targetDock()
        );
        assertEquals(1f, controller.bottomProgress(), EPSILON);
    }

    @Test
    public void invalidTimeAndPileSamplesCannotCorruptProgress() {
        AdaptiveScoreLayoutController controller =
                new AdaptiveScoreLayoutController();
        controller.update(0.80f, Float.NaN, true);
        controller.update(0.80f, -3f, true);
        controller.update(Float.NaN, 10f, true);
        controller.update(Float.POSITIVE_INFINITY, 10f, true);

        assertFalse(controller.isTransitioning());
        assertEquals(1f, controller.bottomProgress(), EPSILON);

        beginTopTransition(controller);
        controller.update(0.80f, 0.20f, true);
        float progress = controller.bottomProgress();
        controller.update(0.80f, -1f, true);
        controller.update(0.80f, Float.NaN, true);
        assertEquals(progress, controller.bottomProgress(), EPSILON);
        assertTrue(Float.isFinite(controller.bottomProgress()));

        // A result screen may stop supplying pile samples, but an already visible transition
        // still needs to reach its endpoint instead of leaving two half-visible score cards.
        controller.update(Float.NaN, 0.42f, false);
        assertFalse(controller.isTransitioning());
        assertEquals(0f, controller.bottomProgress(), EPSILON);
    }

    private static void beginTopTransition(
            AdaptiveScoreLayoutController controller) {
        controller.update(0.60f, 0.65f, true);
        assertTrue(controller.isTransitioning());
    }
}
