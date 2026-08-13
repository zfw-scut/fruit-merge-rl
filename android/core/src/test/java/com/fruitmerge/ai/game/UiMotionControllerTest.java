package com.fruitmerge.ai.game;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public final class UiMotionControllerTest {
    @Test
    public void pressDragAndReleaseCreateBoundedVisualFeedback() {
        UiMotionController controller = new UiMotionController();
        UiMotionController.Bounds bounds =
                new UiMotionController.Bounds(10f, 20f, 100f, 50f);

        assertTrue(controller.begin("play", bounds, 0, 30f, 40f));
        controller.update(0.05f);
        assertTrue(controller.visual("play").scale < 1f);

        assertTrue(controller.drag(0, 400f, -300f));
        controller.update(0.1f);
        UiMotionController.Visual dragged = controller.visual("play");
        assertTrue(Math.abs(dragged.offsetX) <= 8.01f);
        assertTrue(Math.abs(dragged.offsetY) <= 8.01f);

        assertNull(controller.release(0, 400f, -300f));
        assertFalse(controller.hasActiveControl());
    }

    @Test
    public void releaseWithinSlopCommitsAndSprings() {
        UiMotionController controller = new UiMotionController();
        UiMotionController.Bounds bounds =
                new UiMotionController.Bounds(10f, 20f, 100f, 50f);

        assertTrue(controller.begin("settings", bounds, 3, 20f, 30f));
        assertEquals("settings", controller.release(3, 116f, 40f));
        controller.update(0.04f);
        assertTrue(controller.visual("settings").releasePulse > 0f);
    }

    @Test
    public void secondPointerCannotStealActiveControl() {
        UiMotionController controller = new UiMotionController();
        UiMotionController.Bounds bounds =
                new UiMotionController.Bounds(0f, 0f, 40f, 40f);

        assertTrue(controller.begin("first", bounds, 0, 10f, 10f));
        assertFalse(controller.begin("second", bounds, 1, 10f, 10f));
        assertFalse(controller.cancel(1));
        assertTrue(controller.cancel(0));
    }

    @Test
    public void cancelledControlCanBePressedAgainWithFullFeedback() {
        UiMotionController controller = new UiMotionController();
        UiMotionController.Bounds bounds =
                new UiMotionController.Bounds(0f, 0f, 80f, 44f);

        assertTrue(controller.begin("resume", bounds, 0, 20f, 20f));
        assertTrue(controller.cancel(0));
        controller.update(0.1f);

        assertTrue(controller.begin("resume", bounds, 1, 30f, 22f));
        controller.update(0.05f);
        assertTrue(controller.visual("resume").scale < 1f);
        assertEquals("resume", controller.release(1, 30f, 22f));
    }
}
