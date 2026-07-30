package com.fruitmerge.ai.game;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class AiReactionGeometryTest {
    @Test
    public void detectsFruitCirclesInsideTouchingAndOutsideBubble() {
        assertTrue(FruitMergeApplication.circleIntersectsRectangle(
                280f, 315f, 30f,
                24f, 270f, 536f, 360f
        ));
        assertTrue(FruitMergeApplication.circleIntersectsRectangle(
                550f, 315f, 14f,
                24f, 270f, 536f, 360f
        ));
        assertFalse(FruitMergeApplication.circleIntersectsRectangle(
                560f, 400f, 20f,
                24f, 270f, 536f, 360f
        ));
    }
}
