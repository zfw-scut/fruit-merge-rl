package com.fruitmerge.ai.game;

import org.junit.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class PhysicsDecisionClockTest {
    private static final float EPSILON = 0.000001f;

    @Test
    public void stableBoundaryUsesExactlyFifteenPhysicsFrames() {
        PhysicsDecisionClock clock = new PhysicsDecisionClock();
        for (int frame = 1; frame < FruitRules.STABLE_FRAMES; frame++) {
            assertTrue(clock.afterFrame(true, false));
            assertFalse(clock.decisionReady());
        }
        assertFalse(clock.afterFrame(true, false));
        assertTrue(clock.decisionReady());
        assertEquals(FruitRules.STABLE_FRAMES, clock.stableFrames());
    }

    @Test
    public void unstableFrameResetsOnlyStableSequence() {
        PhysicsDecisionClock clock = new PhysicsDecisionClock();
        for (int frame = 0; frame < 9; frame++) {
            clock.afterFrame(true, false);
        }
        clock.afterFrame(false, false);
        assertEquals(0, clock.stableFrames());
        assertEquals(10, clock.framesSinceDrop());
    }

    @Test
    public void timeoutStopsAtSevenHundredTwentyFrames() {
        PhysicsDecisionClock clock = new PhysicsDecisionClock();
        for (int frame = 1;
             frame < FruitRules.MAX_PHYSICS_FRAMES_PER_DROP;
             frame++) {
            assertTrue(clock.afterFrame(false, false));
        }
        assertFalse(clock.afterFrame(false, false));
        assertTrue(clock.decisionReady());
        assertFalse(clock.failed());
    }

    @Test
    public void dangerMatchesTensorStrictGreaterThanLimit() {
        PhysicsDecisionClock clock = new PhysicsDecisionClock();
        for (int frame = 0; frame < FruitRules.DANGER_FRAMES; frame++) {
            clock.afterFrame(false, true);
        }
        assertFalse(clock.failed());
        clock.afterFrame(false, true);
        assertTrue(clock.failed());
        assertEquals(FruitRules.DANGER_FRAMES + 1, clock.dangerFrames());
    }

    @Test
    public void restorePreservesPerDropTimeoutProgress() {
        PhysicsDecisionClock clock = new PhysicsDecisionClock();
        clock.restore(0.05f, 0.25f, 719);
        assertEquals(719, clock.framesSinceDrop());
        assertFalse(clock.afterFrame(false, false));
        assertTrue(clock.decisionReady());
    }

    @Test
    public void physicsObserverStopsOnExactFrameAndDropsWallClockRemainder() {
        FruitPhysicsWorld world = new FruitPhysicsWorld();
        AtomicInteger frames = new AtomicInteger();
        world.step(0.1f, (stable, danger) -> frames.incrementAndGet() < 3);
        assertEquals(3, frames.get());
        assertEquals(0f, world.snapshot().accumulatorSeconds(), EPSILON);

        world.step(1f / FruitRules.PHYSICS_FPS,
                (stable, danger) -> {
                    frames.incrementAndGet();
                    return true;
        });
        assertEquals(4, frames.get());
    }

}
