package com.fruitmerge.ai.game;

import com.badlogic.gdx.utils.Array;
import com.badlogic.gdx.utils.IntArray;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public final class DuelMatchTest {
    @Test
    public void bothLanesReceiveTheSameFruitAndQueueAdvancesOnce() {
        DuelMatch match = new DuelMatch(41L, 3f, 0.05f);
        try {
            assertEquals(FruitRules.QUEUE_LENGTH, match.queueSnapshot().size);
            int firstLevel = match.currentLevel();
            int secondLevel = match.queuedLevel(1);

            assertTrue(match.dropPlayer(120f));
            assertFalse(match.dropPlayer(220f));
            assertTrue(match.dropAi(430f));
            assertTrue(match.awaitingNextRound());
            assertEquals(
                    firstLevel,
                    match.playerLane().physics().fruits().first().level
            );
            assertEquals(
                    firstLevel,
                    match.aiLane().physics().fruits().first().level
            );

            match.update(0.06f, 1f);
            assertTrue(match.roundOpen());
            assertEquals(1, match.roundIndex());
            assertEquals(secondLevel, match.currentLevel());
            assertEquals(FruitRules.QUEUE_LENGTH, match.queueSnapshot().size);
            assertFalse(match.playerLane().submittedThisRound());
            assertFalse(match.aiLane().submittedThisRound());
        } finally {
            match.dispose();
        }
    }

    @Test
    public void atomicDropSubmitsBothSidesAndAdvancesExactlyOnce() {
        DuelMatch match = new DuelMatch(409L, 3f, 0f);
        try {
            int firstLevel = match.currentLevel();
            int secondLevel = match.queuedLevel(1);

            assertTrue(match.dropBoth(96f, 464f));

            assertEquals(1, match.playerLane().physics().fruits().size);
            assertEquals(1, match.aiLane().physics().fruits().size);
            assertEquals(
                    firstLevel,
                    match.playerLane().physics().fruits().first().level
            );
            assertEquals(
                    firstLevel,
                    match.aiLane().physics().fruits().first().level
            );
            assertEquals(1, match.playerLane().stepCount());
            assertEquals(1, match.aiLane().stepCount());
            assertEquals(1, match.roundIndex());
            assertEquals(secondLevel, match.currentLevel());
            assertTrue(match.roundOpen());
            assertFalse(match.playerLane().submittedThisRound());
            assertFalse(match.aiLane().submittedThisRound());
        } finally {
            match.dispose();
        }
    }

    @Test
    public void atomicDropIsAllOrNothingWhenEitherSideAlreadySubmitted() {
        DuelMatch match = new DuelMatch(509L);
        try {
            assertTrue(match.dropAi(320f));
            assertFalse(match.dropBoth(120f, 420f));

            assertEquals(0, match.playerLane().stepCount());
            assertEquals(0, match.playerLane().physics().fruits().size);
            assertEquals(1, match.aiLane().stepCount());
            assertEquals(1, match.aiLane().physics().fruits().size);
            assertFalse(match.playerLane().submittedThisRound());
            assertTrue(match.aiLane().submittedThisRound());
            assertTrue(match.roundOpen());
        } finally {
            match.dispose();
        }
    }

    @Test
    public void atomicTimeoutOnlySubmitsAfterTheClockExpires() {
        DuelMatch match = new DuelMatch(609L, 0.1f, 0.2f);
        try {
            match.setPlayerPreviewX(104f);
            match.setAiPreviewX(456f);
            assertFalse(match.timeoutBoth());

            match.update(0.1f, 1f);

            assertTrue(match.timeoutBoth());
            assertEquals(1, match.playerLane().stepCount());
            assertEquals(1, match.aiLane().stepCount());
            assertTrue(match.awaitingNextRound());
        } finally {
            match.dispose();
        }
    }

    @Test
    public void fixedSeedProducesOneDeterministicSharedSequence() {
        DuelMatch first = new DuelMatch(923L);
        DuelMatch second = new DuelMatch(923L);
        try {
            IntArray firstQueue = first.queueSnapshot();
            IntArray secondQueue = second.queueSnapshot();
            assertEquals(firstQueue.size, secondQueue.size);
            for (int index = 0; index < firstQueue.size; index++) {
                assertEquals(firstQueue.get(index), secondQueue.get(index));
            }
        } finally {
            first.dispose();
            second.dispose();
        }
    }

    @Test
    public void roundClockUsesRealDeltaAndTimeoutIsCallerTriggered() {
        DuelMatch match = new DuelMatch(7L, 1f, 0.1f);
        try {
            match.update(0.25f, 3f);
            assertEquals(
                    0.75f,
                    match.roundRemainingSeconds(),
                    0.0001f
            );
            assertFalse(match.timeoutPlayer());

            match.update(0.75f, 0.5f);
            assertEquals(0f, match.roundRemainingSeconds(), 0.0001f);
            assertTrue(match.timeoutPlayer(150f));
            assertTrue(match.timeoutAi(410f));
            assertTrue(match.awaitingNextRound());
        } finally {
            match.dispose();
        }
    }

    @Test
    public void simultaneousEliminationUsesFinalScoresAndCanDraw() {
        assertEquals(
                DuelMatch.Outcome.PLAYER_WIN,
                DuelMatch.resolveOutcome(true, 800, true, 700)
        );
        assertEquals(
                DuelMatch.Outcome.AI_WIN,
                DuelMatch.resolveOutcome(true, 700, true, 800)
        );
        assertEquals(
                DuelMatch.Outcome.DRAW,
                DuelMatch.resolveOutcome(true, 800, true, 800)
        );
        assertEquals(
                DuelMatch.Outcome.AI_WIN,
                DuelMatch.resolveOutcome(true, 9_999, false, 0)
        );
        assertEquals(
                DuelMatch.Outcome.PLAYER_WIN,
                DuelMatch.resolveOutcome(false, 0, true, 9_999)
        );
        assertEquals(
                DuelMatch.Outcome.IN_PROGRESS,
                DuelMatch.resolveOutcome(false, 0, false, 0)
        );
    }

    @Test
    public void mergeEventsKeepTheirSideAndUpdateWatermelonStatistics() {
        DuelMatch match = new DuelMatch(17L);
        try {
            match.playerLane().physics().addDroppedFruit(10, 220f, 800f);
            match.playerLane().physics().addDroppedFruit(10, 340f, 800f);

            match.update(0.02f, 1f);

            Array<DuelMatch.MergeVisualEvent> events =
                    match.drainMergeVisualEvents();
            assertEquals(1, events.size);
            assertEquals(DuelMatch.Side.PLAYER, events.first().side());
            assertEquals(FruitRules.MAX_LEVEL, events.first().level());
            assertEquals(55, events.first().scoreDelta());
            assertEquals(55, match.playerLane().score());
            assertEquals(1, match.playerLane().watermelonCount());
            assertEquals(0, match.aiLane().score());
        } finally {
            match.dispose();
        }
    }

    @Test
    public void resetClearsBothLanesAndInvalidatesOldAiTickets() {
        DuelMatch match = new DuelMatch(13L);
        try {
            long generation = match.matchGeneration();
            assertTrue(match.dropPlayer(80f));
            assertTrue(match.playerLane().submittedThisRound());

            match.reset();

            assertTrue(match.matchGeneration() > generation);
            assertEquals(0, match.roundIndex());
            assertEquals(0, match.playerLane().stepCount());
            assertEquals(0, match.aiLane().stepCount());
            assertEquals(0, match.playerLane().physics().fruits().size);
            assertEquals(0, match.aiLane().physics().fruits().size);
            assertTrue(match.playerLane().alive());
            assertTrue(match.aiLane().alive());
            assertNull(match.winner());
        } finally {
            match.dispose();
        }
    }

    @Test
    public void snapshotRestoresBothLanesClocksAndFutureSharedSequence() {
        DuelMatch original = new DuelMatch(7123L, 2.5f, 0.2f);
        DuelMatch restored = new DuelMatch(99L, 9f, 1f);
        try {
            original.setPlayerPreviewX(118f);
            original.setAiPreviewX(446f);
            assertTrue(original.dropBoth(118f, 446f));
            original.update(0.075f, 1.25f);
            DuelMatch.Snapshot saved = original.snapshot();

            restored.restore(saved);

            assertEquals(
                    saved.roundDurationSeconds(),
                    restored.roundDurationSeconds(),
                    0f
            );
            assertEquals(
                    saved.nextRoundDelaySeconds(),
                    restored.nextRoundDelaySeconds(),
                    0f
            );
            assertEquals(saved.currentLevel(), restored.currentLevel());
            assertEquals(saved.roundIndex(), restored.roundIndex());
            assertEquals(
                    saved.roundRemainingSeconds(),
                    restored.roundRemainingSeconds(),
                    0f
            );
            assertEquals(
                    saved.nextRoundRemainingSeconds(),
                    restored.nextRoundRemainingSeconds(),
                    0f
            );
            assertEquals(saved.roundOpen(), restored.roundOpen());
            assertEquals(saved.outcome(), restored.outcome());
            assertTrue(
                    restored.matchGeneration() > saved.matchGeneration()
            );
            assertLaneEquals(
                    original.playerLane(),
                    restored.playerLane()
            );
            assertLaneEquals(original.aiLane(), restored.aiLane());

            /*
             * 四颗可见队列耗尽后仍保持一致，验证保存的是 PRNG 状态而不只是当前
             * queue 数组。
             */
            for (int round = 0; round < 8; round++) {
                original.update(0.2f, 1f);
                restored.update(0.2f, 1f);
                assertEquals(original.roundOpen(), restored.roundOpen());
                assertEquals(original.currentLevel(), restored.currentLevel());
                assertQueueEquals(
                        original.queueSnapshot(),
                        restored.queueSnapshot()
                );
                assertTrue(original.dropBoth(140f, 420f));
                assertTrue(restored.dropBoth(140f, 420f));
            }
        } finally {
            original.dispose();
            restored.dispose();
        }
    }

    @Test
    public void snapshotArraysAreDefensiveAndRestoreClearsVisualEvents() {
        DuelMatch match = new DuelMatch(971L);
        try {
            match.playerLane().physics().addDroppedFruit(10, 220f, 800f);
            match.playerLane().physics().addDroppedFruit(10, 340f, 800f);
            match.update(0.02f, 1f);
            assertEquals(1, match.snapshot().player().watermelonCount());

            DuelMatch.Snapshot saved = match.snapshot();
            int expectedFirst = saved.currentLevel();
            int[] exposedQueue = saved.queueLevels();
            exposedQueue[0] = expectedFirst == FruitRules.SPAWN_MIN_LEVEL
                    ? FruitRules.SPAWN_MAX_LEVEL
                    : FruitRules.SPAWN_MIN_LEVEL;
            assertNotEquals(exposedQueue[0], saved.queueLevels()[0]);

            match.restore(saved);
            assertEquals(expectedFirst, match.currentLevel());
            assertEquals(0, match.drainMergeVisualEvents().size);
        } finally {
            match.dispose();
        }
    }

    private static void assertLaneEquals(
            DuelMatch.Lane expected,
            DuelMatch.Lane actual) {
        assertEquals(expected.score(), actual.score());
        assertEquals(expected.lastScore(), actual.lastScore());
        assertEquals(expected.stepCount(), actual.stepCount());
        assertEquals(expected.watermelonCount(), actual.watermelonCount());
        assertEquals(expected.dangerSeconds(), actual.dangerSeconds(), 0f);
        assertEquals(expected.previewX(), actual.previewX(), 0f);
        assertEquals(expected.alive(), actual.alive());
        assertEquals(
                expected.submittedThisRound(),
                actual.submittedThisRound()
        );
        assertPhysicsEquals(
                expected.physics().snapshot(),
                actual.physics().snapshot()
        );
    }

    private static void assertPhysicsEquals(
            FruitPhysicsWorld.Snapshot expected,
            FruitPhysicsWorld.Snapshot actual) {
        assertEquals(expected.nextFruitId(), actual.nextFruitId());
        assertEquals(
                expected.accumulatorSeconds(),
                actual.accumulatorSeconds(),
                0f
        );
        assertEquals(expected.fruitCount(), actual.fruitCount());
        for (int index = 0; index < expected.fruitCount(); index++) {
            FruitPhysicsWorld.FruitState first = expected.fruit(index);
            FruitPhysicsWorld.FruitState second = actual.fruit(index);
            assertEquals(first.id(), second.id());
            assertEquals(first.level(), second.level());
            assertEquals(first.displayRadius(), second.displayRadius(), 0f);
            assertEquals(first.physicsRadius(), second.physicsRadius(), 0f);
            assertEquals(first.x(), second.x(), 0f);
            assertEquals(first.y(), second.y(), 0f);
            assertEquals(first.vx(), second.vx(), 0f);
            assertEquals(first.vy(), second.vy(), 0f);
            assertEquals(first.angle(), second.angle(), 0f);
            assertEquals(
                    first.angularVelocity(),
                    second.angularVelocity(),
                    0f
            );
            assertEquals(first.ageFrames(), second.ageFrames());
        }
    }

    private static void assertQueueEquals(
            IntArray expected,
            IntArray actual) {
        assertEquals(expected.size, actual.size);
        for (int index = 0; index < expected.size; index++) {
            assertEquals(expected.get(index), actual.get(index));
        }
    }
}
