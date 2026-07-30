package com.fruitmerge.ai.game;

import com.badlogic.gdx.utils.Array;
import com.badlogic.gdx.utils.IntArray;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
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
            assertEquals(100, events.first().scoreDelta());
            assertEquals(100, match.playerLane().score());
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
}
