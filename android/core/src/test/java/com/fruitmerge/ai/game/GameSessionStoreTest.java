package com.fruitmerge.ai.game;

import com.badlogic.gdx.Preferences;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public final class GameSessionStoreTest {
    @Test
    public void soloSessionRoundTripsEveryRuleFieldAndPhysicsBody() {
        MemoryPreferences preferences = new MemoryPreferences();
        FruitPhysicsWorld world = new FruitPhysicsWorld();
        try {
            world.addDroppedFruit(2, 132f, 540f);
            world.step(0.013f);
            GameSessionStore.SingleState state = singleState(
                    world.snapshot(),
                    120,
                    false
            );
            GameSessionStore.Session session =
                    GameSessionStore.Session.single(
                            71L,
                            12_345L,
                            GameSessionStore.Mode.SOLO,
                            state
                    );

            new GameSessionStore(preferences).save(session);
            GameSessionStore.Session loaded =
                    new GameSessionStore(preferences).load();

            assertNotNull(loaded);
            assertEquals(71L, loaded.sessionId());
            assertEquals(12_345L, loaded.savedAtEpochMillis());
            assertEquals(GameSessionStore.Mode.SOLO, loaded.mode());
            assertNull(loaded.duel());
            assertSingleEquals(state, loaded.single());
            assertTrue(preferences.flushCount >= 2);
        } finally {
            world.dispose();
        }
    }

    @Test
    public void aiDemoUsesTheSingleBoardPayloadWithoutBecomingManual() {
        MemoryPreferences preferences = new MemoryPreferences();
        FruitPhysicsWorld world = new FruitPhysicsWorld();
        try {
            GameSessionStore.Session session =
                    GameSessionStore.Session.single(
                            72L,
                            99L,
                            GameSessionStore.Mode.AI_DEMO,
                            singleState(world.snapshot(), 35, false)
                    );
            GameSessionStore store = new GameSessionStore(preferences);

            store.save(session);
            store.reload();

            assertTrue(store.hasSavedSession());
            assertEquals(
                    GameSessionStore.Mode.AI_DEMO,
                    store.load().mode()
            );
            assertNotNull(store.load().single());
        } finally {
            world.dispose();
        }
    }

    @Test
    public void duelSessionRoundTripsBothWorldsAndStagedAiDecision() {
        MemoryPreferences preferences = new MemoryPreferences();
        DuelMatch match = new DuelMatch(818L, 7f, 0.2f);
        DuelMatch restored = new DuelMatch(11L);
        try {
            match.setPlayerPreviewX(116f);
            match.setAiPreviewX(444f);
            assertTrue(match.dropBoth(116f, 444f));
            match.update(0.25f, 1.25f);
            assertTrue(match.roundOpen());
            assertFalse(match.playerLane().submittedThisRound());
            assertFalse(match.aiLane().submittedThisRound());
            assertTrue(match.setAiPreviewX(392f));
            GameSessionStore.DuelState state =
                    new GameSessionStore.DuelState(
                            match.snapshot(),
                            GameSessionStore.Side.AI,
                            0f,
                            false,
                            false,
                            0,
                            "",
                            true,
                            392f
                    );
            GameSessionStore.Session session =
                    GameSessionStore.Session.duel(
                            73L,
                            4_321L,
                            state
                    );

            GameSessionStore store = new GameSessionStore(preferences);
            store.save(session);
            GameSessionStore.Session loaded =
                    new GameSessionStore(preferences).load();

            assertNotNull(loaded);
            assertEquals(GameSessionStore.Mode.DUEL, loaded.mode());
            assertNull(loaded.single());
            assertEquals(
                    GameSessionStore.Side.AI,
                    loaded.duel().foreground()
            );
            assertTrue(loaded.duel().aiArmed());
            assertEquals(392f, loaded.duel().aiArmedX(), 0f);
            assertDuelSnapshotEquals(
                    state.match(),
                    loaded.duel().match()
            );

            restored.restore(loaded.duel().match());
            assertEquals(
                    match.playerLane().physics().fruits().size,
                    restored.playerLane().physics().fruits().size
            );
            assertEquals(
                    match.aiLane().physics().fruits().size,
                    restored.aiLane().physics().fruits().size
            );
        } finally {
            match.dispose();
            restored.dispose();
        }
    }

    @Test
    public void corruptedNewestBankFallsBackToPreviousCompleteSave() {
        MemoryPreferences preferences = new MemoryPreferences();
        FruitPhysicsWorld world = new FruitPhysicsWorld();
        try {
            GameSessionStore store = new GameSessionStore(preferences);
            store.save(GameSessionStore.Session.single(
                    74L,
                    100L,
                    GameSessionStore.Mode.SOLO,
                    singleState(world.snapshot(), 10, false)
            ));
            store.save(GameSessionStore.Session.single(
                    74L,
                    200L,
                    GameSessionStore.Mode.SOLO,
                    singleState(world.snapshot(), 90, false)
            ));
            preferences.putString(
                    "session.bank.1.payload",
                    "not-base64"
            );

            GameSessionStore recovered =
                    new GameSessionStore(preferences);

            assertTrue(recovered.hasSavedSession());
            assertEquals(10, recovered.load().single().score());
            assertEquals(
                    0,
                    preferences.getInteger("session.active_bank", -1)
            );
        } finally {
            world.dispose();
        }
    }

    @Test
    public void checksumTamperingInBothBanksClearsTheLogicalSlot() {
        MemoryPreferences preferences = new MemoryPreferences();
        FruitPhysicsWorld world = new FruitPhysicsWorld();
        try {
            GameSessionStore store = new GameSessionStore(preferences);
            store.save(GameSessionStore.Session.single(
                    75L,
                    100L,
                    GameSessionStore.Mode.SOLO,
                    singleState(world.snapshot(), 10, false)
            ));
            store.save(GameSessionStore.Session.single(
                    75L,
                    200L,
                    GameSessionStore.Mode.SOLO,
                    singleState(world.snapshot(), 20, false)
            ));
            preferences.putString(
                    "session.bank.0.sha256",
                    "0".repeat(64)
            );
            preferences.putString(
                    "session.bank.1.sha256",
                    "f".repeat(64)
            );

            GameSessionStore corrupted =
                    new GameSessionStore(preferences);

            assertFalse(corrupted.hasSavedSession());
            assertNull(corrupted.load());
            assertTrue(preferences.get().isEmpty());
        } finally {
            world.dispose();
        }
    }

    @Test
    public void invalidSchemaAndClearNeverExposeStaleProgress() {
        MemoryPreferences preferences = new MemoryPreferences();
        FruitPhysicsWorld world = new FruitPhysicsWorld();
        try {
            GameSessionStore store = new GameSessionStore(preferences);
            store.save(GameSessionStore.Session.single(
                    76L,
                    100L,
                    GameSessionStore.Mode.SOLO,
                    singleState(world.snapshot(), 10, false)
            ));
            preferences.putInteger("session.schema", 99);

            GameSessionStore incompatible =
                    new GameSessionStore(preferences);
            assertNull(incompatible.load());
            assertTrue(preferences.get().isEmpty());

            store.reload();
            assertFalse(store.hasSavedSession());
            store.clear();
            assertFalse(store.hasSavedSession());
        } finally {
            world.dispose();
        }
    }

    @Test
    public void stateArraysAreDefensiveAndInvalidStateCannotBeCreated() {
        FruitPhysicsWorld world = new FruitPhysicsWorld();
        try {
            int[] queue = {2, 3, 1, 4};
            GameSessionStore.SingleState state =
                    new GameSessionStore.SingleState(
                            GameSessionStore.NO_RANDOM_STATE,
                            queue,
                            2,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            200f,
                            200f,
                            0f,
                            0f,
                            0f,
                            0f,
                            true,
                            true,
                            false,
                            0,
                            world.snapshot()
                    );
            queue[0] = 4;
            int[] exposed = state.queueLevels();
            exposed[0] = 1;

            assertArrayEquals(new int[]{2, 3, 1, 4}, state.queueLevels());
            assertThrows(
                    IllegalArgumentException.class,
                    () -> new GameSessionStore.SingleState(
                            GameSessionStore.NO_RANDOM_STATE,
                            new int[]{2, 3, 1, 4},
                            2,
                            10,
                            0,
                            11,
                            0,
                            0,
                            0,
                            200f,
                            200f,
                            0f,
                            0f,
                            0f,
                            0f,
                            true,
                            true,
                            false,
                            0,
                            world.snapshot()
                    )
            );
        } finally {
            world.dispose();
        }
    }

    private static GameSessionStore.SingleState singleState(
            FruitPhysicsWorld.Snapshot physics,
            int score,
            boolean resultRecorded) {
        return new GameSessionStore.SingleState(
                0x1234ABCDL,
                new int[]{2, 3, 1, 4},
                2,
                score,
                Math.max(0, score - 10),
                Math.max(0, score - 5),
                Math.max(500, score),
                8,
                1,
                196f,
                196f,
                0.12f,
                0.08f,
                0.3f,
                0.5f,
                true,
                !resultRecorded,
                resultRecorded,
                resultRecorded ? 42 : 0,
                physics
        );
    }

    private static void assertSingleEquals(
            GameSessionStore.SingleState expected,
            GameSessionStore.SingleState actual) {
        assertEquals(
                expected.queueRandomState(),
                actual.queueRandomState()
        );
        assertArrayEquals(expected.queueLevels(), actual.queueLevels());
        assertEquals(expected.currentLevel(), actual.currentLevel());
        assertEquals(expected.score(), actual.score());
        assertEquals(expected.lastScore(), actual.lastScore());
        assertEquals(expected.displayedScore(), actual.displayedScore());
        assertEquals(
                expected.displayedBestScore(),
                actual.displayedBestScore()
        );
        assertEquals(expected.stepCount(), actual.stepCount());
        assertEquals(
                expected.watermelonCount(),
                actual.watermelonCount()
        );
        assertEquals(expected.previewX(), actual.previewX(), 0f);
        assertEquals(
                expected.previewAnchorX(),
                actual.previewAnchorX(),
                0f
        );
        assertEquals(
                expected.dropCooldownSeconds(),
                actual.dropCooldownSeconds(),
                0f
        );
        assertEquals(expected.stableSeconds(), actual.stableSeconds(), 0f);
        assertEquals(expected.dangerSeconds(), actual.dangerSeconds(), 0f);
        assertEquals(
                expected.aiLoadingSeconds(),
                actual.aiLoadingSeconds(),
                0f
        );
        assertEquals(expected.waiting(), actual.waiting());
        assertEquals(expected.alive(), actual.alive());
        assertEquals(expected.resultRecorded(), actual.resultRecorded());
        assertEquals(
                expected.resultPercentile(),
                actual.resultPercentile()
        );
        assertPhysicsEquals(expected.physics(), actual.physics());
    }

    private static void assertDuelSnapshotEquals(
            DuelMatch.Snapshot expected,
            DuelMatch.Snapshot actual) {
        assertEquals(
                expected.queueRandomState(),
                actual.queueRandomState()
        );
        assertArrayEquals(expected.queueLevels(), actual.queueLevels());
        assertEquals(expected.currentLevel(), actual.currentLevel());
        assertEquals(expected.roundIndex(), actual.roundIndex());
        assertEquals(
                expected.roundRemainingSeconds(),
                actual.roundRemainingSeconds(),
                0f
        );
        assertEquals(
                expected.nextRoundRemainingSeconds(),
                actual.nextRoundRemainingSeconds(),
                0f
        );
        assertEquals(expected.roundOpen(), actual.roundOpen());
        assertEquals(expected.outcome(), actual.outcome());
        assertPhysicsEquals(
                expected.player().physics(),
                actual.player().physics()
        );
        assertPhysicsEquals(
                expected.ai().physics(),
                actual.ai().physics()
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

    private static final class MemoryPreferences implements Preferences {
        private final Map<String, Object> values = new HashMap<>();
        private int flushCount;

        @Override
        public Preferences putBoolean(String key, boolean value) {
            values.put(key, value);
            return this;
        }

        @Override
        public Preferences putInteger(String key, int value) {
            values.put(key, value);
            return this;
        }

        @Override
        public Preferences putLong(String key, long value) {
            values.put(key, value);
            return this;
        }

        @Override
        public Preferences putFloat(String key, float value) {
            values.put(key, value);
            return this;
        }

        @Override
        public Preferences putString(String key, String value) {
            values.put(key, value);
            return this;
        }

        @Override
        public Preferences put(Map<String, ?> values) {
            this.values.putAll(values);
            return this;
        }

        @Override
        public boolean getBoolean(String key) {
            return getBoolean(key, false);
        }

        @Override
        public int getInteger(String key) {
            return getInteger(key, 0);
        }

        @Override
        public long getLong(String key) {
            return getLong(key, 0L);
        }

        @Override
        public float getFloat(String key) {
            return getFloat(key, 0f);
        }

        @Override
        public String getString(String key) {
            return getString(key, "");
        }

        @Override
        public boolean getBoolean(String key, boolean defaultValue) {
            Object value = values.get(key);
            return value instanceof Boolean
                    ? (boolean) value
                    : defaultValue;
        }

        @Override
        public int getInteger(String key, int defaultValue) {
            Object value = values.get(key);
            return value instanceof Integer
                    ? (int) value
                    : defaultValue;
        }

        @Override
        public long getLong(String key, long defaultValue) {
            Object value = values.get(key);
            return value instanceof Long ? (long) value : defaultValue;
        }

        @Override
        public float getFloat(String key, float defaultValue) {
            Object value = values.get(key);
            return value instanceof Float
                    ? (float) value
                    : defaultValue;
        }

        @Override
        public String getString(String key, String defaultValue) {
            Object value = values.get(key);
            return value instanceof String
                    ? (String) value
                    : defaultValue;
        }

        @Override
        public Map<String, ?> get() {
            return new HashMap<>(values);
        }

        @Override
        public boolean contains(String key) {
            return values.containsKey(key);
        }

        @Override
        public void clear() {
            values.clear();
        }

        @Override
        public void remove(String key) {
            values.remove(key);
        }

        @Override
        public void flush() {
            flushCount++;
        }
    }
}
