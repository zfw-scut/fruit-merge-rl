package com.fruitmerge.ai.game;

import com.badlogic.gdx.Preferences;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public final class GameProfileStoreTest {
    @Test
    public void defaultsEnableOnlyMergeVibration() {
        GameProfileStore store = new GameProfileStore(new MemoryPreferences());
        GameProfileStore.Settings settings = store.settings();

        assertEquals(GameProfileStore.DEFAULT_SOUND_VOLUME, settings.soundVolume(), 0.0001f);
        assertTrue(settings.vibrateOnMerge());
        assertFalse(settings.vibrateOnDrop());
        assertFalse(settings.vibrateOnScoreCollect());
        assertEquals(GameProfileStore.DEFAULT_GAME_SPEED, settings.gameSpeed(), 0.0001f);
    }

    @Test
    public void settingsAreClampedAndPersisted() {
        MemoryPreferences preferences = new MemoryPreferences();
        GameProfileStore store = new GameProfileStore(preferences);
        store.setSoundVolume(5f)
                .setGameSpeed(Float.NaN)
                .setVersusDropSeconds(-10f)
                .setResultHoldSeconds(100f)
                .setVibrateOnMerge(false)
                .setVibrateOnDrop(true)
                .setVibrateOnScoreCollect(true);
        store.save();

        GameProfileStore.Settings reloaded =
                new GameProfileStore(preferences).settings();
        assertEquals(GameProfileStore.MAX_SOUND_VOLUME, reloaded.soundVolume(), 0.0001f);
        assertEquals(GameProfileStore.DEFAULT_GAME_SPEED, reloaded.gameSpeed(), 0.0001f);
        assertEquals(
                GameProfileStore.MIN_VERSUS_DROP_SECONDS,
                reloaded.versusDropSeconds(),
                0.0001f);
        assertEquals(
                GameProfileStore.MAX_RESULT_HOLD_SECONDS,
                reloaded.resultHoldSeconds(),
                0.0001f);
        assertFalse(reloaded.vibrateOnMerge());
        assertTrue(reloaded.vibrateOnDrop());
        assertTrue(reloaded.vibrateOnScoreCollect());
        assertEquals(1, preferences.flushCount);
    }

    @Test
    public void recordsSoloAndVersusHistoryWithoutOverflow() {
        MemoryPreferences preferences = new MemoryPreferences();
        GameProfileStore store = new GameProfileStore(preferences);

        store.recordSoloGame(1_200, 1);
        store.recordVersusGame(2_300, 2, GameProfileStore.BattleResult.WIN);
        store.recordVersusGame(900, 0, GameProfileStore.BattleResult.LOSS);
        store.recordVersusGame(1_500, 1, GameProfileStore.BattleResult.DRAW);

        GameProfileStore.History history =
                new GameProfileStore(preferences).history();
        assertEquals(4, history.totalGames());
        assertEquals(2_300, history.highScore());
        assertEquals(2, history.maxWatermelonsInGame());
        assertEquals(4, history.totalWatermelons());
        assertEquals(1, history.versusWins());
        assertEquals(1, history.versusLosses());
        assertEquals(1, history.versusDraws());
        assertEquals(2_300, history.highestVersusScore());
    }

    @Test
    public void sessionAwareSoloResultIsIdempotentAcrossReload() {
        MemoryPreferences preferences = new MemoryPreferences();
        GameProfileStore store = new GameProfileStore(preferences);
        String sessionId = "solo:2026-07-30:alpha";

        assertTrue(store.recordSoloGame(sessionId, 1_250, 2));
        assertFalse(store.recordSoloGame(sessionId, 9_999, 8));
        assertEquals(1, preferences.flushCount);

        GameProfileStore restored = new GameProfileStore(preferences);
        assertTrue(restored.hasRecordedSession(sessionId));
        assertFalse(restored.recordSoloGame(sessionId, 7_500, 6));
        assertEquals(1, preferences.flushCount);

        GameProfileStore.History history = restored.history();
        assertEquals(1, history.totalGames());
        assertEquals(1_250, history.highScore());
        assertEquals(2, history.maxWatermelonsInGame());
        assertEquals(2, history.totalWatermelons());
    }

    @Test
    public void sessionIdIsGlobalAcrossSoloAndVersusResults() {
        MemoryPreferences preferences = new MemoryPreferences();
        GameProfileStore store = new GameProfileStore(preferences);
        String firstSession = "duel:shared-session";

        assertTrue(
                store.recordVersusGame(
                        firstSession,
                        2_400,
                        1,
                        GameProfileStore.BattleResult.WIN));

        GameProfileStore restored = new GameProfileStore(preferences);
        assertFalse(
                restored.recordVersusGame(
                        firstSession,
                        100,
                        0,
                        GameProfileStore.BattleResult.LOSS));
        assertFalse(restored.recordSoloGame(firstSession, 8_000, 9));
        assertTrue(
                restored.recordVersusGame(
                        "duel:second-session",
                        1_500,
                        2,
                        GameProfileStore.BattleResult.DRAW));

        GameProfileStore.History history =
                new GameProfileStore(preferences).history();
        assertEquals(2, history.totalGames());
        assertEquals(1, history.versusWins());
        assertEquals(0, history.versusLosses());
        assertEquals(1, history.versusDraws());
        assertEquals(2_400, history.highestVersusScore());
        assertEquals(3, history.totalWatermelons());
    }

    @Test
    public void resettingHistoryAlsoClearsIdempotencyLedger() {
        MemoryPreferences preferences = new MemoryPreferences();
        GameProfileStore store = new GameProfileStore(preferences);
        String sessionId = "solo:resettable";

        assertTrue(store.recordSoloGame(sessionId, 4_000, 3));
        store.resetHistory();
        assertFalse(store.hasRecordedSession(sessionId));
        assertTrue(store.recordSoloGame(sessionId, 900, 1));

        GameProfileStore.History history =
                new GameProfileStore(preferences).history();
        assertEquals(1, history.totalGames());
        assertEquals(900, history.highScore());
        assertEquals(1, history.totalWatermelons());
    }

    @Test
    public void invalidSessionIdsCannotMutateHistory() {
        GameProfileStore store = new GameProfileStore(new MemoryPreferences());

        assertThrows(
                IllegalArgumentException.class,
                () -> store.recordSoloGame(null, 1_000, 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> store.recordSoloGame("   ", 1_000, 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> store.recordSoloGame("bad\nsession", 1_000, 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> store.recordVersusGame(
                        "x".repeat(257),
                        1_000,
                        1,
                        GameProfileStore.BattleResult.WIN));
        assertEquals(0, store.history().totalGames());
    }

    @Test
    public void resettingHistoryKeepsSettings() {
        MemoryPreferences preferences = new MemoryPreferences();
        GameProfileStore store = new GameProfileStore(preferences);
        store.setSoundVolume(0.25f).setGameSpeed(1.5f);
        store.recordSoloGame(4_000, 3);
        store.resetHistory();

        GameProfileStore reloaded = new GameProfileStore(preferences);
        assertEquals(0, reloaded.history().totalGames());
        assertEquals(0, reloaded.history().highScore());
        assertEquals(0.25f, reloaded.settings().soundVolume(), 0.0001f);
        assertEquals(1.5f, reloaded.settings().gameSpeed(), 0.0001f);
    }

    @Test
    public void resultPercentileIsMonotonicAndMatchesProductAnchors() {
        GameProfileStore store = new GameProfileStore(new MemoryPreferences());

        assertEquals(50, store.resultPercentile(1_000));
        assertEquals(70, store.resultPercentile(2_000));
        assertEquals(90, store.resultPercentile(5_000));
        assertEquals(99, store.resultPercentile(10_000));
        assertEquals(99, store.resultPercentile(Integer.MAX_VALUE));

        int previous = 0;
        for (int score = 0; score <= 12_000; score += 25) {
            int current = store.resultPercentile(score);
            assertTrue(current >= previous);
            assertTrue(current >= 1 && current <= 99);
            previous = current;
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
            return value instanceof Boolean ? (boolean) value : defaultValue;
        }

        @Override
        public int getInteger(String key, int defaultValue) {
            Object value = values.get(key);
            return value instanceof Integer ? (int) value : defaultValue;
        }

        @Override
        public long getLong(String key, long defaultValue) {
            Object value = values.get(key);
            return value instanceof Long ? (long) value : defaultValue;
        }

        @Override
        public float getFloat(String key, float defaultValue) {
            Object value = values.get(key);
            return value instanceof Float ? (float) value : defaultValue;
        }

        @Override
        public String getString(String key, String defaultValue) {
            Object value = values.get(key);
            return value instanceof String ? (String) value : defaultValue;
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
