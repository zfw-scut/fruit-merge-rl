package com.fruitmerge.ai.game;

import com.badlogic.gdx.Preferences;
import com.badlogic.gdx.utils.Base64Coder;
import org.junit.Test;

import java.util.HashMap;
import java.util.List;
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
    public void recordsAllThreeModesAsNewestFirstDurableEntries() {
        MemoryPreferences preferences = new MemoryPreferences();
        GameProfileStore store = new GameProfileStore(preferences);

        assertTrue(store.recordSoloGame(
                "solo:list",
                1_200,
                1,
                38,
                1_000L
        ));
        assertTrue(store.recordVersusGame(
                "duel:list",
                2_300,
                2_050,
                2,
                1,
                51,
                GameProfileStore.BattleResult.WIN,
                2_000L
        ));
        assertTrue(store.recordAiDemoGame(
                "demo:list",
                3_400,
                3,
                64,
                3_000L
        ));

        GameProfileStore reloaded = new GameProfileStore(preferences);
        List<GameProfileStore.GameRecord> records =
                reloaded.gameRecords();
        assertEquals(3, records.size());
        assertEquals(
                GameProfileStore.GameMode.AI_DEMO,
                records.get(0).mode()
        );
        assertEquals(3_400, records.get(0).score());
        assertEquals(64, records.get(0).dropCount());
        assertEquals(
                GameProfileStore.GameMode.DUEL,
                records.get(1).mode()
        );
        assertEquals(2_050, records.get(1).opponentScore());
        assertEquals(
                GameProfileStore.RecordResult.WIN,
                records.get(1).result()
        );
        assertEquals(
                GameProfileStore.GameMode.SOLO,
                records.get(2).mode()
        );

        GameProfileStore.History history = reloaded.history();
        assertEquals(3, history.totalGames());
        assertEquals(1, history.soloGames());
        assertEquals(1, history.aiDemoGames());
        assertEquals(1_200, history.highestSoloScore());
        assertEquals(2_300, history.highestVersusScore());
        assertEquals(3_400, history.highestAiDemoScore());
        assertEquals(3_400, history.highScore());
    }

    @Test
    public void recentListIsBoundedWithoutLosingTotalsOrBests() {
        MemoryPreferences preferences = new MemoryPreferences();
        GameProfileStore store = new GameProfileStore(preferences);
        int count = GameProfileStore.MAX_GAME_RECORDS + 7;

        for (int index = 0; index < count; index++) {
            assertTrue(store.recordSoloGame(
                    "solo:bounded:" + index,
                    index * 10,
                    index % 3,
                    index,
                    index + 1L
            ));
        }

        GameProfileStore reloaded = new GameProfileStore(preferences);
        assertEquals(count, reloaded.history().totalGames());
        assertEquals((count - 1) * 10, reloaded.history().highScore());
        assertEquals(
                GameProfileStore.MAX_GAME_RECORDS,
                reloaded.gameRecords().size()
        );
        assertEquals(
                "solo:bounded:" + (count - 1),
                reloaded.gameRecords().get(0).sessionId()
        );
        assertEquals(
                "solo:bounded:7",
                reloaded.gameRecords().get(
                        GameProfileStore.MAX_GAME_RECORDS - 1
                ).sessionId()
        );
        assertFalse(reloaded.hasRecordedSession("solo:bounded:0"));
        assertTrue(reloaded.hasRecordedSession(
                "solo:bounded:" + (count - 1)
        ));
    }

    @Test
    public void damagedRecordTailKeepsEarlierCompleteEntries() {
        MemoryPreferences preferences = new MemoryPreferences();
        GameProfileStore store = new GameProfileStore(preferences);
        assertTrue(store.recordSoloGame(
                "solo:older",
                700,
                0,
                10,
                10L
        ));
        assertTrue(store.recordAiDemoGame(
                "demo:newer",
                1_100,
                1,
                20,
                20L
        ));

        String key = "history.game_records";
        String encoded = preferences.getString(key);
        preferences.putString(key, encoded + "\nnot-valid-base64");
        GameProfileStore restored = new GameProfileStore(preferences);

        assertEquals(2, restored.gameRecords().size());
        assertEquals(
                "demo:newer",
                restored.gameRecords().get(0).sessionId()
        );
        assertEquals(1_100, restored.history().highScore());
    }

    @Test
    public void validBase64RecordWithChangedPayloadFailsChecksum() {
        MemoryPreferences preferences = new MemoryPreferences();
        GameProfileStore store = new GameProfileStore(preferences);
        assertTrue(store.recordSoloGame(
                "solo:checksum",
                1_700,
                1,
                24,
                100L
        ));

        String key = "history.game_records";
        byte[] bytes = Base64Coder.decode(preferences.getString(key));
        bytes[bytes.length / 2] ^= 0x01;
        preferences.putString(
                key,
                new String(Base64Coder.encode(bytes))
        );

        GameProfileStore restored = new GameProfileStore(preferences);
        assertTrue(restored.gameRecords().isEmpty());
    }

    @Test
    public void detailRowsRepairATruncatedIdempotencyLedger() {
        MemoryPreferences preferences = new MemoryPreferences();
        GameProfileStore store = new GameProfileStore(preferences);
        String sessionId = "solo:ledger-repair";
        assertTrue(store.recordSoloGame(sessionId, 2_100, 2));
        preferences.putString(
                "history.recorded_session_ids",
                "18:truncated"
        );

        GameProfileStore restored = new GameProfileStore(preferences);
        assertTrue(restored.hasRecordedSession(sessionId));
        assertFalse(restored.recordSoloGame(sessionId, 9_900, 8));
        assertEquals(1, restored.history().totalGames());
        assertEquals(2_100, restored.history().highScore());
    }

    @Test
    public void duelLossAndDrawDetailsRoundTrip() {
        MemoryPreferences preferences = new MemoryPreferences();
        GameProfileStore store = new GameProfileStore(preferences);
        assertTrue(store.recordVersusGame(
                "duel:loss",
                800,
                1_200,
                0,
                1,
                21,
                GameProfileStore.BattleResult.LOSS,
                100L
        ));
        assertTrue(store.recordVersusGame(
                "duel:draw",
                1_400,
                1_400,
                2,
                2,
                35,
                GameProfileStore.BattleResult.DRAW,
                200L
        ));

        List<GameProfileStore.GameRecord> records =
                new GameProfileStore(preferences).gameRecords();
        assertEquals(2, records.size());
        assertEquals(
                GameProfileStore.RecordResult.DRAW,
                records.get(0).result()
        );
        assertEquals(1_400, records.get(0).opponentScore());
        assertEquals(2, records.get(0).opponentWatermelonsCreated());
        assertEquals(
                GameProfileStore.RecordResult.LOSS,
                records.get(1).result()
        );
        assertEquals(1_200, records.get(1).opponentScore());
    }

    @Test
    public void legacyAggregatesRemainWithoutInventingOldRows() {
        MemoryPreferences preferences = new MemoryPreferences();
        preferences
                .putInteger("history.total_games", 9)
                .putInteger("history.high_score", 4_800)
                .putInteger("history.versus_wins", 2)
                .putInteger("history.versus_losses", 1)
                .putInteger("history.versus_draws", 0)
                .putInteger("history.highest_versus_score", 3_000);

        GameProfileStore restored = new GameProfileStore(preferences);
        assertEquals(9, restored.history().totalGames());
        assertEquals(4_800, restored.history().highScore());
        assertEquals(6, restored.history().soloGames());
        assertEquals(0, restored.history().aiDemoGames());
        assertTrue(restored.gameRecords().isEmpty());
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
        assertTrue(reloaded.gameRecords().isEmpty());
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
