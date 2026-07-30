package com.fruitmerge.ai.game;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Preferences;

import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

/**
 * Persistent user settings and aggregate game history.
 *
 * <p>The store owns only durable data. Runtime screen state, the current round, and physics
 * objects deliberately stay outside this class so the same data contract can be reused by the
 * solo and versus screens.</p>
 */
public final class GameProfileStore {
    public static final String PREFERENCES_NAME = "fruit-merge-ai-profile-v1";

    public static final float MIN_SOUND_VOLUME = 0f;
    public static final float MAX_SOUND_VOLUME = 1f;
    public static final float DEFAULT_SOUND_VOLUME = 0.72f;

    public static final float MIN_GAME_SPEED = 0.5f;
    public static final float MAX_GAME_SPEED = 2f;
    public static final float DEFAULT_GAME_SPEED = 1f;

    public static final float MIN_VERSUS_DROP_SECONDS = 3f;
    public static final float MAX_VERSUS_DROP_SECONDS = 30f;
    public static final float DEFAULT_VERSUS_DROP_SECONDS = 8f;

    public static final float MIN_RESULT_HOLD_SECONDS = 2f;
    public static final float MAX_RESULT_HOLD_SECONDS = 12f;
    public static final float DEFAULT_RESULT_HOLD_SECONDS = 4f;

    private static final int SCHEMA_VERSION = 2;
    private static final int MAX_SESSION_ID_LENGTH = 256;

    private static final String KEY_SCHEMA_VERSION = "schema_version";
    private static final String KEY_SOUND_VOLUME = "settings.sound_volume";
    private static final String KEY_VIBRATE_MERGE = "settings.vibrate_merge";
    private static final String KEY_VIBRATE_DROP = "settings.vibrate_drop";
    private static final String KEY_VIBRATE_SCORE_COLLECT = "settings.vibrate_score_collect";
    private static final String KEY_GAME_SPEED = "settings.game_speed";
    private static final String KEY_VERSUS_DROP_SECONDS = "settings.versus_drop_seconds";
    private static final String KEY_RESULT_HOLD_SECONDS = "settings.result_hold_seconds";

    private static final String KEY_TOTAL_GAMES = "history.total_games";
    private static final String KEY_HIGH_SCORE = "history.high_score";
    private static final String KEY_MAX_WATERMELONS_IN_GAME =
            "history.max_watermelons_in_game";
    private static final String KEY_TOTAL_WATERMELONS = "history.total_watermelons";
    private static final String KEY_VERSUS_WINS = "history.versus_wins";
    private static final String KEY_VERSUS_LOSSES = "history.versus_losses";
    private static final String KEY_VERSUS_DRAWS = "history.versus_draws";
    private static final String KEY_HIGHEST_VERSUS_SCORE = "history.highest_versus_score";
    private static final String KEY_RECORDED_SESSION_IDS =
            "history.recorded_session_ids";

    private final Preferences preferences;

    private float soundVolume;
    private boolean vibrateOnMerge;
    private boolean vibrateOnDrop;
    private boolean vibrateOnScoreCollect;
    private float gameSpeed;
    private float versusDropSeconds;
    private float resultHoldSeconds;

    private int totalGames;
    private int highScore;
    private int maxWatermelonsInGame;
    private int totalWatermelons;
    private int versusWins;
    private int versusLosses;
    private int versusDraws;
    private int highestVersusScore;
    private final Set<String> recordedSessionIds = new LinkedHashSet<>();

    /**
     * Opens the platform preference file. Call this only after libGDX has created the app.
     */
    public static GameProfileStore open() {
        if (Gdx.app == null) {
            throw new IllegalStateException("libGDX application is not ready");
        }
        return new GameProfileStore(Gdx.app.getPreferences(PREFERENCES_NAME));
    }

    /**
     * Injectable constructor used by platform code and unit tests.
     */
    public GameProfileStore(Preferences preferences) {
        this.preferences = Objects.requireNonNull(preferences, "preferences");
        reload();
    }

    public synchronized void reload() {
        soundVolume = sanitizeFinite(
                preferences.getFloat(KEY_SOUND_VOLUME, DEFAULT_SOUND_VOLUME),
                DEFAULT_SOUND_VOLUME,
                MIN_SOUND_VOLUME,
                MAX_SOUND_VOLUME);
        vibrateOnMerge = preferences.getBoolean(KEY_VIBRATE_MERGE, true);
        vibrateOnDrop = preferences.getBoolean(KEY_VIBRATE_DROP, false);
        vibrateOnScoreCollect =
                preferences.getBoolean(KEY_VIBRATE_SCORE_COLLECT, false);
        gameSpeed = sanitizeFinite(
                preferences.getFloat(KEY_GAME_SPEED, DEFAULT_GAME_SPEED),
                DEFAULT_GAME_SPEED,
                MIN_GAME_SPEED,
                MAX_GAME_SPEED);
        versusDropSeconds = sanitizeFinite(
                preferences.getFloat(
                        KEY_VERSUS_DROP_SECONDS, DEFAULT_VERSUS_DROP_SECONDS),
                DEFAULT_VERSUS_DROP_SECONDS,
                MIN_VERSUS_DROP_SECONDS,
                MAX_VERSUS_DROP_SECONDS);
        resultHoldSeconds = sanitizeFinite(
                preferences.getFloat(
                        KEY_RESULT_HOLD_SECONDS, DEFAULT_RESULT_HOLD_SECONDS),
                DEFAULT_RESULT_HOLD_SECONDS,
                MIN_RESULT_HOLD_SECONDS,
                MAX_RESULT_HOLD_SECONDS);

        totalGames = nonNegative(preferences.getInteger(KEY_TOTAL_GAMES, 0));
        highScore = nonNegative(preferences.getInteger(KEY_HIGH_SCORE, 0));
        maxWatermelonsInGame =
                nonNegative(preferences.getInteger(KEY_MAX_WATERMELONS_IN_GAME, 0));
        totalWatermelons =
                nonNegative(preferences.getInteger(KEY_TOTAL_WATERMELONS, 0));
        versusWins = nonNegative(preferences.getInteger(KEY_VERSUS_WINS, 0));
        versusLosses = nonNegative(preferences.getInteger(KEY_VERSUS_LOSSES, 0));
        versusDraws = nonNegative(preferences.getInteger(KEY_VERSUS_DRAWS, 0));
        highestVersusScore =
                nonNegative(preferences.getInteger(KEY_HIGHEST_VERSUS_SCORE, 0));
        recordedSessionIds.clear();
        recordedSessionIds.addAll(
                decodeSessionIds(
                        preferences.getString(KEY_RECORDED_SESSION_IDS, "")));
        repairHistoryInvariants();
    }

    public synchronized Settings settings() {
        return new Settings(
                soundVolume,
                vibrateOnMerge,
                vibrateOnDrop,
                vibrateOnScoreCollect,
                gameSpeed,
                versusDropSeconds,
                resultHoldSeconds);
    }

    public synchronized History history() {
        return new History(
                totalGames,
                highScore,
                maxWatermelonsInGame,
                totalWatermelons,
                versusWins,
                versusLosses,
                versusDraws,
                highestVersusScore);
    }

    public synchronized GameProfileStore setSoundVolume(float value) {
        soundVolume = sanitizeFinite(
                value, DEFAULT_SOUND_VOLUME, MIN_SOUND_VOLUME, MAX_SOUND_VOLUME);
        return this;
    }

    public synchronized GameProfileStore setVibrateOnMerge(boolean value) {
        vibrateOnMerge = value;
        return this;
    }

    public synchronized GameProfileStore setVibrateOnDrop(boolean value) {
        vibrateOnDrop = value;
        return this;
    }

    public synchronized GameProfileStore setVibrateOnScoreCollect(boolean value) {
        vibrateOnScoreCollect = value;
        return this;
    }

    public synchronized GameProfileStore setGameSpeed(float value) {
        gameSpeed = sanitizeFinite(
                value, DEFAULT_GAME_SPEED, MIN_GAME_SPEED, MAX_GAME_SPEED);
        return this;
    }

    public synchronized GameProfileStore setVersusDropSeconds(float value) {
        versusDropSeconds = sanitizeFinite(
                value,
                DEFAULT_VERSUS_DROP_SECONDS,
                MIN_VERSUS_DROP_SECONDS,
                MAX_VERSUS_DROP_SECONDS);
        return this;
    }

    public synchronized GameProfileStore setResultHoldSeconds(float value) {
        resultHoldSeconds = sanitizeFinite(
                value,
                DEFAULT_RESULT_HOLD_SECONDS,
                MIN_RESULT_HOLD_SECONDS,
                MAX_RESULT_HOLD_SECONDS);
        return this;
    }

    /**
     * Persists the current settings and history in one flush.
     */
    public synchronized void save() {
        preferences
                .putInteger(KEY_SCHEMA_VERSION, SCHEMA_VERSION)
                .putFloat(KEY_SOUND_VOLUME, soundVolume)
                .putBoolean(KEY_VIBRATE_MERGE, vibrateOnMerge)
                .putBoolean(KEY_VIBRATE_DROP, vibrateOnDrop)
                .putBoolean(KEY_VIBRATE_SCORE_COLLECT, vibrateOnScoreCollect)
                .putFloat(KEY_GAME_SPEED, gameSpeed)
                .putFloat(KEY_VERSUS_DROP_SECONDS, versusDropSeconds)
                .putFloat(KEY_RESULT_HOLD_SECONDS, resultHoldSeconds)
                .putInteger(KEY_TOTAL_GAMES, totalGames)
                .putInteger(KEY_HIGH_SCORE, highScore)
                .putInteger(KEY_MAX_WATERMELONS_IN_GAME, maxWatermelonsInGame)
                .putInteger(KEY_TOTAL_WATERMELONS, totalWatermelons)
                .putInteger(KEY_VERSUS_WINS, versusWins)
                .putInteger(KEY_VERSUS_LOSSES, versusLosses)
                .putInteger(KEY_VERSUS_DRAWS, versusDraws)
                .putInteger(KEY_HIGHEST_VERSUS_SCORE, highestVersusScore)
                .putString(
                        KEY_RECORDED_SESSION_IDS,
                        encodeSessionIds(recordedSessionIds));
        preferences.flush();
    }

    /**
     * Restores only settings. Historical records remain untouched.
     */
    public synchronized void resetSettings() {
        soundVolume = DEFAULT_SOUND_VOLUME;
        vibrateOnMerge = true;
        vibrateOnDrop = false;
        vibrateOnScoreCollect = false;
        gameSpeed = DEFAULT_GAME_SPEED;
        versusDropSeconds = DEFAULT_VERSUS_DROP_SECONDS;
        resultHoldSeconds = DEFAULT_RESULT_HOLD_SECONDS;
        save();
    }

    /**
     * Clears only aggregate history. Settings remain untouched.
     */
    public synchronized void resetHistory() {
        totalGames = 0;
        highScore = 0;
        maxWatermelonsInGame = 0;
        totalWatermelons = 0;
        versusWins = 0;
        versusLosses = 0;
        versusDraws = 0;
        highestVersusScore = 0;
        // 用户主动清空历史后，旧会话也不应继续占据“已结算”名单。
        recordedSessionIds.clear();
        save();
    }

    public synchronized void recordSoloGame(int score, int watermelonsCreated) {
        recordCommonResult(score, watermelonsCreated);
        save();
    }

    /**
     * 幂等记录一局单人游戏。
     *
     * <p>自动存档可能在“历史已经写入、草稿尚未来得及删除”之间被系统终止。恢复
     * 后结算流程会再次到达这里，因此 sessionId 与历史聚合必须在同一次 Preferences
     * flush 中落盘。返回 {@code true} 表示本次首次计入，{@code false} 表示该会话
     * 已结算，调用方可以安全地继续删除残留草稿而不重复累计。</p>
     */
    public synchronized boolean recordSoloGame(
            String sessionId,
            int score,
            int watermelonsCreated) {
        String safeSessionId = requireSessionId(sessionId);
        if (recordedSessionIds.contains(safeSessionId)) {
            return false;
        }

        recordCommonResult(score, watermelonsCreated);
        recordedSessionIds.add(safeSessionId);
        save();
        return true;
    }

    public synchronized void recordVersusGame(
            int playerScore,
            int watermelonsCreated,
            BattleResult result) {
        BattleResult safeResult = Objects.requireNonNull(result, "result");
        recordVersusResult(playerScore, watermelonsCreated, safeResult);
        save();
    }

    /**
     * 幂等记录一局 AI 对战；同一个 sessionId 在单人和对战入口之间也只允许写一次。
     */
    public synchronized boolean recordVersusGame(
            String sessionId,
            int playerScore,
            int watermelonsCreated,
            BattleResult result) {
        String safeSessionId = requireSessionId(sessionId);
        BattleResult safeResult = Objects.requireNonNull(result, "result");
        if (recordedSessionIds.contains(safeSessionId)) {
            return false;
        }

        recordVersusResult(playerScore, watermelonsCreated, safeResult);
        recordedSessionIds.add(safeSessionId);
        save();
        return true;
    }

    public synchronized boolean recordVersusGame(
            String sessionId,
            int playerScore,
            int watermelonsCreated,
            boolean playerWon) {
        return recordVersusGame(
                sessionId,
                playerScore,
                watermelonsCreated,
                playerWon ? BattleResult.WIN : BattleResult.LOSS);
    }

    /**
     * 查询某局是否已经进入聚合历史，不修改 Preferences。
     */
    public synchronized boolean hasRecordedSession(String sessionId) {
        return recordedSessionIds.contains(requireSessionId(sessionId));
    }

    private void recordVersusResult(
            int playerScore,
            int watermelonsCreated,
            BattleResult result) {
        recordCommonResult(playerScore, watermelonsCreated);
        switch (result) {
            case WIN:
                versusWins = saturatedIncrement(versusWins);
                break;
            case LOSS:
                versusLosses = saturatedIncrement(versusLosses);
                break;
            case DRAW:
                versusDraws = saturatedIncrement(versusDraws);
                break;
        }
        highestVersusScore = Math.max(highestVersusScore, nonNegative(playerScore));
        repairHistoryInvariants();
    }

    public synchronized void recordVersusGame(
            int playerScore,
            int watermelonsCreated,
            boolean playerWon) {
        recordVersusGame(
                playerScore,
                watermelonsCreated,
                playerWon ? BattleResult.WIN : BattleResult.LOSS);
    }

    /**
     * Returns the integer shown on the result screen. The estimation details intentionally stay
     * internal so UI code cannot accidentally expose an explanation or raw thresholds.
     */
    public int resultPercentile(int score) {
        return calculateResultPercentile(nonNegative(score));
    }

    private void recordCommonResult(int score, int watermelonsCreated) {
        int safeScore = nonNegative(score);
        int safeWatermelons = nonNegative(watermelonsCreated);
        totalGames = saturatedIncrement(totalGames);
        highScore = Math.max(highScore, safeScore);
        maxWatermelonsInGame = Math.max(maxWatermelonsInGame, safeWatermelons);
        totalWatermelons = saturatedAdd(totalWatermelons, safeWatermelons);
        repairHistoryInvariants();
    }

    private void repairHistoryInvariants() {
        int versusGames = saturatedAdd(
                saturatedAdd(versusWins, versusLosses), versusDraws);
        totalGames = Math.max(totalGames, versusGames);
        totalWatermelons = Math.max(totalWatermelons, maxWatermelonsInGame);
        highScore = Math.max(highScore, highestVersusScore);
    }

    private static String requireSessionId(String sessionId) {
        if (sessionId == null) {
            throw new IllegalArgumentException("session id must not be null");
        }
        String normalized = sessionId.trim();
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException("session id must not be blank");
        }
        if (normalized.length() > MAX_SESSION_ID_LENGTH) {
            throw new IllegalArgumentException("session id is too long");
        }
        for (int index = 0; index < normalized.length(); index++) {
            if (Character.isISOControl(normalized.charAt(index))) {
                throw new IllegalArgumentException(
                        "session id must not contain control characters");
            }
        }
        return normalized;
    }

    /**
     * 采用长度前缀而不是换行或逗号分隔，sessionId 即便包含分隔符也能无歧义恢复。
     * 格式为重复的 {@code 字符数:原文}；遇到损坏尾部时保留此前完整条目并停止解析。
     */
    private static String encodeSessionIds(Set<String> sessionIds) {
        StringBuilder encoded = new StringBuilder();
        for (String sessionId : sessionIds) {
            encoded.append(sessionId.length())
                    .append(':')
                    .append(sessionId);
        }
        return encoded.toString();
    }

    private static Set<String> decodeSessionIds(String encoded) {
        Set<String> decoded = new LinkedHashSet<>();
        if (encoded == null || encoded.isEmpty()) {
            return decoded;
        }

        int cursor = 0;
        while (cursor < encoded.length()) {
            int colon = encoded.indexOf(':', cursor);
            if (colon <= cursor) {
                break;
            }
            int length;
            try {
                length = Integer.parseInt(encoded.substring(cursor, colon));
            } catch (NumberFormatException ignored) {
                break;
            }
            if (length <= 0 || length > MAX_SESSION_ID_LENGTH) {
                break;
            }
            int valueStart = colon + 1;
            int valueEnd = valueStart + length;
            if (valueEnd < valueStart || valueEnd > encoded.length()) {
                break;
            }
            String sessionId = encoded.substring(valueStart, valueEnd);
            try {
                decoded.add(requireSessionId(sessionId));
            } catch (IllegalArgumentException ignored) {
                // 单条非法即视为存档尾部损坏，避免后续字符被错位解释为新长度。
                break;
            }
            cursor = valueEnd;
        }
        return decoded;
    }

    private static int calculateResultPercentile(int score) {
        int[] scoreAnchors = {0, 1_000, 2_000, 5_000, 10_000};
        float[] percentileAnchors = {4f, 50f, 70f, 90f, 99f};

        if (score >= scoreAnchors[scoreAnchors.length - 1]) {
            return 99;
        }

        for (int index = 1; index < scoreAnchors.length; index++) {
            if (score <= scoreAnchors[index]) {
                float range = scoreAnchors[index] - scoreAnchors[index - 1];
                float progress = (score - scoreAnchors[index - 1]) / range;
                float smoothProgress = progress * progress * (3f - 2f * progress);
                float estimate = percentileAnchors[index - 1]
                        + (percentileAnchors[index] - percentileAnchors[index - 1])
                        * smoothProgress;
                return clamp(Math.round(estimate), 1, 99);
            }
        }
        return 99;
    }

    private static float sanitizeFinite(
            float value,
            float fallback,
            float minimum,
            float maximum) {
        if (!Float.isFinite(value)) {
            return fallback;
        }
        return clamp(value, minimum, maximum);
    }

    private static int nonNegative(int value) {
        return Math.max(0, value);
    }

    private static int saturatedIncrement(int value) {
        return value == Integer.MAX_VALUE ? value : value + 1;
    }

    private static int saturatedAdd(int first, int second) {
        long result = (long) first + second;
        return result >= Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) result;
    }

    private static float clamp(float value, float minimum, float maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private static int clamp(int value, int minimum, int maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    public enum BattleResult {
        WIN,
        LOSS,
        DRAW
    }

    public static final class Settings {
        private final float soundVolume;
        private final boolean vibrateOnMerge;
        private final boolean vibrateOnDrop;
        private final boolean vibrateOnScoreCollect;
        private final float gameSpeed;
        private final float versusDropSeconds;
        private final float resultHoldSeconds;

        private Settings(
                float soundVolume,
                boolean vibrateOnMerge,
                boolean vibrateOnDrop,
                boolean vibrateOnScoreCollect,
                float gameSpeed,
                float versusDropSeconds,
                float resultHoldSeconds) {
            this.soundVolume = soundVolume;
            this.vibrateOnMerge = vibrateOnMerge;
            this.vibrateOnDrop = vibrateOnDrop;
            this.vibrateOnScoreCollect = vibrateOnScoreCollect;
            this.gameSpeed = gameSpeed;
            this.versusDropSeconds = versusDropSeconds;
            this.resultHoldSeconds = resultHoldSeconds;
        }

        public float soundVolume() {
            return soundVolume;
        }

        public boolean vibrateOnMerge() {
            return vibrateOnMerge;
        }

        public boolean vibrateOnDrop() {
            return vibrateOnDrop;
        }

        public boolean vibrateOnScoreCollect() {
            return vibrateOnScoreCollect;
        }

        public float gameSpeed() {
            return gameSpeed;
        }

        public float versusDropSeconds() {
            return versusDropSeconds;
        }

        public float resultHoldSeconds() {
            return resultHoldSeconds;
        }
    }

    public static final class History {
        private final int totalGames;
        private final int highScore;
        private final int maxWatermelonsInGame;
        private final int totalWatermelons;
        private final int versusWins;
        private final int versusLosses;
        private final int versusDraws;
        private final int highestVersusScore;

        private History(
                int totalGames,
                int highScore,
                int maxWatermelonsInGame,
                int totalWatermelons,
                int versusWins,
                int versusLosses,
                int versusDraws,
                int highestVersusScore) {
            this.totalGames = totalGames;
            this.highScore = highScore;
            this.maxWatermelonsInGame = maxWatermelonsInGame;
            this.totalWatermelons = totalWatermelons;
            this.versusWins = versusWins;
            this.versusLosses = versusLosses;
            this.versusDraws = versusDraws;
            this.highestVersusScore = highestVersusScore;
        }

        public int totalGames() {
            return totalGames;
        }

        public int highScore() {
            return highScore;
        }

        public int maxWatermelonsInGame() {
            return maxWatermelonsInGame;
        }

        public int totalWatermelons() {
            return totalWatermelons;
        }

        public int versusWins() {
            return versusWins;
        }

        public int versusLosses() {
            return versusLosses;
        }

        public int versusDraws() {
            return versusDraws;
        }

        public int highestVersusScore() {
            return highestVersusScore;
        }
    }
}
