package com.fruitmerge.ai.game;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Preferences;
import com.badlogic.gdx.utils.Base64Coder;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.zip.CRC32;

/**
 * Persistent user settings, aggregate bests, and recent completed games.
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

    private static final int SCHEMA_VERSION = 3;
    public static final int MAX_GAME_RECORDS = 200;
    public static final int MAX_RECORDED_SESSION_IDS = MAX_GAME_RECORDS;
    private static final int MAX_SESSION_ID_LENGTH = 256;
    private static final int LEGACY_RECORD_BINARY_VERSION = 1;
    private static final int RECORD_BINARY_VERSION = 2;
    private static final int MAX_RECORD_BYTES = 2_048;

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
    private static final String KEY_HIGHEST_SOLO_SCORE =
            "history.highest_solo_score";
    private static final String KEY_HIGHEST_AI_DEMO_SCORE =
            "history.highest_ai_demo_score";
    private static final String KEY_SOLO_GAMES = "history.solo_games";
    private static final String KEY_AI_DEMO_GAMES = "history.ai_demo_games";
    private static final String KEY_MAX_WATERMELONS_IN_GAME =
            "history.max_watermelons_in_game";
    private static final String KEY_TOTAL_WATERMELONS = "history.total_watermelons";
    private static final String KEY_VERSUS_WINS = "history.versus_wins";
    private static final String KEY_VERSUS_LOSSES = "history.versus_losses";
    private static final String KEY_VERSUS_DRAWS = "history.versus_draws";
    private static final String KEY_HIGHEST_VERSUS_SCORE = "history.highest_versus_score";
    private static final String KEY_RECORDED_SESSION_IDS =
            "history.recorded_session_ids";
    private static final String KEY_GAME_RECORDS = "history.game_records";

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
    private int highestSoloScore;
    private int highestAiDemoScore;
    private int soloGames;
    private int aiDemoGames;
    private int maxWatermelonsInGame;
    private int totalWatermelons;
    private int versusWins;
    private int versusLosses;
    private int versusDraws;
    private int highestVersusScore;
    private final Set<String> recordedSessionIds = new LinkedHashSet<>();
    private final List<GameRecord> gameRecords = new ArrayList<>();

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
        highestSoloScore = nonNegative(
                preferences.getInteger(KEY_HIGHEST_SOLO_SCORE, 0));
        highestAiDemoScore = nonNegative(
                preferences.getInteger(KEY_HIGHEST_AI_DEMO_SCORE, 0));
        soloGames = nonNegative(
                preferences.getInteger(KEY_SOLO_GAMES, 0));
        aiDemoGames = nonNegative(
                preferences.getInteger(KEY_AI_DEMO_GAMES, 0));
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
        gameRecords.clear();
        gameRecords.addAll(
                decodeGameRecords(
                        preferences.getString(KEY_GAME_RECORDS, "")));
        /*
         * A valid detail row is itself durable proof that the session was recorded. Refresh the
         * bounded ledger from oldest to newest so a truncated ledger cannot make a surviving
         * terminal save count the same result twice.
         */
        for (int index = gameRecords.size() - 1; index >= 0; index--) {
            String sessionId = gameRecords.get(index).sessionId();
            recordedSessionIds.remove(sessionId);
            recordedSessionIds.add(sessionId);
        }
        trimRecordedSessionIds();
        /*
         * v1/v2 只有 overall/versus 聚合，没有足够信息把旧最高分可靠归属于单人或
         * AI 演示。保留 overall 原值但不伪造逐局记录；可确定的非对战旧局计入单人
         * 局数，新的分类最高分从 v3 结算开始精确累计。
         */
        int versusGames = saturatedAdd(
                saturatedAdd(versusWins, versusLosses),
                versusDraws
        );
        if (!preferences.contains(KEY_SOLO_GAMES)
                && !preferences.contains(KEY_AI_DEMO_GAMES)) {
            soloGames = Math.max(0, totalGames - versusGames);
            aiDemoGames = 0;
        }
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
                highestSoloScore,
                highestAiDemoScore,
                soloGames,
                aiDemoGames,
                maxWatermelonsInGame,
                totalWatermelons,
                versusWins,
                versusLosses,
                versusDraws,
                highestVersusScore);
    }

    /**
     * Returns newest-first immutable recent results.
     */
    public synchronized List<GameRecord> gameRecords() {
        return Collections.unmodifiableList(
                new ArrayList<>(gameRecords));
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
                .putInteger(KEY_HIGHEST_SOLO_SCORE, highestSoloScore)
                .putInteger(
                        KEY_HIGHEST_AI_DEMO_SCORE,
                        highestAiDemoScore)
                .putInteger(KEY_SOLO_GAMES, soloGames)
                .putInteger(KEY_AI_DEMO_GAMES, aiDemoGames)
                .putInteger(KEY_MAX_WATERMELONS_IN_GAME, maxWatermelonsInGame)
                .putInteger(KEY_TOTAL_WATERMELONS, totalWatermelons)
                .putInteger(KEY_VERSUS_WINS, versusWins)
                .putInteger(KEY_VERSUS_LOSSES, versusLosses)
                .putInteger(KEY_VERSUS_DRAWS, versusDraws)
                .putInteger(KEY_HIGHEST_VERSUS_SCORE, highestVersusScore)
                .putString(
                        KEY_RECORDED_SESSION_IDS,
                        encodeSessionIds(recordedSessionIds))
                .putString(
                        KEY_GAME_RECORDS,
                        encodeGameRecords(gameRecords));
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
        highestSoloScore = 0;
        highestAiDemoScore = 0;
        soloGames = 0;
        aiDemoGames = 0;
        maxWatermelonsInGame = 0;
        totalWatermelons = 0;
        versusWins = 0;
        versusLosses = 0;
        versusDraws = 0;
        highestVersusScore = 0;
        gameRecords.clear();
        // 用户主动清空历史后，旧会话也不应继续占据“已结算”名单。
        recordedSessionIds.clear();
        save();
    }

    public synchronized void recordSoloGame(int score, int watermelonsCreated) {
        recordGame(
                new GameRecord(
                        anonymousSessionId(GameMode.SOLO),
                        GameMode.SOLO,
                        System.currentTimeMillis(),
                        nonNegative(score),
                        0,
                        nonNegative(watermelonsCreated),
                        0,
                        0,
                        RecordResult.COMPLETED
                ),
                false
        );
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
        return recordSoloGame(
                sessionId,
                score,
                watermelonsCreated,
                0
        );
    }

    public synchronized boolean recordSoloGame(
            String sessionId,
            int score,
            int watermelonsCreated,
            int dropCount) {
        return recordSoloGame(
                sessionId,
                score,
                watermelonsCreated,
                dropCount,
                System.currentTimeMillis()
        );
    }

    synchronized boolean recordSoloGame(
            String sessionId,
            int score,
            int watermelonsCreated,
            int dropCount,
            long completedAtEpochMillis) {
        String safeSessionId = requireSessionId(sessionId);
        return recordGame(
                new GameRecord(
                        safeSessionId,
                        GameMode.SOLO,
                        completedAtEpochMillis,
                        nonNegative(score),
                        0,
                        nonNegative(watermelonsCreated),
                        0,
                        nonNegative(dropCount),
                        RecordResult.COMPLETED
                ),
                true
        );
    }

    public synchronized boolean recordAiDemoGame(
            String sessionId,
            int score,
            int watermelonsCreated,
            int dropCount) {
        return recordAiDemoGame(
                sessionId,
                score,
                watermelonsCreated,
                dropCount,
                System.currentTimeMillis()
        );
    }

    synchronized boolean recordAiDemoGame(
            String sessionId,
            int score,
            int watermelonsCreated,
            int dropCount,
            long completedAtEpochMillis) {
        return recordGame(
                new GameRecord(
                        requireSessionId(sessionId),
                        GameMode.AI_DEMO,
                        completedAtEpochMillis,
                        nonNegative(score),
                        0,
                        nonNegative(watermelonsCreated),
                        0,
                        nonNegative(dropCount),
                        RecordResult.COMPLETED
                ),
                true
        );
    }

    public synchronized void recordVersusGame(
            int playerScore,
            int watermelonsCreated,
            BattleResult result) {
        recordGame(
                new GameRecord(
                        anonymousSessionId(GameMode.DUEL),
                        GameMode.DUEL,
                        System.currentTimeMillis(),
                        nonNegative(playerScore),
                        0,
                        nonNegative(watermelonsCreated),
                        0,
                        0,
                        toRecordResult(
                                Objects.requireNonNull(result, "result"))
                ),
                false
        );
    }

    /**
     * 幂等记录一局 AI 对战；同一个 sessionId 在单人和对战入口之间也只允许写一次。
     */
    public synchronized boolean recordVersusGame(
            String sessionId,
            int playerScore,
            int watermelonsCreated,
            BattleResult result) {
        return recordVersusGame(
                sessionId,
                playerScore,
                0,
                watermelonsCreated,
                0,
                0,
                result
        );
    }

    public synchronized boolean recordVersusGame(
            String sessionId,
            int playerScore,
            int aiScore,
            int playerWatermelons,
            int aiWatermelons,
            int playerDropCount,
            BattleResult result) {
        return recordVersusGame(
                sessionId,
                playerScore,
                aiScore,
                playerWatermelons,
                aiWatermelons,
                playerDropCount,
                result,
                System.currentTimeMillis()
        );
    }

    synchronized boolean recordVersusGame(
            String sessionId,
            int playerScore,
            int aiScore,
            int playerWatermelons,
            int aiWatermelons,
            int playerDropCount,
            BattleResult result,
            long completedAtEpochMillis) {
        String safeSessionId = requireSessionId(sessionId);
        BattleResult safeResult = Objects.requireNonNull(result, "result");
        return recordGame(
                new GameRecord(
                        safeSessionId,
                        GameMode.DUEL,
                        completedAtEpochMillis,
                        nonNegative(playerScore),
                        nonNegative(aiScore),
                        nonNegative(playerWatermelons),
                        nonNegative(aiWatermelons),
                        nonNegative(playerDropCount),
                        toRecordResult(safeResult)
                ),
                true
        );
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

    private boolean recordGame(
            GameRecord record,
            boolean enforceIdempotency) {
        if (enforceIdempotency
                && recordedSessionIds.contains(record.sessionId())) {
            return false;
        }

        recordCommonResult(
                record.mode(),
                record.score(),
                record.watermelonsCreated()
        );
        if (record.mode() == GameMode.DUEL) {
            recordVersusResult(
                    record.score(),
                    toBattleResult(record.result())
            );
        }
        gameRecords.add(0, record);
        while (gameRecords.size() > MAX_GAME_RECORDS) {
            gameRecords.remove(gameRecords.size() - 1);
        }
        if (enforceIdempotency) {
            recordedSessionIds.remove(record.sessionId());
            recordedSessionIds.add(record.sessionId());
            trimRecordedSessionIds();
        }
        save();
        return true;
    }

    private void recordVersusResult(
            int playerScore,
            BattleResult result) {
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
        recordCommonResult(
                GameMode.SOLO,
                score,
                watermelonsCreated
        );
    }

    private void recordCommonResult(
            GameMode mode,
            int score,
            int watermelonsCreated) {
        int safeScore = nonNegative(score);
        int safeWatermelons = nonNegative(watermelonsCreated);
        totalGames = saturatedIncrement(totalGames);
        highScore = Math.max(highScore, safeScore);
        if (mode == GameMode.SOLO) {
            soloGames = saturatedIncrement(soloGames);
            highestSoloScore = Math.max(highestSoloScore, safeScore);
        } else if (mode == GameMode.AI_DEMO) {
            aiDemoGames = saturatedIncrement(aiDemoGames);
            highestAiDemoScore = Math.max(
                    highestAiDemoScore,
                    safeScore
            );
        }
        maxWatermelonsInGame = Math.max(maxWatermelonsInGame, safeWatermelons);
        totalWatermelons = saturatedAdd(totalWatermelons, safeWatermelons);
        repairHistoryInvariants();
    }

    private void repairHistoryInvariants() {
        int versusGames = saturatedAdd(
                saturatedAdd(versusWins, versusLosses), versusDraws);
        totalGames = Math.max(totalGames, versusGames);
        totalGames = Math.max(
                totalGames,
                saturatedAdd(
                        saturatedAdd(soloGames, aiDemoGames),
                        versusGames
                )
        );
        totalWatermelons = Math.max(totalWatermelons, maxWatermelonsInGame);
        highScore = Math.max(
                highScore,
                Math.max(
                        highestVersusScore,
                        Math.max(highestSoloScore, highestAiDemoScore)
                )
        );
    }

    private String anonymousSessionId(GameMode mode) {
        return "anonymous:"
                + mode.name().toLowerCase(java.util.Locale.ROOT)
                + ":"
                + System.currentTimeMillis()
                + ":"
                + totalGames;
    }

    private void trimRecordedSessionIds() {
        while (recordedSessionIds.size() > MAX_RECORDED_SESSION_IDS) {
            java.util.Iterator<String> iterator =
                    recordedSessionIds.iterator();
            if (!iterator.hasNext()) {
                return;
            }
            iterator.next();
            iterator.remove();
        }
    }

    private static RecordResult toRecordResult(BattleResult result) {
        switch (result) {
            case WIN:
                return RecordResult.WIN;
            case LOSS:
                return RecordResult.LOSS;
            case DRAW:
                return RecordResult.DRAW;
            default:
                throw new IllegalStateException("unknown battle result");
        }
    }

    private static BattleResult toBattleResult(RecordResult result) {
        switch (result) {
            case WIN:
                return BattleResult.WIN;
            case LOSS:
                return BattleResult.LOSS;
            case DRAW:
                return BattleResult.DRAW;
            default:
                throw new IllegalArgumentException(
                        "duel record requires win/loss/draw");
        }
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

    /**
     * Stores every result as an independently decodable Base64 line. If the Preferences value is
     * truncated, complete newer records before the damaged line remain readable.
     */
    private static String encodeGameRecords(List<GameRecord> records) {
        StringBuilder encoded = new StringBuilder(records.size() * 120);
        for (GameRecord record : records) {
            if (encoded.length() > 0) {
                encoded.append('\n');
            }
            encoded.append(new String(Base64Coder.encode(
                    encodeGameRecord(record)
            )));
        }
        return encoded.toString();
    }

    private static byte[] encodeGameRecord(GameRecord record) {
        try {
            ByteArrayOutputStream payloadBuffer =
                    new ByteArrayOutputStream(128);
            try (DataOutputStream output =
                         new DataOutputStream(payloadBuffer)) {
                output.writeInt(0x46524731);
                output.writeInt(RECORD_BINARY_VERSION);
                output.writeUTF(record.sessionId());
                output.writeByte(record.mode().ordinal());
                output.writeLong(record.completedAtEpochMillis());
                output.writeInt(record.score());
                output.writeInt(record.opponentScore());
                output.writeInt(record.watermelonsCreated());
                output.writeInt(record.opponentWatermelonsCreated());
                output.writeInt(record.dropCount());
                output.writeByte(record.result().ordinal());
            }
            byte[] payload = payloadBuffer.toByteArray();
            CRC32 checksum = new CRC32();
            checksum.update(payload);
            ByteArrayOutputStream framedBuffer =
                    new ByteArrayOutputStream(payload.length + Integer.BYTES);
            framedBuffer.write(payload, 0, payload.length);
            try (DataOutputStream output =
                         new DataOutputStream(framedBuffer)) {
                output.writeInt((int) checksum.getValue());
            }
            byte[] bytes = framedBuffer.toByteArray();
            if (bytes.length > MAX_RECORD_BYTES) {
                throw new IllegalStateException("game record is too large");
            }
            return bytes;
        } catch (IOException error) {
            throw new IllegalStateException(
                    "unable to encode game record",
                    error
            );
        }
    }

    private static List<GameRecord> decodeGameRecords(String encoded) {
        List<GameRecord> decoded = new ArrayList<>();
        if (encoded == null || encoded.isEmpty()) {
            return decoded;
        }
        Set<String> seenSessions = new LinkedHashSet<>();
        String[] lines = encoded
                .replace("\r\n", "\n")
                .replace('\r', '\n')
                .split("\n", -1);
        for (String line : lines) {
            if (line.isEmpty()) {
                break;
            }
            try {
                byte[] bytes = Base64Coder.decode(line);
                if (bytes.length <= 0 || bytes.length > MAX_RECORD_BYTES) {
                    break;
                }
                GameRecord record = decodeGameRecord(bytes);
                if (!seenSessions.add(record.sessionId())) {
                    continue;
                }
                decoded.add(record);
                if (decoded.size() >= MAX_GAME_RECORDS) {
                    break;
                }
            } catch (RuntimeException error) {
                break;
            }
        }
        return decoded;
    }

    private static GameRecord decodeGameRecord(byte[] bytes) {
        try {
            int payloadLength = bytes.length;
            int recordVersion;
            try (DataInputStream header = new DataInputStream(
                    new ByteArrayInputStream(bytes))) {
                if (header.readInt() != 0x46524731) {
                    throw new IllegalArgumentException(
                            "unsupported game record format");
                }
                recordVersion = header.readInt();
            }
            if (recordVersion == RECORD_BINARY_VERSION) {
                if (bytes.length <= Integer.BYTES * 3) {
                    throw new IllegalArgumentException(
                            "game record checksum is missing");
                }
                payloadLength = bytes.length - Integer.BYTES;
                int storedChecksum;
                try (DataInputStream checksumInput =
                             new DataInputStream(
                                     new ByteArrayInputStream(
                                             bytes,
                                             payloadLength,
                                             Integer.BYTES))) {
                    storedChecksum = checksumInput.readInt();
                }
                CRC32 checksum = new CRC32();
                checksum.update(bytes, 0, payloadLength);
                if ((int) checksum.getValue() != storedChecksum) {
                    throw new IllegalArgumentException(
                            "game record checksum does not match");
                }
            } else if (recordVersion != LEGACY_RECORD_BINARY_VERSION) {
                throw new IllegalArgumentException(
                        "unsupported game record format");
            }
            try (DataInputStream input = new DataInputStream(
                    new ByteArrayInputStream(
                            bytes,
                            0,
                            payloadLength))) {
                if (input.readInt() != 0x46524731
                        || input.readInt() != recordVersion) {
                    throw new IllegalArgumentException(
                            "game record header changed while decoding");
                }
                String sessionId = requireSessionId(input.readUTF());
                GameMode mode = enumValue(
                        GameMode.values(),
                        input.readUnsignedByte(),
                        "game mode"
                );
                long completedAt = input.readLong();
                int score = requireNonNegative(input.readInt(), "score");
                int opponentScore = requireNonNegative(
                        input.readInt(),
                        "opponent score"
                );
                int watermelons = requireNonNegative(
                        input.readInt(),
                        "watermelons"
                );
                int opponentWatermelons = requireNonNegative(
                        input.readInt(),
                        "opponent watermelons"
                );
                int dropCount = requireNonNegative(
                        input.readInt(),
                        "drop count"
                );
                RecordResult result = enumValue(
                        RecordResult.values(),
                        input.readUnsignedByte(),
                        "record result"
                );
                if (input.available() != 0) {
                    throw new IllegalArgumentException(
                            "game record contains trailing bytes");
                }
                return new GameRecord(
                        sessionId,
                        mode,
                        completedAt,
                        score,
                        opponentScore,
                        watermelons,
                        opponentWatermelons,
                        dropCount,
                        result
                );
            }
        } catch (IOException error) {
            throw new IllegalArgumentException(
                    "unable to decode game record",
                    error
            );
        }
    }

    private static int requireNonNegative(int value, String field) {
        if (value < 0) {
            throw new IllegalArgumentException(
                    field + " must be non-negative");
        }
        return value;
    }

    private static <T> T enumValue(
            T[] values,
            int ordinal,
            String field) {
        if (ordinal < 0 || ordinal >= values.length) {
            throw new IllegalArgumentException(
                    field + " ordinal is invalid");
        }
        return values[ordinal];
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

    public enum GameMode {
        SOLO,
        DUEL,
        AI_DEMO
    }

    public enum RecordResult {
        COMPLETED,
        WIN,
        LOSS,
        DRAW
    }

    public static final class GameRecord {
        private final String sessionId;
        private final GameMode mode;
        private final long completedAtEpochMillis;
        private final int score;
        private final int opponentScore;
        private final int watermelonsCreated;
        private final int opponentWatermelonsCreated;
        private final int dropCount;
        private final RecordResult result;

        private GameRecord(
                String sessionId,
                GameMode mode,
                long completedAtEpochMillis,
                int score,
                int opponentScore,
                int watermelonsCreated,
                int opponentWatermelonsCreated,
                int dropCount,
                RecordResult result) {
            this.sessionId = requireSessionId(sessionId);
            this.mode = Objects.requireNonNull(mode, "mode");
            this.completedAtEpochMillis = Math.max(
                    0L,
                    completedAtEpochMillis
            );
            this.score = requireNonNegative(score, "score");
            this.opponentScore = requireNonNegative(
                    opponentScore,
                    "opponent score"
            );
            this.watermelonsCreated = requireNonNegative(
                    watermelonsCreated,
                    "watermelons"
            );
            this.opponentWatermelonsCreated = requireNonNegative(
                    opponentWatermelonsCreated,
                    "opponent watermelons"
            );
            this.dropCount = requireNonNegative(
                    dropCount,
                    "drop count"
            );
            this.result = Objects.requireNonNull(result, "result");
            if (mode == GameMode.DUEL
                    && result == RecordResult.COMPLETED) {
                throw new IllegalArgumentException(
                        "duel record requires win/loss/draw");
            }
            if (mode != GameMode.DUEL
                    && result != RecordResult.COMPLETED) {
                throw new IllegalArgumentException(
                        "single-board record must be completed");
            }
        }

        public String sessionId() {
            return sessionId;
        }

        public GameMode mode() {
            return mode;
        }

        public long completedAtEpochMillis() {
            return completedAtEpochMillis;
        }

        public int score() {
            return score;
        }

        public int opponentScore() {
            return opponentScore;
        }

        public int watermelonsCreated() {
            return watermelonsCreated;
        }

        public int opponentWatermelonsCreated() {
            return opponentWatermelonsCreated;
        }

        public int dropCount() {
            return dropCount;
        }

        public RecordResult result() {
            return result;
        }
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
        private final int highestSoloScore;
        private final int highestAiDemoScore;
        private final int soloGames;
        private final int aiDemoGames;
        private final int maxWatermelonsInGame;
        private final int totalWatermelons;
        private final int versusWins;
        private final int versusLosses;
        private final int versusDraws;
        private final int highestVersusScore;

        private History(
                int totalGames,
                int highScore,
                int highestSoloScore,
                int highestAiDemoScore,
                int soloGames,
                int aiDemoGames,
                int maxWatermelonsInGame,
                int totalWatermelons,
                int versusWins,
                int versusLosses,
                int versusDraws,
                int highestVersusScore) {
            this.totalGames = totalGames;
            this.highScore = highScore;
            this.highestSoloScore = highestSoloScore;
            this.highestAiDemoScore = highestAiDemoScore;
            this.soloGames = soloGames;
            this.aiDemoGames = aiDemoGames;
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

        public int highestSoloScore() {
            return highestSoloScore;
        }

        public int highestAiDemoScore() {
            return highestAiDemoScore;
        }

        public int soloGames() {
            return soloGames;
        }

        public int aiDemoGames() {
            return aiDemoGames;
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
