package com.fruitmerge.ai.game;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Preferences;
import com.badlogic.gdx.utils.Base64Coder;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * 单个未完成游戏的实时存档。
 *
 * <p>设置和历史仍由 {@link GameProfileStore} 管理；本类只保存可恢复的当前局。
 * 对外是一个逻辑槽，内部使用两个交替 bank：先完整写入非活动 bank 并 flush，再
 * 切换活动指针。应用在任一写入点被 Android 杀死时，读取端都能退回上一份完整
 * 快照，而不会把半截 Base64 当成有效进度。</p>
 *
 * <p>每个 bank 同时保存 schema、递增代数和 SHA-256。读取会先验证摘要，再按带
 * 长度上限的二进制格式解码，并逐项检查物理与规则不变量。损坏的最新 bank 会回退
 * 到旧 bank；两个 bank 都损坏时视为空槽并清理，绝不把异常传播到游戏启动流程。</p>
 */
public final class GameSessionStore {
    public static final String PREFERENCES_NAME =
            "fruit-merge-ai-session-v1";
    public static final long NO_RANDOM_STATE = -1L;

    private static final int SCHEMA_VERSION = 1;
    private static final int PAYLOAD_MAGIC = 0x464D5331; // "FMS1"
    private static final int MAX_PAYLOAD_BYTES = 1_048_576;
    private static final int MAX_PHYSICS_FRUITS = 4096;
    private static final int MAX_REASON_BYTES = 512;
    private static final float FIXED_PHYSICS_STEP = 1f / 120f;
    private static final float RADIUS_EPSILON = 0.001f;
    private static final long RANDOM_STATE_MASK = (1L << 48) - 1L;

    private static final String KEY_SCHEMA = "session.schema";
    private static final String KEY_ACTIVE_BANK = "session.active_bank";
    private static final String BANK_PREFIX = "session.bank.";
    private static final String PAYLOAD_SUFFIX = ".payload";
    private static final String DIGEST_SUFFIX = ".sha256";
    private static final String GENERATION_SUFFIX = ".generation";

    private final Preferences preferences;
    private Session cachedSession;
    private long cachedGeneration;
    private int cachedBank = -1;
    private boolean cacheLoaded;

    /** 打开 Android/libGDX 应用私有的进度槽。 */
    public static GameSessionStore open() {
        if (Gdx.app == null) {
            throw new IllegalStateException("libGDX application is not ready");
        }
        return new GameSessionStore(
                Gdx.app.getPreferences(PREFERENCES_NAME)
        );
    }

    /** 可注入 Preferences 的构造器，供平台代码和单元测试使用。 */
    public GameSessionStore(Preferences preferences) {
        this.preferences = Objects.requireNonNull(
                preferences,
                "preferences"
        );
    }

    /** 是否存在通过版本、摘要和字段校验的存档。 */
    public synchronized boolean hasSavedSession() {
        ensureLoaded();
        return cachedSession != null;
    }

    /**
     * 读取当前存档；空槽或两个 bank 均损坏时返回 {@code null}。
     *
     * <p>{@link Session} 及其子状态均不可变，数组 getter 返回副本，调用方可以安全
     * 长期持有返回值。</p>
     */
    public synchronized Session load() {
        ensureLoaded();
        return cachedSession;
    }

    /**
     * 保存一份完整快照。
     *
     * <p>序列化和校验在写 Preferences 前完成，所以调用方传入非法状态不会覆盖
     * 上一份有效进度。</p>
     */
    public synchronized void save(Session session) {
        Session safeSession = Objects.requireNonNull(session, "session");
        validateSession(safeSession);
        byte[] bytes = encode(safeSession);
        if (bytes.length > MAX_PAYLOAD_BYTES) {
            throw new IllegalArgumentException(
                    "session payload is too large"
            );
        }

        ensureLoaded();
        if (cachedGeneration == Long.MAX_VALUE) {
            /*
             * 理论上的代数回绕必须同时清掉旧 MAX_VALUE bank，否则下次启动会把
             * 新 generation=1 错判为更旧。正常设备不可能在寿命内触达此分支。
             */
            preferences.clear();
            preferences.flush();
            cachedGeneration = 0L;
            cachedBank = -1;
        }
        long nextGeneration = cachedGeneration + 1L;
        if (nextGeneration <= 0L) {
            nextGeneration = 1L;
        }
        int nextBank = cachedBank == 0 ? 1 : 0;
        String payload = new String(Base64Coder.encode(bytes));
        String digest = sha256Hex(bytes);

        preferences
                .putInteger(KEY_SCHEMA, SCHEMA_VERSION)
                .putString(bankKey(nextBank, PAYLOAD_SUFFIX), payload)
                .putString(bankKey(nextBank, DIGEST_SUFFIX), digest)
                .putLong(
                        bankKey(nextBank, GENERATION_SUFFIX),
                        nextGeneration
                );
        preferences.flush();

        preferences.putInteger(KEY_ACTIVE_BANK, nextBank);
        preferences.flush();

        cachedSession = safeSession;
        cachedGeneration = nextGeneration;
        cachedBank = nextBank;
        cacheLoaded = true;
    }

    /** 删除逻辑槽及两个内部 bank。 */
    public synchronized void clear() {
        preferences.clear();
        preferences.flush();
        cachedSession = null;
        cachedGeneration = 0L;
        cachedBank = -1;
        cacheLoaded = true;
    }

    /**
     * 丢弃内存缓存并重新读取 Preferences。
     *
     * <p>正常应用不需要调用；主要用于测试和平台恢复后显式刷新。</p>
     */
    public synchronized void reload() {
        cachedSession = null;
        cachedGeneration = 0L;
        cachedBank = -1;
        cacheLoaded = false;
        ensureLoaded();
    }

    private void ensureLoaded() {
        if (cacheLoaded) {
            return;
        }
        cacheLoaded = true;

        if (preferences.getInteger(KEY_SCHEMA, SCHEMA_VERSION)
                != SCHEMA_VERSION) {
            clear();
            return;
        }

        Bank first = readBank(0);
        Bank second = readBank(1);
        Bank selected = selectNewest(first, second);
        if (selected == null) {
            if (hasAnyBankData()) {
                clear();
            }
            return;
        }

        cachedSession = selected.session;
        cachedGeneration = selected.generation;
        cachedBank = selected.index;
        int active = preferences.getInteger(KEY_ACTIVE_BANK, -1);
        if (active != selected.index) {
            // 修复因进程在第二次 flush 前终止而留下的过期活动指针。
            preferences
                    .putInteger(KEY_SCHEMA, SCHEMA_VERSION)
                    .putInteger(KEY_ACTIVE_BANK, selected.index);
            preferences.flush();
        }
    }

    private Bank readBank(int index) {
        String payloadKey = bankKey(index, PAYLOAD_SUFFIX);
        String digestKey = bankKey(index, DIGEST_SUFFIX);
        String generationKey = bankKey(index, GENERATION_SUFFIX);
        if (!preferences.contains(payloadKey)
                || !preferences.contains(digestKey)
                || !preferences.contains(generationKey)) {
            return null;
        }

        long generation = preferences.getLong(generationKey, 0L);
        if (generation <= 0L) {
            return null;
        }
        String encoded = preferences.getString(payloadKey, "");
        String expectedDigest = preferences.getString(digestKey, "");
        if (encoded.isEmpty() || expectedDigest.length() != 64) {
            return null;
        }

        try {
            byte[] bytes = Base64Coder.decode(encoded);
            if (bytes.length == 0 || bytes.length > MAX_PAYLOAD_BYTES) {
                return null;
            }
            byte[] expected = parseHex(expectedDigest);
            byte[] actual = sha256(bytes);
            if (!MessageDigest.isEqual(expected, actual)) {
                return null;
            }
            Session session = decode(bytes);
            validateSession(session);
            return new Bank(index, generation, session);
        } catch (RuntimeException | IOException exception) {
            return null;
        }
    }

    private Bank selectNewest(Bank first, Bank second) {
        if (first == null) {
            return second;
        }
        if (second == null) {
            return first;
        }
        if (first.generation == second.generation) {
            int active = preferences.getInteger(KEY_ACTIVE_BANK, -1);
            return active == second.index ? second : first;
        }
        return first.generation > second.generation ? first : second;
    }

    private boolean hasAnyBankData() {
        return preferences.contains(bankKey(0, PAYLOAD_SUFFIX))
                || preferences.contains(bankKey(0, DIGEST_SUFFIX))
                || preferences.contains(bankKey(0, GENERATION_SUFFIX))
                || preferences.contains(bankKey(1, PAYLOAD_SUFFIX))
                || preferences.contains(bankKey(1, DIGEST_SUFFIX))
                || preferences.contains(bankKey(1, GENERATION_SUFFIX));
    }

    private static String bankKey(int bank, String suffix) {
        return BANK_PREFIX + bank + suffix;
    }

    private static byte[] encode(Session session) {
        try {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            DataOutputStream output = new DataOutputStream(bytes);
            output.writeInt(PAYLOAD_MAGIC);
            output.writeInt(SCHEMA_VERSION);
            output.writeLong(session.sessionId);
            output.writeLong(session.savedAtEpochMillis);
            output.writeByte(session.mode.wireId);
            if (session.mode == Mode.DUEL) {
                writeDuelState(output, session.duel);
            } else {
                writeSingleState(output, session.single);
            }
            output.flush();
            return bytes.toByteArray();
        } catch (IOException impossible) {
            throw new IllegalStateException(
                    "in-memory session encoding failed",
                    impossible
            );
        }
    }

    private static Session decode(byte[] bytes) throws IOException {
        DataInputStream input = new DataInputStream(
                new ByteArrayInputStream(bytes)
        );
        if (input.readInt() != PAYLOAD_MAGIC) {
            throw new IllegalArgumentException("session magic mismatch");
        }
        if (input.readInt() != SCHEMA_VERSION) {
            throw new IllegalArgumentException("session version mismatch");
        }
        long sessionId = input.readLong();
        long savedAt = input.readLong();
        Mode mode = Mode.fromWireId(input.readUnsignedByte());
        Session result;
        if (mode == Mode.DUEL) {
            result = new Session(
                    sessionId,
                    savedAt,
                    mode,
                    null,
                    readDuelState(input)
            );
        } else {
            result = new Session(
                    sessionId,
                    savedAt,
                    mode,
                    readSingleState(input),
                    null
            );
        }
        if (input.read() != -1) {
            throw new IllegalArgumentException(
                    "session payload has trailing data"
            );
        }
        return result;
    }

    private static void writeSingleState(
            DataOutputStream output,
            SingleState state) throws IOException {
        output.writeLong(state.queueRandomState);
        writeIntArray(output, state.queueLevels);
        output.writeInt(state.currentLevel);
        output.writeInt(state.score);
        output.writeInt(state.lastScore);
        output.writeInt(state.displayedScore);
        output.writeInt(state.displayedBestScore);
        output.writeInt(state.stepCount);
        output.writeInt(state.watermelonCount);
        output.writeFloat(state.previewX);
        output.writeFloat(state.previewAnchorX);
        output.writeFloat(state.dropCooldownSeconds);
        output.writeFloat(state.stableSeconds);
        output.writeFloat(state.dangerSeconds);
        output.writeFloat(state.aiLoadingSeconds);
        output.writeBoolean(state.waiting);
        output.writeBoolean(state.alive);
        output.writeBoolean(state.resultRecorded);
        output.writeInt(state.resultPercentile);
        writePhysics(output, state.physics);
    }

    private static SingleState readSingleState(
            DataInputStream input) throws IOException {
        return new SingleState(
                input.readLong(),
                readIntArray(input, FruitRules.QUEUE_LENGTH),
                input.readInt(),
                input.readInt(),
                input.readInt(),
                input.readInt(),
                input.readInt(),
                input.readInt(),
                input.readInt(),
                input.readFloat(),
                input.readFloat(),
                input.readFloat(),
                input.readFloat(),
                input.readFloat(),
                input.readFloat(),
                input.readBoolean(),
                input.readBoolean(),
                input.readBoolean(),
                input.readInt(),
                readPhysics(input)
        );
    }

    private static void writeDuelState(
            DataOutputStream output,
            DuelState state) throws IOException {
        writeDuelSnapshot(output, state.match);
        output.writeByte(state.foreground.wireId);
        output.writeFloat(state.resultHoldRemainingSeconds);
        output.writeBoolean(state.resultVisible);
        output.writeBoolean(state.resultRecorded);
        output.writeInt(state.resultPercentile);
        writeString(output, state.resultReason);
        output.writeBoolean(state.aiArmed);
        output.writeFloat(state.aiArmedX);
    }

    private static DuelState readDuelState(
            DataInputStream input) throws IOException {
        return new DuelState(
                readDuelSnapshot(input),
                Side.fromWireId(input.readUnsignedByte()),
                input.readFloat(),
                input.readBoolean(),
                input.readBoolean(),
                input.readInt(),
                readString(input),
                input.readBoolean(),
                input.readFloat()
        );
    }

    private static void writeDuelSnapshot(
            DataOutputStream output,
            DuelMatch.Snapshot snapshot) throws IOException {
        output.writeLong(snapshot.queueRandomState());
        output.writeFloat(snapshot.roundDurationSeconds());
        output.writeFloat(snapshot.nextRoundDelaySeconds());
        writeIntArray(output, snapshot.queueLevels());
        output.writeInt(snapshot.currentLevel());
        output.writeInt(snapshot.roundIndex());
        output.writeLong(snapshot.matchGeneration());
        output.writeFloat(snapshot.roundRemainingSeconds());
        output.writeFloat(snapshot.nextRoundRemainingSeconds());
        output.writeBoolean(snapshot.roundOpen());
        output.writeByte(outcomeWireId(snapshot.outcome()));
        writeLaneSnapshot(output, snapshot.player());
        writeLaneSnapshot(output, snapshot.ai());
    }

    private static DuelMatch.Snapshot readDuelSnapshot(
            DataInputStream input) throws IOException {
        return new DuelMatch.Snapshot(
                input.readLong(),
                input.readFloat(),
                input.readFloat(),
                readIntArray(input, FruitRules.QUEUE_LENGTH),
                input.readInt(),
                input.readInt(),
                input.readLong(),
                input.readFloat(),
                input.readFloat(),
                input.readBoolean(),
                outcomeFromWireId(input.readUnsignedByte()),
                readLaneSnapshot(input),
                readLaneSnapshot(input)
        );
    }

    private static void writeLaneSnapshot(
            DataOutputStream output,
            DuelMatch.LaneSnapshot snapshot) throws IOException {
        output.writeInt(snapshot.score());
        output.writeInt(snapshot.lastScore());
        output.writeInt(snapshot.stepCount());
        output.writeInt(snapshot.watermelonCount());
        output.writeFloat(snapshot.dangerSeconds());
        output.writeFloat(snapshot.previewX());
        output.writeBoolean(snapshot.alive());
        output.writeBoolean(snapshot.submittedThisRound());
        writePhysics(output, snapshot.physics());
    }

    private static DuelMatch.LaneSnapshot readLaneSnapshot(
            DataInputStream input) throws IOException {
        return new DuelMatch.LaneSnapshot(
                input.readInt(),
                input.readInt(),
                input.readInt(),
                input.readInt(),
                input.readFloat(),
                input.readFloat(),
                input.readBoolean(),
                input.readBoolean(),
                readPhysics(input)
        );
    }

    private static void writePhysics(
            DataOutputStream output,
            FruitPhysicsWorld.Snapshot snapshot) throws IOException {
        output.writeInt(snapshot.nextFruitId());
        output.writeFloat(snapshot.accumulatorSeconds());
        output.writeInt(snapshot.fruitCount());
        for (FruitPhysicsWorld.FruitState fruit : snapshot.fruits()) {
            output.writeInt(fruit.id());
            output.writeInt(fruit.level());
            output.writeFloat(fruit.displayRadius());
            output.writeFloat(fruit.physicsRadius());
            output.writeFloat(fruit.x());
            output.writeFloat(fruit.y());
            output.writeFloat(fruit.vx());
            output.writeFloat(fruit.vy());
            output.writeFloat(fruit.angle());
            output.writeFloat(fruit.angularVelocity());
            output.writeInt(fruit.ageFrames());
        }
    }

    private static FruitPhysicsWorld.Snapshot readPhysics(
            DataInputStream input) throws IOException {
        int nextFruitId = input.readInt();
        float accumulator = input.readFloat();
        int count = input.readInt();
        if (count < 0 || count > MAX_PHYSICS_FRUITS) {
            throw new IllegalArgumentException(
                    "physics fruit count is invalid"
            );
        }
        FruitPhysicsWorld.FruitState[] fruits =
                new FruitPhysicsWorld.FruitState[count];
        for (int index = 0; index < count; index++) {
            fruits[index] = new FruitPhysicsWorld.FruitState(
                    input.readInt(),
                    input.readInt(),
                    input.readFloat(),
                    input.readFloat(),
                    input.readFloat(),
                    input.readFloat(),
                    input.readFloat(),
                    input.readFloat(),
                    input.readFloat(),
                    input.readFloat(),
                    input.readInt()
            );
        }
        return new FruitPhysicsWorld.Snapshot(
                nextFruitId,
                accumulator,
                fruits
        );
    }

    private static void writeIntArray(
            DataOutputStream output,
            int[] values) throws IOException {
        output.writeInt(values.length);
        for (int value : values) {
            output.writeInt(value);
        }
    }

    private static int[] readIntArray(
            DataInputStream input,
            int expectedLength) throws IOException {
        int length = input.readInt();
        if (length != expectedLength) {
            throw new IllegalArgumentException(
                    "session queue length is invalid"
            );
        }
        int[] result = new int[length];
        for (int index = 0; index < length; index++) {
            result[index] = input.readInt();
        }
        return result;
    }

    private static void writeString(
            DataOutputStream output,
            String value) throws IOException {
        byte[] encoded = value.getBytes(StandardCharsets.UTF_8);
        if (encoded.length > MAX_REASON_BYTES) {
            throw new IllegalArgumentException(
                    "session text is too long"
            );
        }
        output.writeInt(encoded.length);
        output.write(encoded);
    }

    private static String readString(
            DataInputStream input) throws IOException {
        int length = input.readInt();
        if (length < 0 || length > MAX_REASON_BYTES) {
            throw new IllegalArgumentException(
                    "session text length is invalid"
            );
        }
        byte[] encoded = new byte[length];
        input.readFully(encoded);
        return new String(encoded, StandardCharsets.UTF_8);
    }

    private static int outcomeWireId(DuelMatch.Outcome outcome) {
        switch (outcome) {
            case IN_PROGRESS:
                return 0;
            case PLAYER_WIN:
                return 1;
            case AI_WIN:
                return 2;
            case DRAW:
                return 3;
            default:
                throw new IllegalArgumentException("unknown duel outcome");
        }
    }

    private static DuelMatch.Outcome outcomeFromWireId(int wireId) {
        switch (wireId) {
            case 0:
                return DuelMatch.Outcome.IN_PROGRESS;
            case 1:
                return DuelMatch.Outcome.PLAYER_WIN;
            case 2:
                return DuelMatch.Outcome.AI_WIN;
            case 3:
                return DuelMatch.Outcome.DRAW;
            default:
                throw new IllegalArgumentException(
                        "unknown duel outcome id"
                );
        }
    }

    private static void validateSession(Session session) {
        if (session.sessionId <= 0L) {
            throw new IllegalArgumentException(
                    "sessionId must be positive"
            );
        }
        if (session.savedAtEpochMillis < 0L) {
            throw new IllegalArgumentException(
                    "savedAtEpochMillis must not be negative"
            );
        }
        if (session.mode == Mode.DUEL) {
            if (session.duel == null || session.single != null) {
                throw new IllegalArgumentException(
                        "duel mode must contain only duel state"
                );
            }
            validateDuelState(session.duel);
        } else {
            if (session.single == null || session.duel != null) {
                throw new IllegalArgumentException(
                        "single-board mode must contain only single state"
                );
            }
            validateSingleState(session.single);
        }
    }

    private static void validateSingleState(SingleState state) {
        if (state.queueRandomState != NO_RANDOM_STATE
                && (state.queueRandomState & ~RANDOM_STATE_MASK) != 0L) {
            throw new IllegalArgumentException(
                    "single queue random state must fit in 48 bits"
            );
        }
        validateQueue(state.queueLevels, state.currentLevel);
        requireNonNegative(state.score, "score");
        requireNonNegative(state.lastScore, "lastScore");
        requireNonNegative(state.displayedScore, "displayedScore");
        requireNonNegative(
                state.displayedBestScore,
                "displayedBestScore"
        );
        requireNonNegative(state.stepCount, "stepCount");
        requireNonNegative(state.watermelonCount, "watermelonCount");
        if (state.lastScore > state.score
                || state.displayedScore > state.score) {
            throw new IllegalArgumentException(
                    "single score presentation exceeds logical score"
            );
        }
        validatePreview(state.previewX, state.currentLevel, "previewX");
        validatePreview(
                state.previewAnchorX,
                state.currentLevel,
                "previewAnchorX"
        );
        requireFiniteNonNegative(
                state.dropCooldownSeconds,
                "dropCooldownSeconds"
        );
        requireFiniteNonNegative(state.stableSeconds, "stableSeconds");
        requireFiniteNonNegative(state.dangerSeconds, "dangerSeconds");
        requireFiniteNonNegative(
                state.aiLoadingSeconds,
                "aiLoadingSeconds"
        );
        if (state.resultPercentile < 0
                || state.resultPercentile > 99) {
            throw new IllegalArgumentException(
                    "resultPercentile must be in [0, 99]"
            );
        }
        if (state.resultRecorded && state.alive) {
            throw new IllegalArgumentException(
                    "a recorded result cannot still be alive"
            );
        }
        validatePhysics(state.physics);
    }

    private static void validateDuelState(DuelState state) {
        Objects.requireNonNull(state.match, "duel match");
        Objects.requireNonNull(state.foreground, "duel foreground");
        Objects.requireNonNull(state.resultReason, "duel resultReason");
        requireFiniteNonNegative(
                state.resultHoldRemainingSeconds,
                "resultHoldRemainingSeconds"
        );
        if (state.resultPercentile < 0
                || state.resultPercentile > 99) {
            throw new IllegalArgumentException(
                    "resultPercentile must be in [0, 99]"
            );
        }
        if (state.resultReason.getBytes(StandardCharsets.UTF_8).length
                > MAX_REASON_BYTES) {
            throw new IllegalArgumentException(
                    "duel resultReason is too long"
            );
        }
        if (!Float.isFinite(state.aiArmedX)) {
            throw new IllegalArgumentException(
                    "duel aiArmedX must be finite"
            );
        }
        DuelMatch.Outcome outcome = state.match.outcome();
        if (outcome == DuelMatch.Outcome.IN_PROGRESS
                && (state.resultVisible || state.resultRecorded)) {
            throw new IllegalArgumentException(
                    "an active duel cannot have a recorded result"
            );
        }
        if (state.aiArmed
                && (outcome != DuelMatch.Outcome.IN_PROGRESS
                || !state.match.roundOpen()
                || state.match.player().submittedThisRound()
                || state.match.ai().submittedThisRound())) {
            throw new IllegalArgumentException(
                    "duel AI can only be armed while both sides may submit"
            );
        }
        if (state.aiArmed) {
            validatePreview(
                    state.aiArmedX,
                    state.match.currentLevel(),
                    "duel aiArmedX"
            );
            if (Math.abs(
                    state.aiArmedX
                            - state.match.ai().previewX()
            ) > 0.01f) {
                throw new IllegalArgumentException(
                        "armed AI position must match its preview"
                );
            }
        }
        validateDuelSnapshot(state.match);
    }

    private static void validateDuelSnapshot(
            DuelMatch.Snapshot snapshot) {
        validateQueue(snapshot.queueLevels(), snapshot.currentLevel());
        validateLaneSnapshot(snapshot.player(), snapshot.currentLevel());
        validateLaneSnapshot(snapshot.ai(), snapshot.currentLevel());
    }

    private static void validateLaneSnapshot(
            DuelMatch.LaneSnapshot lane,
            int currentLevel) {
        Objects.requireNonNull(lane, "duel lane");
        validatePreview(lane.previewX(), currentLevel, "lane previewX");
        validatePhysics(lane.physics());
    }

    private static void validateQueue(int[] queue, int currentLevel) {
        Objects.requireNonNull(queue, "queue");
        if (queue.length != FruitRules.QUEUE_LENGTH) {
            throw new IllegalArgumentException(
                    "queue must contain exactly "
                            + FruitRules.QUEUE_LENGTH + " levels"
            );
        }
        for (int level : queue) {
            if (level < FruitRules.SPAWN_MIN_LEVEL
                    || level > FruitRules.SPAWN_MAX_LEVEL) {
                throw new IllegalArgumentException(
                        "queue level is outside spawn range"
                );
            }
        }
        if (currentLevel != queue[0]) {
            throw new IllegalArgumentException(
                    "currentLevel must equal queue head"
            );
        }
    }

    private static void validatePreview(
            float preview,
            int level,
            String name) {
        if (!Float.isFinite(preview)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
        float clamped = FruitRules.clampDropX(preview, level);
        if (Math.abs(preview - clamped) > 0.01f) {
            throw new IllegalArgumentException(
                    name + " is outside legal drop bounds"
            );
        }
    }

    private static void validatePhysics(
            FruitPhysicsWorld.Snapshot snapshot) {
        Objects.requireNonNull(snapshot, "physics snapshot");
        if (snapshot.fruitCount() < 0
                || snapshot.fruitCount() > MAX_PHYSICS_FRUITS) {
            throw new IllegalArgumentException(
                    "physics snapshot contains too many fruits"
            );
        }
        if (snapshot.nextFruitId() <= 0
                || snapshot.nextFruitId() == Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "physics nextFruitId is invalid"
            );
        }
        float accumulator = snapshot.accumulatorSeconds();
        if (!Float.isFinite(accumulator)
                || accumulator < 0f
                || accumulator >= FIXED_PHYSICS_STEP) {
            throw new IllegalArgumentException(
                    "physics accumulator is invalid"
            );
        }

        Set<Integer> ids = new HashSet<>();
        int greatestId = 0;
        for (FruitPhysicsWorld.FruitState fruit : snapshot.fruits()) {
            if (fruit == null) {
                throw new IllegalArgumentException(
                        "physics snapshot contains a null fruit"
                );
            }
            if (fruit.id() <= 0 || !ids.add(fruit.id())) {
                throw new IllegalArgumentException(
                        "physics fruit id is invalid or duplicated"
                );
            }
            greatestId = Math.max(greatestId, fruit.id());
            if (fruit.level() < FruitRules.MIN_LEVEL
                    || fruit.level() > FruitRules.MAX_LEVEL) {
                throw new IllegalArgumentException(
                        "physics fruit level is invalid"
                );
            }
            if (!approximatelyEqual(
                    fruit.displayRadius(),
                    FruitRules.displayRadius(fruit.level()))) {
                throw new IllegalArgumentException(
                        "physics display radius is invalid"
                );
            }
            float dropped = FruitRules.droppedPhysicsRadius(fruit.level());
            float merged = FruitRules.mergedPhysicsRadius(fruit.level());
            boolean radiusMatches = approximatelyEqual(
                    fruit.physicsRadius(),
                    dropped
            ) || (fruit.level() > FruitRules.MIN_LEVEL
                    && approximatelyEqual(
                    fruit.physicsRadius(),
                    merged
            ));
            if (!radiusMatches) {
                throw new IllegalArgumentException(
                        "physics collision radius is invalid"
                );
            }
            if (!Float.isFinite(fruit.x())
                    || !Float.isFinite(fruit.y())
                    || !Float.isFinite(fruit.vx())
                    || !Float.isFinite(fruit.vy())
                    || !Float.isFinite(fruit.angle())
                    || !Float.isFinite(fruit.angularVelocity())
                    || fruit.ageFrames() < 0) {
                throw new IllegalArgumentException(
                        "physics fruit motion is invalid"
                );
            }
        }
        if (snapshot.nextFruitId() <= greatestId) {
            throw new IllegalArgumentException(
                    "physics nextFruitId must exceed existing ids"
            );
        }
    }

    private static boolean approximatelyEqual(float first, float second) {
        return Float.isFinite(first)
                && Math.abs(first - second) <= RADIUS_EPSILON;
    }

    private static void requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0");
        }
    }

    private static void requireFiniteNonNegative(
            float value,
            String name) {
        if (!Float.isFinite(value) || value < 0f) {
            throw new IllegalArgumentException(
                    name + " must be finite and >= 0"
            );
        }
    }

    private static String sha256Hex(byte[] value) {
        byte[] digest = sha256(value);
        StringBuilder result = new StringBuilder(digest.length * 2);
        for (byte element : digest) {
            result.append(Character.forDigit((element >>> 4) & 0xF, 16));
            result.append(Character.forDigit(element & 0xF, 16));
        }
        return result.toString();
    }

    private static byte[] sha256(byte[] value) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(value);
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException(
                    "SHA-256 is unavailable",
                    impossible
            );
        }
    }

    private static byte[] parseHex(String value) {
        if (value.length() != 64) {
            throw new IllegalArgumentException("invalid SHA-256 text");
        }
        byte[] result = new byte[value.length() / 2];
        for (int index = 0; index < value.length(); index += 2) {
            int high = Character.digit(value.charAt(index), 16);
            int low = Character.digit(value.charAt(index + 1), 16);
            if (high < 0 || low < 0) {
                throw new IllegalArgumentException(
                        "invalid SHA-256 text"
                );
            }
            result[index / 2] = (byte) ((high << 4) | low);
        }
        return result;
    }

    /** 三种可保存的运行模式。 */
    public enum Mode {
        SOLO(0),
        AI_DEMO(1),
        DUEL(2);

        private final int wireId;

        Mode(int wireId) {
            this.wireId = wireId;
        }

        private static Mode fromWireId(int wireId) {
            for (Mode mode : values()) {
                if (mode.wireId == wireId) {
                    return mode;
                }
            }
            throw new IllegalArgumentException("unknown session mode");
        }
    }

    /** 对战画面当前清晰显示的一方。 */
    public enum Side {
        PLAYER(0),
        AI(1);

        private final int wireId;

        Side(int wireId) {
            this.wireId = wireId;
        }

        private static Side fromWireId(int wireId) {
            for (Side side : values()) {
                if (side.wireId == wireId) {
                    return side;
                }
            }
            throw new IllegalArgumentException(
                    "unknown foreground side"
            );
        }
    }

    /** 一个逻辑进度槽中的不可变顶层值。 */
    public static final class Session {
        private final long sessionId;
        private final long savedAtEpochMillis;
        private final Mode mode;
        private final SingleState single;
        private final DuelState duel;

        public Session(
                long sessionId,
                long savedAtEpochMillis,
                Mode mode,
                SingleState single,
                DuelState duel) {
            this.sessionId = sessionId;
            this.savedAtEpochMillis = savedAtEpochMillis;
            this.mode = Objects.requireNonNull(mode, "mode");
            this.single = single;
            this.duel = duel;
            validateSession(this);
        }

        public static Session single(
                long sessionId,
                long savedAtEpochMillis,
                Mode mode,
                SingleState state) {
            if (mode == Mode.DUEL) {
                throw new IllegalArgumentException(
                        "single session mode cannot be DUEL"
                );
            }
            return new Session(
                    sessionId,
                    savedAtEpochMillis,
                    mode,
                    state,
                    null
            );
        }

        public static Session duel(
                long sessionId,
                long savedAtEpochMillis,
                DuelState state) {
            return new Session(
                    sessionId,
                    savedAtEpochMillis,
                    Mode.DUEL,
                    null,
                    state
            );
        }

        public long sessionId() {
            return sessionId;
        }

        public long savedAtEpochMillis() {
            return savedAtEpochMillis;
        }

        public Mode mode() {
            return mode;
        }

        public SingleState single() {
            return single;
        }

        public DuelState duel() {
            return duel;
        }
    }

    /** 单人和 AI 演示共用的单棋盘规则状态。 */
    public static final class SingleState {
        private final long queueRandomState;
        private final int[] queueLevels;
        private final int currentLevel;
        private final int score;
        private final int lastScore;
        private final int displayedScore;
        private final int displayedBestScore;
        private final int stepCount;
        private final int watermelonCount;
        private final float previewX;
        private final float previewAnchorX;
        private final float dropCooldownSeconds;
        private final float stableSeconds;
        private final float dangerSeconds;
        private final float aiLoadingSeconds;
        private final boolean waiting;
        private final boolean alive;
        private final boolean resultRecorded;
        private final int resultPercentile;
        private final FruitPhysicsWorld.Snapshot physics;

        public SingleState(
                long queueRandomState,
                int[] queueLevels,
                int currentLevel,
                int score,
                int lastScore,
                int displayedScore,
                int displayedBestScore,
                int stepCount,
                int watermelonCount,
                float previewX,
                float previewAnchorX,
                float dropCooldownSeconds,
                float stableSeconds,
                float dangerSeconds,
                float aiLoadingSeconds,
                boolean waiting,
                boolean alive,
                boolean resultRecorded,
                int resultPercentile,
                FruitPhysicsWorld.Snapshot physics) {
            this.queueRandomState = queueRandomState;
            this.queueLevels = Objects.requireNonNull(
                    queueLevels,
                    "queueLevels"
            ).clone();
            this.currentLevel = currentLevel;
            this.score = score;
            this.lastScore = lastScore;
            this.displayedScore = displayedScore;
            this.displayedBestScore = displayedBestScore;
            this.stepCount = stepCount;
            this.watermelonCount = watermelonCount;
            this.previewX = previewX;
            this.previewAnchorX = previewAnchorX;
            this.dropCooldownSeconds = dropCooldownSeconds;
            this.stableSeconds = stableSeconds;
            this.dangerSeconds = dangerSeconds;
            this.aiLoadingSeconds = aiLoadingSeconds;
            this.waiting = waiting;
            this.alive = alive;
            this.resultRecorded = resultRecorded;
            this.resultPercentile = resultPercentile;
            this.physics = Objects.requireNonNull(physics, "physics");
            validateSingleState(this);
        }

        public long queueRandomState() {
            return queueRandomState;
        }

        public int[] queueLevels() {
            return queueLevels.clone();
        }

        public int currentLevel() {
            return currentLevel;
        }

        public int score() {
            return score;
        }

        public int lastScore() {
            return lastScore;
        }

        public int displayedScore() {
            return displayedScore;
        }

        public int displayedBestScore() {
            return displayedBestScore;
        }

        public int stepCount() {
            return stepCount;
        }

        public int watermelonCount() {
            return watermelonCount;
        }

        public float previewX() {
            return previewX;
        }

        public float previewAnchorX() {
            return previewAnchorX;
        }

        public float dropCooldownSeconds() {
            return dropCooldownSeconds;
        }

        public float stableSeconds() {
            return stableSeconds;
        }

        public float dangerSeconds() {
            return dangerSeconds;
        }

        public float aiLoadingSeconds() {
            return aiLoadingSeconds;
        }

        public boolean waiting() {
            return waiting;
        }

        public boolean alive() {
            return alive;
        }

        public boolean resultRecorded() {
            return resultRecorded;
        }

        public int resultPercentile() {
            return resultPercentile;
        }

        public FruitPhysicsWorld.Snapshot physics() {
            return physics;
        }
    }

    /** AI 对战的规则状态与少量不可推导的表现/结算状态。 */
    public static final class DuelState {
        private final DuelMatch.Snapshot match;
        private final Side foreground;
        private final float resultHoldRemainingSeconds;
        private final boolean resultVisible;
        private final boolean resultRecorded;
        private final int resultPercentile;
        private final String resultReason;
        private final boolean aiArmed;
        private final float aiArmedX;

        public DuelState(
                DuelMatch.Snapshot match,
                Side foreground,
                float resultHoldRemainingSeconds,
                boolean resultVisible,
                boolean resultRecorded,
                int resultPercentile,
                String resultReason,
                boolean aiArmed,
                float aiArmedX) {
            this.match = Objects.requireNonNull(match, "match");
            this.foreground = Objects.requireNonNull(
                    foreground,
                    "foreground"
            );
            this.resultHoldRemainingSeconds =
                    resultHoldRemainingSeconds;
            this.resultVisible = resultVisible;
            this.resultRecorded = resultRecorded;
            this.resultPercentile = resultPercentile;
            this.resultReason = Objects.requireNonNull(
                    resultReason,
                    "resultReason"
            );
            this.aiArmed = aiArmed;
            this.aiArmedX = aiArmedX;
            validateDuelState(this);
        }

        public DuelMatch.Snapshot match() {
            return match;
        }

        public Side foreground() {
            return foreground;
        }

        public float resultHoldRemainingSeconds() {
            return resultHoldRemainingSeconds;
        }

        public boolean resultVisible() {
            return resultVisible;
        }

        public boolean resultRecorded() {
            return resultRecorded;
        }

        public int resultPercentile() {
            return resultPercentile;
        }

        public String resultReason() {
            return resultReason;
        }

        public boolean aiArmed() {
            return aiArmed;
        }

        public float aiArmedX() {
            return aiArmedX;
        }
    }

    private static final class Bank {
        private final int index;
        private final long generation;
        private final Session session;

        private Bank(int index, long generation, Session session) {
            this.index = index;
            this.generation = generation;
            this.session = session;
        }
    }
}
