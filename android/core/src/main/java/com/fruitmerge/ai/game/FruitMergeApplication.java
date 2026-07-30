package com.fruitmerge.ai.game;

import com.badlogic.gdx.ApplicationAdapter;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.InputProcessor;
import com.badlogic.gdx.audio.Sound;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.GlyphLayout;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.math.MathUtils;
import com.badlogic.gdx.math.Vector3;
import com.badlogic.gdx.utils.Align;
import com.badlogic.gdx.utils.Array;
import com.badlogic.gdx.utils.IntArray;
import com.badlogic.gdx.utils.IntIntMap;
import com.badlogic.gdx.utils.ScreenUtils;
import com.badlogic.gdx.utils.viewport.FitViewport;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * Android 版《合成大西瓜》的完整游戏循环。
 *
 * <p>画面采用 560×1120 固定逻辑坐标，FitViewport 在不同手机上只增加留边，不改变
 * 训练时的几何。手动模式允许拖动并抬手投放；AI 模式只在连续稳定 0.2 秒后请求一次
 * 模型，并把离散动作包装成停顿、试探、回拉和轻微颤动的拟人轨迹。</p>
 */
public final class FruitMergeApplication extends ApplicationAdapter
        implements InputProcessor {
    private static final float PREVIEW_GAP = 12f;
    private static final float MANUAL_DROP_COOLDOWN_SECONDS = 0.14f;
    private static final float AI_DROP_COOLDOWN_SECONDS = 0.36f;
    private static final float MANUAL_INPUT_TOP = 144f;
    private static final float GAME_OVER_SECONDS = 2f;
    private static final float AI_LOADING_FALLBACK_SECONDS = 12f;
    private static final float MODE_BUTTON_LEFT = 188f;
    private static final float MODE_BUTTON_TOP = 18f;
    private static final float MODE_BUTTON_WIDTH = 86f;
    private static final float MODE_BUTTON_HEIGHT = 42f;
    private static final float AI_TOGGLE_LEFT = 282f;
    private static final float AI_TOGGLE_TOP = 18f;
    private static final float AI_TOGGLE_WIDTH = 148f;
    private static final float AI_TOGGLE_HEIGHT = 42f;
    private static final float HISTORY_BUTTON_LEFT = 440f;
    private static final float SETTINGS_BUTTON_LEFT = 490f;
    private static final float UTILITY_BUTTON_TOP = 18f;
    private static final float UTILITY_BUTTON_SIZE = 42f;
    private static final float MERGE_CUE_GAP_SECONDS = 0.105f;
    private static final float MERGE_PRESENTATION_SETTLE_SECONDS = 0.28f;
    private static final float MERGE_PRESENTATION_MAX_HOLD_SECONDS = 1.45f;
    private static final float SCORE_TOKEN_POP_SECONDS = 0.34f;
    private static final float SCORE_TOKEN_FLIGHT_SECONDS = 0.48f;
    private static final float SCORE_TARGET_X = 84f;
    private static final float SCORE_TARGET_Y = 101f;
    private static final float POPUP_FONT_SCALE = 0.50f;

    /*
     * “暖色果园”主题只负责外围表现。水果仍从原来的 01.png～11.png 加载，
     * 颜色、半径、队列位置和 Box2D 坐标都不参与主题换肤。
     */
    private static final Color BACKGROUND_TOP =
            new Color(1f, 0.98f, 0.92f, 1f);
    private static final Color BACKGROUND_BOTTOM =
            new Color(1f, 0.92f, 0.80f, 1f);
    private static final Color BOARD_COLOR =
            new Color(1f, 0.99f, 0.95f, 1f);
    private static final Color BOARD_FRAME =
            new Color(0.96f, 0.68f, 0.41f, 1f);
    private static final Color BOARD_FRAME_SOFT =
            new Color(1f, 0.82f, 0.61f, 1f);
    private static final Color PANEL_COLOR =
            new Color(1f, 0.97f, 0.89f, 0.98f);
    private static final Color SCORE_CARD =
            new Color(1f, 0.99f, 0.95f, 1f);
    private static final Color NEXT_CARD =
            new Color(1f, 0.97f, 0.90f, 1f);
    private static final Color CARD_SHADOW =
            new Color(0.62f, 0.32f, 0.14f, 0.16f);
    private static final Color ACCENT =
            new Color(0.24f, 0.76f, 0.55f, 1f);
    private static final Color ACCENT_DARK =
            new Color(0.08f, 0.58f, 0.39f, 1f);
    private static final Color ACCENT_SOFT =
            new Color(0.73f, 0.94f, 0.83f, 1f);
    private static final Color DANGER =
            new Color(0.95f, 0.43f, 0.30f, 0.94f);
    private static final Color TEXT_PRIMARY =
            new Color(0.34f, 0.16f, 0.09f, 1f);
    private static final Color TEXT_MUTED =
            new Color(0.60f, 0.39f, 0.28f, 1f);
    private static final Color SWITCH_OFF =
            new Color(0.91f, 0.83f, 0.75f, 1f);
    private static final Color MANUAL_HINT =
            new Color(TEXT_MUTED.r, TEXT_MUTED.g, TEXT_MUTED.b, 0.78f);
    private static final Color GAME_OVER_PANEL =
            new Color(1f, 0.97f, 0.89f, 1f);
    private static final Color LEAF_LIGHT =
            new Color(0.53f, 0.75f, 0.26f, 0.16f);
    private static final Color LEAF_DARK =
            new Color(0.29f, 0.59f, 0.18f, 0.13f);
    private static final Color LEAF_ACCENT =
            new Color(0.40f, 0.67f, 0.19f, 0.58f);
    private static final Color ORCHARD_GLOW =
            new Color(1f, 0.60f, 0.30f, 0.10f);
    private static final Color SCORE_GLOW =
            new Color(1f, 0.68f, 0.18f, 1f);
    private static final Color PLAYER_TINT =
            new Color(0.97f, 0.48f, 0.46f, 1f);
    private static final Color PLAYER_TINT_SOFT =
            new Color(1f, 0.84f, 0.82f, 1f);
    private static final Color AI_TINT =
            new Color(0.39f, 0.66f, 0.96f, 1f);
    private static final Color AI_TINT_SOFT =
            new Color(0.81f, 0.90f, 1f, 1f);
    private static final Color OVERLAY_DIM =
            new Color(0.25f, 0.12f, 0.07f, 0.62f);

    private final AiService aiService;
    /*
     * 内容随机、拟人手势和纯表现随机必须相互独立。否则多画几颗爆浆粒子就会改变
     * 后续水果队列，表现层会意外影响游戏规则与模型输入。
     */
    private final Random queueRandom = new Random();
    private final Random motionRandom = new Random();
    private final Random effectRandom = new Random();
    private final IntArray queue = new IntArray();
    private final Array<Particle> particles = new Array<>();
    private final Array<Ring> rings = new Array<>();
    private final Array<MergeBurst> mergeBursts = new Array<>();
    private final Array<MergeCue> mergeCues = new Array<>();
    private final Array<ScoreToken> scoreTokens = new Array<>();
    private final Array<DuelScoreToken> duelScoreTokens = new Array<>();
    private final Array<ScoreSequence> scoreRollQueue = new Array<>();
    private final Vector3 touchPoint = new Vector3();
    private final Color scratchColor = new Color();

    private OrthographicCamera camera;
    private FitViewport viewport;
    private SpriteBatch batch;
    private ShapeRenderer shapes;
    private Texture uiFontTexture;
    private BitmapFont smallFont;
    private BitmapFont normalFont;
    private BitmapFont titleFont;
    private BitmapFont popupFont;
    private GlyphLayout glyphLayout;
    private TextureRegion[] fruitTextures;
    private FruitPhysicsWorld physics;
    private Sound mergePopSound;
    private Sound mergeSoftSound;
    private Sound scoreCollectSound;
    private GameProfileStore profileStore;
    private GameProfileStore.Settings settings;

    private int score;
    private int displayedScore;
    private int displayedBestScore;
    private int lastScore;
    private int bestScore;
    private int stepCount;
    private int currentLevel;
    private boolean waiting;
    private boolean alive;
    private volatile boolean disposed;
    private boolean aiEnabled = true;
    private boolean aiRequestInFlight;
    private int activeDragPointer = -1;
    private float previewX;
    private float dropCooldown;
    private float stableSeconds;
    private float dangerSeconds;
    private float aiLoadingSeconds;
    private float elapsedSeconds;
    private float previewAnchorX;
    private float scorePulse;
    private float scoreRollElapsed;
    private float scoreRollDuration;
    private int scoreRollStart;
    private int scoreRollTarget;
    private int nextScoreSequenceId;
    private int currentWatermelons;
    private int soloPercentile;
    private long decisionEpoch;
    private float historyResetConfirmSeconds;
    private float modeSwitchConfirmSeconds;
    private boolean soloResultRecorded;
    private MotionPlan motionPlan;
    private ScoreSequence activeScoreSequence;
    private ScoreSequence rollingScoreSequence;
    private DuelMatch duelMatch;
    private DuelMatch.Side duelForeground = DuelMatch.Side.PLAYER;
    private MotionPlan duelAiMotionPlan;
    private boolean duelAiRequestInFlight;
    private long duelDecisionEpoch;
    private float duelResultHoldRemaining;
    private boolean duelResultVisible;
    private boolean duelResultRecorded;
    private int duelPercentile;
    private String duelResultReason = "";
    private AiState aiState = AiState.OBSERVING;
    private String aiDetail = "启动中";
    private GameMode gameMode = GameMode.CLASSIC;
    private OverlayPage overlayPage = OverlayPage.NONE;

    public FruitMergeApplication(AiService aiService) {
        this.aiService = aiService;
    }

    @Override
    public void create() {
        camera = new OrthographicCamera();
        viewport = new FitViewport(
                FruitRules.BOARD_WIDTH,
                FruitRules.BOARD_HEIGHT,
                camera
        );
        viewport.apply(true);

        batch = new SpriteBatch();
        shapes = new ShapeRenderer();
        /*
         * 图集原始字号为 64px。所有运行字号都从高分辨率向下采样，避免旧版把
         * libGDX 内置 15px 字体放大 2～4 倍后产生的灰边、重影和锯齿。
         * 四个字号共享同一张纹理，避免 1024² 中文图集被重复上传到 GPU。
         */
        uiFontTexture = new Texture(Gdx.files.internal("fonts/ui-cute.png"));
        uiFontTexture.setFilter(
                Texture.TextureFilter.Linear,
                Texture.TextureFilter.Linear
        );
        smallFont = createFont(0.25f);
        normalFont = createFont(0.38f);
        titleFont = createFont(0.47f);
        popupFont = createFont(POPUP_FONT_SCALE);
        glyphLayout = new GlyphLayout();
        loadFruitTextures();
        loadAudio();
        profileStore = GameProfileStore.open();
        settings = profileStore.settings();
        bestScore = profileStore.history().highScore();

        physics = new FruitPhysicsWorld();
        Gdx.input.setInputProcessor(this);
        resetGame();
    }

    @Override
    public void resize(int width, int height) {
        viewport.update(width, height, true);
    }

    @Override
    public void render() {
        float realDelta = Math.min(Gdx.graphics.getDeltaTime(), 0.05f);
        elapsedSeconds += realDelta;
        historyResetConfirmSeconds = Math.max(
                0f,
                historyResetConfirmSeconds - realDelta
        );
        modeSwitchConfirmSeconds = Math.max(
                0f,
                modeSwitchConfirmSeconds - realDelta
        );
        if (overlayPage == OverlayPage.NONE) {
            float gameDelta = realDelta * settings.gameSpeed();
            if (gameMode == GameMode.CLASSIC) {
                updateGame(gameDelta);
            } else {
                updateDuelGame(realDelta, gameDelta);
            }
        }
        drawGame();
    }

    @Override
    public void pause() {
        // 后台期间不推进冷却或物理；恢复后的 delta 还会被 render 上限保护。
    }

    @Override
    public void dispose() {
        disposed = true;
        decisionEpoch += 1;
        aiRequestInFlight = false;
        motionPlan = null;
        activeDragPointer = -1;
        Gdx.input.setInputProcessor(null);
        if (physics != null) {
            physics.dispose();
        }
        disposeDuelGame();
        if (fruitTextures != null) {
            for (int level = FruitRules.MIN_LEVEL;
                    level <= FruitRules.MAX_LEVEL;
                    level++) {
                if (fruitTextures[level] != null) {
                    fruitTextures[level].getTexture().dispose();
                }
            }
        }
        if (smallFont != null) {
            smallFont.dispose();
        }
        if (normalFont != null) {
            normalFont.dispose();
        }
        if (titleFont != null) {
            titleFont.dispose();
        }
        if (popupFont != null) {
            popupFont.dispose();
        }
        if (uiFontTexture != null) {
            uiFontTexture.dispose();
        }
        if (mergePopSound != null) {
            mergePopSound.dispose();
        }
        if (mergeSoftSound != null) {
            mergeSoftSound.dispose();
        }
        if (scoreCollectSound != null) {
            scoreCollectSound.dispose();
        }
        if (batch != null) {
            batch.dispose();
        }
        if (shapes != null) {
            shapes.dispose();
        }
    }

    private BitmapFont createFont(float scale) {
        BitmapFont font = new BitmapFont(
                Gdx.files.internal("fonts/ui-cute.fnt"),
                new TextureRegion(uiFontTexture),
                false
        );
        font.getData().setScale(scale);
        font.setColor(TEXT_PRIMARY);
        font.setUseIntegerPositions(false);
        return font;
    }

    private void loadAudio() {
        mergePopSound = Gdx.audio.newSound(
                Gdx.files.internal("audio/merge-pop.ogg")
        );
        mergeSoftSound = Gdx.audio.newSound(
                Gdx.files.internal("audio/merge-soft.ogg")
        );
        scoreCollectSound = Gdx.audio.newSound(
                Gdx.files.internal("audio/score-collect.ogg")
        );
    }

    private void loadFruitTextures() {
        fruitTextures = new TextureRegion[FruitRules.MAX_LEVEL + 1];
        for (int level = FruitRules.MIN_LEVEL;
                level <= FruitRules.MAX_LEVEL;
                level++) {
            String filename = String.format("fruits/%02d.png", level);
            Texture texture = new Texture(Gdx.files.internal(filename));
            texture.setFilter(
                    Texture.TextureFilter.Linear,
                    Texture.TextureFilter.Linear
            );
            fruitTextures[level] = new TextureRegion(texture);
        }
    }

    private void resetGame() {
        decisionEpoch += 1;
        aiRequestInFlight = false;
        motionPlan = null;
        activeDragPointer = -1;
        physics.clear();
        queue.clear();
        fillQueue();
        score = 0;
        displayedScore = 0;
        displayedBestScore = bestScore;
        lastScore = 0;
        stepCount = 0;
        stableSeconds = 0f;
        dangerSeconds = 0f;
        aiLoadingSeconds = 0f;
        dropCooldown = 0.18f;
        previewX = FruitRules.BOARD_WIDTH / 2f;
        previewAnchorX = previewX;
        alive = true;
        waiting = true;
        currentLevel = queue.first();
        aiState = aiEnabled ? AiState.OBSERVING : AiState.MANUAL;
        aiDetail = aiService.isAiReady()
                ? "模型已就绪"
                : sanitizeStatus(aiService.aiRuntimeStatus());
        particles.clear();
        rings.clear();
        mergeBursts.clear();
        mergeCues.clear();
        scoreTokens.clear();
        scoreRollQueue.clear();
        activeScoreSequence = null;
        rollingScoreSequence = null;
        scorePulse = 0f;
        scoreRollElapsed = 0f;
        scoreRollDuration = 0f;
        scoreRollStart = 0;
        scoreRollTarget = 0;
        nextScoreSequenceId = 0;
        currentWatermelons = 0;
        soloPercentile = 0;
        soloResultRecorded = false;
    }

    private void fillQueue() {
        while (queue.size < FruitRules.QUEUE_LENGTH) {
            queue.add(
                    FruitRules.SPAWN_MIN_LEVEL
                            + queueRandom.nextInt(
                            FruitRules.SPAWN_MAX_LEVEL
                                    - FruitRules.SPAWN_MIN_LEVEL + 1)
            );
        }
    }

    private void resetDuelGame() {
        disposeDuelGame();
        duelMatch = new DuelMatch(
                queueRandom,
                settings.versusDropSeconds(),
                0.20f
        );
        duelForeground = DuelMatch.Side.PLAYER;
        duelResultHoldRemaining = 0f;
        duelResultVisible = false;
        duelResultRecorded = false;
        duelPercentile = 0;
        duelResultReason = "";
        invalidateDuelDecision();
        activeDragPointer = -1;
        particles.clear();
        rings.clear();
        mergeBursts.clear();
        mergeCues.clear();
        scoreTokens.clear();
        duelScoreTokens.clear();
        scoreRollQueue.clear();
        activeScoreSequence = null;
        rollingScoreSequence = null;
        scorePulse = 0f;
    }

    private void disposeDuelGame() {
        invalidateDuelDecision();
        if (duelMatch != null) {
            duelMatch.dispose();
            duelMatch = null;
        }
        duelScoreTokens.clear();
    }

    private void invalidateDuelDecision() {
        duelDecisionEpoch += 1L;
        duelAiRequestInFlight = false;
        duelAiMotionPlan = null;
    }

    private void updateDuelGame(float realDelta, float gameDelta) {
        if (duelMatch == null) {
            resetDuelGame();
        }
        if (duelMatch.outcome() != DuelMatch.Outcome.IN_PROGRESS) {
            updateEffects(gameDelta);
            updateDuelScoreTokens(gameDelta);
            if (!duelResultVisible) {
                duelResultHoldRemaining = Math.max(
                        0f,
                        duelResultHoldRemaining - realDelta
                );
                if (duelResultHoldRemaining <= 0f) {
                    duelResultVisible = true;
                }
            }
            return;
        }

        int previousRound = duelMatch.roundIndex();
        duelMatch.update(realDelta, settings.gameSpeed());
        consumeDuelMergeEvents();
        updateEffects(gameDelta);
        updateDuelScoreTokens(gameDelta);

        if (duelMatch.outcome() != DuelMatch.Outcome.IN_PROGRESS) {
            beginDuelResult();
            return;
        }
        if (duelMatch.roundIndex() != previousRound) {
            invalidateDuelDecision();
        }

        updateDuelAi(realDelta);
        if (duelMatch.roundOpen()
                && duelMatch.roundRemainingSeconds() <= 0f) {
            if (!duelMatch.playerLane().submittedThisRound()) {
                boolean dropped = duelMatch.timeoutPlayer();
                if (dropped) {
                    spawnDuelDropFeedback(DuelMatch.Side.PLAYER);
                }
            }
            if (!duelMatch.aiLane().submittedThisRound()) {
                AiDecision fallback = fallbackDecision(
                        createDuelSnapshot(duelMatch.aiLane())
                );
                float x = FruitRules.actionDropX(
                        fallback.actionIndex,
                        duelMatch.currentLevel()
                );
                boolean dropped = duelMatch.timeoutAi(x);
                if (dropped) {
                    spawnDuelDropFeedback(DuelMatch.Side.AI);
                }
                invalidateDuelDecision();
            }
        }
    }

    private void consumeDuelMergeEvents() {
        boolean aiSceneChanged = false;
        for (DuelMatch.MergeVisualEvent event
                : duelMatch.drainMergeVisualEvents()) {
            if (event.side() == DuelMatch.Side.AI) {
                /*
                 * AI 的局面在请求发出后仍可能因低速碰撞继续合成。该请求基于旧图，
                 * 必须只失效 AI 自己的票据；玩家场合成不能无端取消对手决策。
                 */
                aiSceneChanged = true;
            }
            boolean foreground = event.side() == duelForeground;
            if (foreground) {
                spawnMergeEffect(
                        event.x(),
                        event.y(),
                        event.level(),
                        1
                );
                duelScoreTokens.add(new DuelScoreToken(
                        event.side(),
                        event.x(),
                        event.y(),
                        event.scoreDelta(),
                        fruitAccent(event.level())
                ));
            }
            float pan = MathUtils.clamp(
                    (event.x() / FruitRules.BOARD_WIDTH - 0.5f) * 1.1f,
                    -0.72f,
                    0.72f
            );
            float pitch = MathUtils.clamp(
                    0.94f + event.level() * 0.018f,
                    0.90f,
                    1.24f
            );
            float sceneVolume = foreground ? 1f : 0.32f;
            playSound(
                    mergePopSound,
                    0.34f * sceneVolume,
                    pitch,
                    pan
            );
            if (event.level() >= 5) {
                playSound(
                        mergeSoftSound,
                        0.15f * sceneVolume,
                        pitch * 0.97f,
                        pan
                );
            }
            if (event.side() == DuelMatch.Side.PLAYER) {
                vibrateIf(
                        settings.vibrateOnMerge(),
                        event.level() >= 8 ? 36 : 18
                );
            }
            scorePulse = 1f;
        }
        if (aiSceneChanged) {
            invalidateDuelDecision();
        }
    }

    private void updateDuelAi(float realDelta) {
        if (duelMatch == null
                || !duelMatch.roundOpen()
                || duelMatch.aiLane().submittedThisRound()
                || duelMatch.outcome()
                != DuelMatch.Outcome.IN_PROGRESS) {
            return;
        }
        float remaining = duelMatch.roundRemainingSeconds();
        if (duelAiMotionPlan != null) {
            MotionSample sample = duelAiMotionPlan.update(realDelta);
            duelMatch.setAiPreviewX(sample.x);
            if (sample.finished) {
                float x = FruitRules.actionDropX(
                        duelAiMotionPlan.decision.actionIndex,
                        duelMatch.currentLevel()
                );
                duelAiMotionPlan = null;
                dropDuelAiAt(x);
            }
            return;
        }
        if (remaining <= 0.65f) {
            AiDecision fallback = fallbackDecision(
                    createDuelSnapshot(duelMatch.aiLane())
            );
            invalidateDuelDecision();
            dropDuelAiAt(FruitRules.actionDropX(
                    fallback.actionIndex,
                    duelMatch.currentLevel()
            ));
            return;
        }
        if (duelAiRequestInFlight) {
            return;
        }

        boolean stable = duelMatch.aiLane().physics().isStable();
        if (!stable && remaining > 1.35f) {
            return;
        }
        if (!stable || !aiService.isAiReady()) {
            AiDecision fallback = fallbackDecision(
                    createDuelSnapshot(duelMatch.aiLane())
            );
            if (remaining > 1.25f) {
                duelAiMotionPlan = MotionPlan.create(
                        fallback,
                        duelMatch.aiLane().previewX(),
                        duelMatch.currentLevel(),
                        motionRandom
                );
            } else {
                dropDuelAiAt(FruitRules.actionDropX(
                        fallback.actionIndex,
                        duelMatch.currentLevel()
                ));
            }
            return;
        }
        requestDuelAiDecision();
    }

    private void requestDuelAiDecision() {
        DuelMatch match = duelMatch;
        if (match == null || !match.roundOpen()) {
            return;
        }
        duelAiRequestInFlight = true;
        int requestRound = match.roundIndex();
        long requestGeneration = match.matchGeneration();
        long ticket = ++duelDecisionEpoch;
        int requestedLevel = match.currentLevel();
        GameSnapshot snapshot = createDuelSnapshot(match.aiLane());
        aiService.requestDecision(snapshot, new AiService.DecisionCallback() {
            @Override
            public void onSuccess(AiDecision decision) {
                deliverDuelDecision(
                        ticket,
                        requestGeneration,
                        requestRound,
                        requestedLevel,
                        decision
                );
            }

            @Override
            public void onFailure(String message) {
                deliverDuelDecision(
                        ticket,
                        requestGeneration,
                        requestRound,
                        requestedLevel,
                        null
                );
            }
        });
    }

    private void deliverDuelDecision(
            long ticket,
            long generation,
            int round,
            int level,
            AiDecision decision) {
        if (disposed || Gdx.app == null) {
            return;
        }
        Gdx.app.postRunnable(() -> {
            DuelMatch match = duelMatch;
            if (disposed
                    || gameMode != GameMode.DUEL
                    || overlayPage != OverlayPage.NONE
                    || match == null
                    || ticket != duelDecisionEpoch
                    || generation != match.matchGeneration()
                    || round != match.roundIndex()
                    || level != match.currentLevel()
                    || !match.roundOpen()
                    || match.aiLane().submittedThisRound()
                    || match.outcome()
                    != DuelMatch.Outcome.IN_PROGRESS) {
                return;
            }
            duelAiRequestInFlight = false;
            AiDecision selected = decision == null
                    ? fallbackDecision(createDuelSnapshot(match.aiLane()))
                    : decision;
            if (match.roundRemainingSeconds() > 1.25f) {
                duelAiMotionPlan = MotionPlan.create(
                        selected,
                        match.aiLane().previewX(),
                        match.currentLevel(),
                        motionRandom
                );
            } else {
                dropDuelAiAt(FruitRules.actionDropX(
                        selected.actionIndex,
                        match.currentLevel()
                ));
            }
        });
    }

    private GameSnapshot createDuelSnapshot(DuelMatch.Lane lane) {
        List<GameSnapshot.FruitSnapshot> fruitSnapshots = new ArrayList<>();
        for (FruitPhysicsWorld.FruitBody fruit : lane.physics().fruits()) {
            fruitSnapshots.add(new GameSnapshot.FruitSnapshot(
                    fruit.id,
                    fruit.level,
                    fruit.displayRadius,
                    fruit.physicsRadius,
                    fruit.x(),
                    fruit.y(),
                    fruit.vx(),
                    fruit.vy(),
                    fruit.angle(),
                    fruit.angularVelocity(),
                    fruit.ageFrames(),
                    fruit.isStable()
            ));
        }
        return new GameSnapshot(
                lane.score(),
                lane.lastScore(),
                lane.stepCount(),
                duelMatch.queueSnapshot().toArray(),
                fruitSnapshots
        );
    }

    private void beginDuelResult() {
        invalidateDuelDecision();
        activeDragPointer = -1;
        duelResultHoldRemaining = settings.resultHoldSeconds();
        duelResultVisible = false;
        DuelMatch.Lane player = duelMatch.playerLane();
        DuelMatch.Lane ai = duelMatch.aiLane();
        boolean bothEliminated = !player.alive() && !ai.alive();
        if (bothEliminated) {
            duelResultReason = "双方同时越线，按得分判定";
        } else if (!player.alive()) {
            duelResultReason = "玩家水果持续越过警戒线";
            duelForeground = DuelMatch.Side.PLAYER;
        } else {
            duelResultReason = "AI水果持续越过警戒线";
            duelForeground = DuelMatch.Side.AI;
        }
        duelPercentile = profileStore.resultPercentile(player.score());
        if (!duelResultRecorded) {
            duelResultRecorded = true;
            profileStore.recordVersusGame(
                    player.score(),
                    player.watermelonCount(),
                    toStoredBattleResult(duelMatch.outcome())
            );
        }
    }

    private GameProfileStore.BattleResult toStoredBattleResult(
            DuelMatch.Outcome outcome) {
        if (outcome == DuelMatch.Outcome.PLAYER_WIN) {
            return GameProfileStore.BattleResult.WIN;
        }
        if (outcome == DuelMatch.Outcome.AI_WIN) {
            return GameProfileStore.BattleResult.LOSS;
        }
        return GameProfileStore.BattleResult.DRAW;
    }

    private void updateDuelScoreTokens(float delta) {
        for (int index = duelScoreTokens.size - 1;
                index >= 0;
                index--) {
            DuelScoreToken token = duelScoreTokens.get(index);
            token.age += delta;
            float popProgress = MathUtils.clamp(
                    token.age / 0.32f,
                    0f,
                    1f
            );
            if (token.age <= 0.50f) {
                float eased = 1f - (float) Math.pow(1f - popProgress, 3f);
                token.x = token.originX;
                token.y = token.originY - 30f * eased;
                token.scale = 0.45f
                        + eased * 0.55f
                        + MathUtils.sin(popProgress * MathUtils.PI) * 0.16f;
                token.alpha = MathUtils.clamp(popProgress * 5f, 0f, 1f);
                continue;
            }
            float flight = MathUtils.clamp(
                    (token.age - 0.50f) / 0.50f,
                    0f,
                    1f
            );
            float accelerated = flight * flight;
            float targetX = token.side == DuelMatch.Side.PLAYER
                    ? 84f : 206f;
            token.x = MathUtils.lerp(
                    token.originX,
                    targetX,
                    accelerated
            );
            token.y = MathUtils.lerp(
                    token.originY - 30f,
                    SCORE_TARGET_Y,
                    accelerated
            ) - MathUtils.sin(accelerated * MathUtils.PI) * 34f;
            token.scale = 1f - accelerated * 0.55f;
            token.alpha = 1f - Math.max(
                    0f,
                    accelerated - 0.82f
            ) / 0.18f;
            if (flight < 1f) {
                continue;
            }
            duelScoreTokens.removeIndex(index);
            playSound(
                    scoreCollectSound,
                    token.side == DuelMatch.Side.PLAYER ? 0.18f : 0.08f,
                    1.03f,
                    token.side == DuelMatch.Side.PLAYER ? -0.55f : -0.25f
            );
            if (token.side == DuelMatch.Side.PLAYER) {
                vibrateIf(settings.vibrateOnScoreCollect(), 22);
            }
        }
    }

    /**
     * 只供 Windows 共享渲染器验收合成表现，不由 Android 入口调用。
     *
     * <p>它不创建或移动 Box2D 水果，只排入三次带间隔的视觉合成事件，便于自动
     * 截图检查爆浆、立体分值、吸附和计分板反馈。</p>
     */
    public void startPresentationShowcase() {
        if (physics == null || disposed) {
            return;
        }
        particles.clear();
        rings.clear();
        mergeBursts.clear();
        mergeCues.clear();
        scoreTokens.clear();
        scoreRollQueue.clear();
        displayedScore = 0;
        displayedBestScore = 0;
        score = 0;
        lastScore = 0;
        dropCooldown = 60f;

        ScoreSequence sequence = new ScoreSequence(
                ++nextScoreSequenceId,
                score
        );
        activeScoreSequence = sequence;
        int[] levels = {3, 5, 8};
        int[] values = {3, 5, 8};
        float[] xs = {182f, 354f, 268f};
        float[] ys = {638f, 548f, 444f};
        for (int index = 0; index < levels.length; index++) {
            lastScore = score;
            score += values[index];
            sequence.scoreTarget = score;
            sequence.hasMerges = true;
            sequence.pendingCues += 1;
            sequence.maxCascadeDepth = Math.max(
                    sequence.maxCascadeDepth,
                    index + 1
            );
            mergeCues.add(new MergeCue(
                    xs[index],
                    ys[index],
                    levels[index],
                    values[index],
                    index + 1,
                    index * 0.16f,
                    sequence
            ));
        }
        bestScore = Math.max(bestScore, score);
    }

    /**
     * 只供 Windows 共享渲染器打开设置、历史或对战画面做自动截图。
     */
    public void startScreenShowcase(String screen) {
        if (screen == null || physics == null || disposed) {
            return;
        }
        String normalized = screen.trim().toLowerCase(
                java.util.Locale.ROOT
        );
        if ("settings".equals(normalized)) {
            overlayPage = OverlayPage.SETTINGS;
            return;
        }
        if ("history".equals(normalized)) {
            overlayPage = OverlayPage.HISTORY;
            return;
        }
        if (!"duel".equals(normalized)) {
            return;
        }
        gameMode = GameMode.DUEL;
        resetDuelGame();
        for (int round = 0; round < 5; round++) {
            float playerX = 105f + round * 72f;
            float aiX = 455f - round * 70f;
            duelMatch.setPlayerPreviewX(playerX);
            duelMatch.setAiPreviewX(aiX);
            duelMatch.dropPlayer(playerX);
            duelMatch.dropAi(aiX);
            duelMatch.update(0.22f, 1f);
            consumeDuelMergeEvents();
        }
        duelForeground = DuelMatch.Side.PLAYER;
    }

    private void updateGame(float delta) {
        if (!alive) {
            updateEffects(delta);
            return;
        }

        physics.step(delta);
        consumeMergeEvents();
        updateEffects(delta);
        updateDangerTimer(delta);
        if (!alive) {
            return;
        }

        if (dropCooldown > 0f) {
            dropCooldown = Math.max(0f, dropCooldown - delta);
        }
        if (!waiting && dropCooldown <= 0f) {
            fillQueue();
            currentLevel = queue.first();
            waiting = true;
            previewX = FruitRules.clampDropX(previewX, currentLevel);
            previewAnchorX = previewX;
            stableSeconds = 0f;
        }

        boolean stable = physics.isStable();
        if (stable) {
            stableSeconds += delta;
        } else {
            stableSeconds = 0f;
            cancelPendingDecision("等待水果稳定");
        }

        if (!waiting || dropCooldown > 0f) {
            return;
        }
        if (aiEnabled) {
            updateAi(delta);
        } else {
            aiState = AiState.MANUAL;
            aiDetail = "拖动水果，松手投放";
        }
    }

    private void consumeMergeEvents() {
        Array<FruitPhysicsWorld.MergeEvent> events = physics.drainMergeEvents();
        if (events.size > 0) {
            // 合成会改变图拓扑，即使新刚体瞬时速度为零，也必须重新累计稳定窗口，
            // 并让已经发出的异步决策失效。
            stableSeconds = 0f;
            cancelPendingDecision("局面变化");
        }
        for (FruitPhysicsWorld.MergeEvent event : events) {
            ScoreSequence sequence = ensureActiveScoreSequence();
            lastScore = score;
            score += event.scoreDelta;
            bestScore = Math.max(bestScore, score);
            if (event.level == FruitRules.MAX_LEVEL) {
                currentWatermelons += 1;
            }
            sequence.scoreTarget = score;
            sequence.hasMerges = true;
            sequence.secondsSinceLastCue = 0f;
            int cascadeDepth = sequence.registerMerge(
                    event.sourceFruitIdA,
                    event.sourceFruitIdB,
                    event.fruitId
            );
            float delay = sequence.pendingCues * MERGE_CUE_GAP_SECONDS;
            sequence.pendingCues += 1;
            mergeCues.add(new MergeCue(
                    event.x,
                    event.y,
                    event.level,
                    event.scoreDelta,
                    cascadeDepth,
                    delay,
                    sequence
            ));
        }
    }

    private ScoreSequence ensureActiveScoreSequence() {
        if (activeScoreSequence == null || activeScoreSequence.released) {
            activeScoreSequence = new ScoreSequence(
                    ++nextScoreSequenceId,
                    score
            );
        }
        return activeScoreSequence;
    }

    private void updateDangerTimer(float delta) {
        boolean danger = false;
        Array<FruitPhysicsWorld.FruitBody> fruits = physics.fruits();
        // 与桌面入口一致，跳过列表最后一颗新果，避免刚投放/刚合成时瞬间判负。
        for (int index = 0; index < Math.max(0, fruits.size - 1); index++) {
            FruitPhysicsWorld.FruitBody fruit = fruits.get(index);
            // 与桌面/headless 规则一致：圆心持续越过生成线才计入失败倒计时。
            if (fruit.y() < FruitRules.SPAWN_Y) {
                danger = true;
                break;
            }
        }
        dangerSeconds = danger ? dangerSeconds + delta : 0f;
        if (dangerSeconds >= GAME_OVER_SECONDS) {
            alive = false;
            bestScore = Math.max(bestScore, score);
            cancelPendingDecision("游戏结束");
            finishScorePresentationAtGameOver();
            aiState = AiState.GAME_OVER;
            aiDetail = "点击重新开始";
            if (!soloResultRecorded) {
                soloResultRecorded = true;
                soloPercentile = profileStore.resultPercentile(score);
                profileStore.recordSoloGame(score, currentWatermelons);
                settings = profileStore.settings();
            }
        }
    }

    private void updateAi(float delta) {
        if (motionPlan != null) {
            updateMotionPlan(delta);
            return;
        }
        if (aiRequestInFlight) {
            aiState = AiState.THINKING;
            return;
        }
        if (!isScorePresentationReady()) {
            aiState = AiState.OBSERVING;
            aiDetail = "等待连锁结算";
            return;
        }
        if (stableSeconds < FruitRules.STABLE_WINDOW_SECONDS) {
            aiState = AiState.OBSERVING;
            aiDetail = "等待水果稳定";
            // 颤动围绕固定锚点采样，不能逐帧反写并累加成随机游走。
            previewX = FruitRules.clampDropX(
                    previewAnchorX + tremor(0.24f),
                    currentLevel
            );
            return;
        }
        if (aiService.isAiReady()) {
            aiLoadingSeconds = 0f;
        } else if (isAiRuntimeLoading()
                && aiLoadingSeconds < AI_LOADING_FALLBACK_SECONDS) {
            // ONNX/Python 在部分手机上首次解压会慢数秒。这一阶段不能把“尚未就绪”
            // 当成模型故障，否则首局前几手会被启发式策略悄悄接管。
            aiLoadingSeconds += delta;
            aiState = AiState.THINKING;
            aiDetail = "正在加载模型";
            previewX = FruitRules.clampDropX(
                    previewAnchorX + tremor(0.16f),
                    currentLevel
            );
            return;
        }
        requestAiDecision();
    }

    private void requestAiDecision() {
        aiRequestInFlight = true;
        aiState = AiState.THINKING;
        aiDetail = aiService.isAiReady() ? "正在观察局面" : "安全策略";
        long ticket = ++decisionEpoch;
        GameSnapshot snapshot = createSnapshot();

        if (!aiService.isAiReady()) {
            // 构图或模型初始化失败时仍保持游戏可玩；UI 会明确标注 safe fallback。
            startMotionPlan(fallbackDecision(snapshot));
            aiRequestInFlight = false;
            return;
        }

        aiService.requestDecision(snapshot, new AiService.DecisionCallback() {
            @Override
            public void onSuccess(AiDecision decision) {
                if (disposed || Gdx.app == null) {
                    return;
                }
                Gdx.app.postRunnable(() -> {
                    if (disposed
                            || ticket != decisionEpoch
                            || !alive
                            || !waiting
                            || !aiEnabled) {
                        return;
                    }
                    decisionEpoch += 1;
                    aiRequestInFlight = false;
                    if (decision == null) {
                        aiDetail = "安全策略：模型无决定";
                        startMotionPlan(fallbackDecision(createSnapshot()));
                    } else {
                        startMotionPlan(decision);
                    }
                });
            }

            @Override
            public void onFailure(String message) {
                if (disposed || Gdx.app == null) {
                    return;
                }
                Gdx.app.postRunnable(() -> {
                    if (disposed
                            || ticket != decisionEpoch
                            || !alive
                            || !waiting
                            || !aiEnabled) {
                        return;
                    }
                    decisionEpoch += 1;
                    aiRequestInFlight = false;
                    aiDetail = "安全策略：" + sanitizeStatus(message);
                    startMotionPlan(fallbackDecision(createSnapshot()));
                });
            }
        });
    }

    private GameSnapshot createSnapshot() {
        List<GameSnapshot.FruitSnapshot> fruitSnapshots = new ArrayList<>();
        for (FruitPhysicsWorld.FruitBody fruit : physics.fruits()) {
            fruitSnapshots.add(new GameSnapshot.FruitSnapshot(
                    fruit.id,
                    fruit.level,
                    fruit.displayRadius,
                    fruit.physicsRadius,
                    fruit.x(),
                    fruit.y(),
                    fruit.vx(),
                    fruit.vy(),
                    fruit.angle(),
                    fruit.angularVelocity(),
                    fruit.ageFrames(),
                    fruit.isStable()
            ));
        }
        return new GameSnapshot(
                score,
                lastScore,
                stepCount,
                queue.toArray(),
                fruitSnapshots
        );
    }

    private AiDecision fallbackDecision(GameSnapshot snapshot) {
        float[] utilities = new float[FruitRules.ACTION_COUNT];
        /*
         * 对战与经典模式共用启发式，但两者的当前水果不一定相同。局面快照的 q0
         * 才是这次决策的真实水果，不能读取经典模式遗留的 currentLevel。
         */
        int snapshotLevel = snapshot.queue.length > 0
                ? snapshot.queue[0]
                : currentLevel;
        float currentRadius = FruitRules.droppedPhysicsRadius(snapshotLevel);
        for (int action = 0; action < FruitRules.ACTION_COUNT; action++) {
            float x = FruitRules.actionDropX(action, snapshotLevel);
            float landingY = FruitRules.FLOOR_Y - currentRadius;
            int firstLevel = -1;
            for (GameSnapshot.FruitSnapshot fruit : snapshot.fruits) {
                float horizontal = Math.abs(x - fruit.x);
                float sumRadius = currentRadius + fruit.physicsRadius;
                if (horizontal >= sumRadius) {
                    continue;
                }
                float vertical = (float) Math.sqrt(
                        Math.max(0f, sumRadius * sumRadius - horizontal * horizontal)
                );
                float candidateY = fruit.y - vertical;
                if (candidateY < landingY) {
                    landingY = candidateY;
                    firstLevel = fruit.level;
                }
            }
            float depthUtility = landingY / FruitRules.BOARD_HEIGHT;
            float mergeUtility = firstLevel == snapshotLevel ? 0.8f : 0f;
            float edgePreference = Math.abs(x - FruitRules.BOARD_WIDTH / 2f)
                    / FruitRules.BOARD_WIDTH * 0.04f;
            utilities[action] = depthUtility + mergeUtility + edgePreference;
        }

        Integer[] order = new Integer[FruitRules.ACTION_COUNT];
        for (int index = 0; index < order.length; index++) {
            order[index] = index;
        }
        java.util.Arrays.sort(
                order,
                Comparator.comparingDouble((Integer index) -> utilities[index])
                        .reversed()
        );
        return new AiDecision(
                order[0],
                order[1],
                utilities[order[0]],
                utilities[order[1]],
                utilities,
                "safe-fallback"
        );
    }

    private void startMotionPlan(AiDecision decision) {
        motionPlan = MotionPlan.create(
                decision,
                previewX,
                currentLevel,
                motionRandom
        );
        aiState = AiState.THINKING;
        aiDetail = "正在考虑位置 " + (decision.actionIndex + 1) + "/21";
    }

    private void updateMotionPlan(float delta) {
        if (motionPlan == null) {
            return;
        }
        MotionSample sample = motionPlan.update(delta);
        previewX = FruitRules.clampDropX(sample.x, currentLevel);
        aiState = sample.state;
        aiDetail = sample.detail;
        if (!sample.finished) {
            return;
        }

        // 颤动只影响观察轨迹，真实投放必须回到模型选中的规范动作列。
        previewX = FruitRules.actionDropX(
                motionPlan.decision.actionIndex,
                currentLevel
        );
        motionPlan = null;
        dropCurrent(previewX);
    }

    private void cancelPendingDecision(String detail) {
        if (motionPlan == null && !aiRequestInFlight) {
            return;
        }
        /*
         * 取消只停止“继续向目标列移动”，不能把悬浮水果瞬移回旧锚点。
         * 后续观察抖动应当从玩家刚刚看到的位置自然接续。
         */
        if (waiting) {
            previewAnchorX = FruitRules.clampDropX(
                    previewX,
                    currentLevel
            );
        }
        decisionEpoch += 1;
        motionPlan = null;
        aiRequestInFlight = false;
        aiState = aiEnabled ? AiState.OBSERVING : AiState.MANUAL;
        aiDetail = detail;
    }

    private void dropCurrent(float x) {
        if (!canDropCurrent()) {
            return;
        }
        x = FruitRules.clampDropX(x, currentLevel);
        /*
         * 表现序列在第一笔真实合成发生时懒创建。手动玩家可以在上一颗仍运动时
         * 连续投放，因此这里绝不能覆盖尚未释放的合成序列；连续运动窗口内产生的
         * 合成会自然聚合为同一组浮分，稳定后再统一吸入 HUD。
         */
        float y = previewY(currentLevel);
        physics.addDroppedFruit(currentLevel, x, y);
        spawnDropEffect(x, y, currentLevel);
        vibrateIf(settings.vibrateOnDrop(), 10);

        if (queue.size > 0) {
            queue.removeIndex(0);
        }
        fillQueue();
        stepCount += 1;
        dropCooldown = aiEnabled
                ? AI_DROP_COOLDOWN_SECONDS
                : MANUAL_DROP_COOLDOWN_SECONDS;
        if (aiEnabled) {
            waiting = false;
        } else {
            /*
             * 人工模式立即展示并允许拖动下一颗；极短 cooldown 只拦截误触投放，
             * 不再把“能否预摆位置”与上一颗水果的稳定状态绑在一起。
             */
            currentLevel = queue.first();
            waiting = true;
            previewX = FruitRules.clampDropX(previewX, currentLevel);
        }
        stableSeconds = 0f;
        decisionEpoch += 1;
        aiRequestInFlight = false;
        motionPlan = null;
        aiState = aiEnabled ? AiState.OBSERVING : AiState.MANUAL;
        previewAnchorX = previewX;
    }

    private boolean canDropCurrent() {
        return aiEnabled ? canAiDropCurrent() : canManualDropCurrent();
    }

    private boolean isBaseDropReady() {
        return alive && waiting && dropCooldown <= 0f;
    }

    private boolean isScorePresentationReady() {
        return activeScoreSequence == null
                || !activeScoreSequence.hasMerges
                || activeScoreSequence.released;
    }

    private boolean canManualDropCurrent() {
        return !aiEnabled && isBaseDropReady();
    }

    private boolean canManualDragCurrent() {
        return !aiEnabled && alive && waiting;
    }

    private boolean canAiDropCurrent() {
        return aiEnabled
                && isBaseDropReady()
                && physics.isStable()
                && stableSeconds >= FruitRules.STABLE_WINDOW_SECONDS
                && isScorePresentationReady();
    }

    private float previewY(int level) {
        return FruitRules.SPAWN_Y
                - FruitRules.displayRadius(level)
                - PREVIEW_GAP;
    }

    private float tremor(float amplitude) {
        // 低频、有界、连续的相关偏移，模拟悬停手指而不是电子故障噪声。
        return MathUtils.sin(elapsedSeconds * 5.1f) * amplitude
                + MathUtils.sin(elapsedSeconds * 9.3f + 0.7f)
                * amplitude * 0.28f;
    }

    private void updateEffects(float delta) {
        updateMergeCues(delta);
        for (int index = particles.size - 1; index >= 0; index--) {
            Particle particle = particles.get(index);
            particle.life -= delta;
            particle.x += particle.vx * delta;
            particle.y += particle.vy * delta;
            float damping = Math.max(0f, 1f - particle.drag * delta);
            particle.vx *= damping;
            particle.vy *= damping;
            particle.vy += particle.gravity * delta;
            if (particle.life <= 0f) {
                particles.removeIndex(index);
            }
        }
        for (int index = rings.size - 1; index >= 0; index--) {
            Ring ring = rings.get(index);
            ring.life -= delta;
            ring.radius += ring.speed * delta;
            if (ring.life <= 0f) {
                rings.removeIndex(index);
            }
        }
        for (int index = mergeBursts.size - 1; index >= 0; index--) {
            MergeBurst burst = mergeBursts.get(index);
            burst.life -= delta;
            if (burst.life <= 0f) {
                mergeBursts.removeIndex(index);
            }
        }
        updateScoreTokens(delta);
        updateScoreRoll(delta);

        ScoreSequence sequence = activeScoreSequence;
        if (sequence != null && sequence.hasMerges && !sequence.released) {
            if (sequence.pendingCues == 0) {
                sequence.secondsSinceLastCue += delta;
            } else {
                sequence.secondsSinceLastCue = 0f;
            }
            if (sequence.pendingCues == 0
                    && sequence.secondsSinceLastCue
                    >= MERGE_PRESENTATION_SETTLE_SECONDS
                    && (sequence.forceRelease
                    || physics.isStable()
                    || sequence.secondsSinceLastCue
                    >= MERGE_PRESENTATION_MAX_HOLD_SECONDS)) {
                releaseScoreSequence(sequence);
            }
        }
        if (!alive
                && mergeCues.size == 0
                && scoreTokens.size == 0
                && scoreRollQueue.size == 0
                && rollingScoreSequence == null) {
            // 结束画面的分数最终必须与逻辑分数收敛，不能留下半截滚动动画。
            displayedScore = score;
            displayedBestScore = Math.max(
                    displayedBestScore,
                    displayedScore
            );
        }
    }

    private void updateMergeCues(float delta) {
        for (int index = 0; index < mergeCues.size; ) {
            MergeCue cue = mergeCues.get(index);
            cue.delay -= delta;
            if (cue.delay > 0f) {
                index += 1;
                continue;
            }
            mergeCues.removeIndex(index);
            fireMergeCue(cue);
        }
    }

    private void fireMergeCue(MergeCue cue) {
        ScoreSequence sequence = cue.sequence;
        sequence.pendingCues = Math.max(0, sequence.pendingCues - 1);
        sequence.secondsSinceLastCue = 0f;
        Color color = fruitAccent(cue.level);
        sequence.lastColor.set(color);
        spawnMergeEffect(
                cue.x,
                cue.y,
                cue.level,
                cue.cascadeDepth
        );
        scoreTokens.add(new ScoreToken(
                cue.x,
                cue.y,
                cue.scoreDelta,
                color,
                sequence
        ));
        sequence.pendingTokens += 1;

        float pan = MathUtils.clamp(
                (cue.x / FruitRules.BOARD_WIDTH - 0.5f) * 1.1f,
                -0.72f,
                0.72f
        );
        float pitch = MathUtils.clamp(
                0.94f
                        + cue.level * 0.018f
                        + Math.min(3, cue.cascadeDepth - 1) * 0.045f
                        + signedEffectRandom(0.025f),
                0.90f,
                1.30f
        );
        playSound(mergePopSound, 0.36f, pitch, pan);
        float softVolume = cue.level >= 5 || cue.cascadeDepth > 1
                ? 0.20f
                : 0.10f;
        playSound(mergeSoftSound, softVolume, pitch * 0.97f, pan);
        vibrateIf(
                settings.vibrateOnMerge(),
                cue.level >= 8 ? 36 : 18
        );
    }

    private void finishScorePresentationAtGameOver() {
        /*
         * 游戏结束后 Box2D 会停止推进，因此不能再等待 physics.isStable()。
         * 只解除“必须物理稳定”的门槛，cue 仍由 updateMergeCues 按原间隔播放，
         * 避免在结束判定的同一帧叠放全部动画、音效和震动。
         */
        if (activeScoreSequence != null) {
            activeScoreSequence.forceRelease = true;
        }
        for (MergeCue cue : mergeCues) {
            cue.sequence.forceRelease = true;
            if (activeScoreSequence == null) {
                activeScoreSequence = cue.sequence;
            }
        }
        if (activeScoreSequence != null
                && !activeScoreSequence.hasMerges) {
            activeScoreSequence = null;
        }
    }

    private void releaseScoreSequence(ScoreSequence sequence) {
        sequence.released = true;
        for (ScoreToken token : scoreTokens) {
            if (token.sequence != sequence || token.phase == TokenPhase.FLY) {
                continue;
            }
            token.phase = TokenPhase.FLY;
            token.flightAge = 0f;
            token.flightStartX = token.x;
            token.flightStartY = token.y;
            token.curveOffset = signedEffectRandom(34f);
        }
        if (sequence.pendingTokens == 0) {
            enqueueScoreRoll(sequence);
        }
        if (activeScoreSequence == sequence) {
            activeScoreSequence = null;
        }
    }

    private void updateScoreTokens(float delta) {
        for (int index = scoreTokens.size - 1; index >= 0; index--) {
            ScoreToken token = scoreTokens.get(index);
            if (token.phase == TokenPhase.POP) {
                token.age += delta;
                float progress = MathUtils.clamp(
                        token.age / SCORE_TOKEN_POP_SECONDS,
                        0f,
                        1f
                );
                float eased = 1f - (float) Math.pow(1f - progress, 3f);
                token.x = token.originX;
                token.y = token.originY - 31f * eased;
                token.scale = 0.40f
                        + 0.60f * eased
                        + MathUtils.sin(progress * MathUtils.PI) * 0.18f;
                token.alpha = MathUtils.clamp(progress * 5f, 0f, 1f);
                if (progress >= 1f) {
                    token.phase = TokenPhase.HOLD;
                    token.age = 0f;
                    token.scale = 1f;
                }
                continue;
            }
            if (token.phase == TokenPhase.HOLD) {
                token.age += delta;
                token.x = token.originX;
                token.y = token.originY - 31f
                        + MathUtils.sin(token.age * 4.2f + token.sequence.id)
                        * 1.4f;
                token.scale = 1f;
                token.alpha = 1f;
                continue;
            }

            token.flightAge += delta;
            float progress = MathUtils.clamp(
                    token.flightAge / SCORE_TOKEN_FLIGHT_SECONDS,
                    0f,
                    1f
            );
            // p² 让令牌先悬一下、随后明显加速吸入计分板。
            float accelerated = progress * progress;
            float arc = MathUtils.sin(accelerated * MathUtils.PI);
            token.x = MathUtils.lerp(
                    token.flightStartX,
                    SCORE_TARGET_X,
                    accelerated
            ) + arc * token.curveOffset;
            token.y = MathUtils.lerp(
                    token.flightStartY,
                    SCORE_TARGET_Y,
                    accelerated
            ) - arc * 42f;
            token.scale = 1f - accelerated * 0.58f;
            token.alpha = MathUtils.clamp(
                    1f - Math.max(0f, accelerated - 0.82f) / 0.18f,
                    0f,
                    1f
            );
            token.trailClock -= delta;
            if (token.trailClock <= 0f) {
                token.trailClock = 0.035f;
                particles.add(new Particle(
                        token.x,
                        token.y,
                        signedEffectRandom(12f),
                        signedEffectRandom(12f),
                        0.20f,
                        token.color,
                        2.4f,
                        0f,
                        4f
                ));
            }
            if (progress < 1f) {
                continue;
            }

            scoreTokens.removeIndex(index);
            token.sequence.pendingTokens = Math.max(
                    0,
                    token.sequence.pendingTokens - 1
            );
            if (token.sequence.released
                    && token.sequence.pendingTokens == 0) {
                enqueueScoreRoll(token.sequence);
            }
        }
    }

    private void enqueueScoreRoll(ScoreSequence sequence) {
        if (sequence.rollQueued) {
            return;
        }
        sequence.rollQueued = true;
        if (rollingScoreSequence != null) {
            mergeScoreRollPresentation(rollingScoreSequence, sequence);
            int mergedTarget = Math.max(
                    scoreRollTarget,
                    sequence.scoreTarget
            );
            if (mergedTarget > scoreRollTarget) {
                // 从当前可见值重新起一段短滚动，始终追赶最新累计目标。
                scoreRollStart = displayedScore;
                scoreRollTarget = mergedTarget;
                scoreRollElapsed = 0f;
                scoreRollDuration = scoreRollDurationFor(
                        scoreRollTarget - scoreRollStart
                );
            }
            spawnScoreImpact(sequence);
            return;
        }
        if (scoreRollQueue.size > 0) {
            // 尚未开始的多个序列只保留一个累计目标，防止 HUD 队列越滚越慢。
            mergeScoreRollPresentation(scoreRollQueue.first(), sequence);
            return;
        }
        scoreRollQueue.add(sequence);
    }

    private void updateScoreRoll(float delta) {
        scorePulse = Math.max(0f, scorePulse - delta * 1.7f);
        if (rollingScoreSequence == null && scoreRollQueue.size > 0) {
            rollingScoreSequence = scoreRollQueue.removeIndex(0);
            scoreRollStart = displayedScore;
            scoreRollTarget = Math.max(
                    displayedScore,
                    rollingScoreSequence.scoreTarget
            );
            scoreRollDuration = scoreRollDurationFor(
                    scoreRollTarget - scoreRollStart
            );
            scoreRollElapsed = 0f;
            spawnScoreImpact(rollingScoreSequence);
        }
        if (rollingScoreSequence == null) {
            return;
        }

        scoreRollElapsed += delta;
        float progress = MathUtils.clamp(
                scoreRollElapsed / scoreRollDuration,
                0f,
                1f
        );
        float eased = 1f - (float) Math.pow(1f - progress, 3.2f);
        displayedScore = Math.min(
                scoreRollTarget,
                scoreRollStart
                        + Math.round((scoreRollTarget - scoreRollStart) * eased)
        );
        displayedBestScore = Math.max(displayedBestScore, displayedScore);
        if (progress >= 1f) {
            displayedScore = scoreRollTarget;
            displayedBestScore = Math.max(
                    displayedBestScore,
                    displayedScore
            );
            rollingScoreSequence = null;
        }
    }

    private float scoreRollDurationFor(int scoreDelta) {
        int safeDelta = Math.max(1, scoreDelta);
        return MathUtils.clamp(
                0.34f
                        + (float) Math.log10(safeDelta + 1f) * 0.16f,
                0.38f,
                0.82f
        );
    }

    private void mergeScoreRollPresentation(
            ScoreSequence target,
            ScoreSequence incoming) {
        if (incoming.scoreTarget >= target.scoreTarget) {
            target.lastColor.set(incoming.lastColor);
        }
        target.scoreTarget = Math.max(
                target.scoreTarget,
                incoming.scoreTarget
        );
        target.maxCascadeDepth = Math.max(
                target.maxCascadeDepth,
                incoming.maxCascadeDepth
        );
    }

    private void spawnScoreImpact(ScoreSequence sequence) {
        scorePulse = 1f;
        Color color = sequence.lastColor;
        rings.add(new Ring(
                SCORE_TARGET_X,
                SCORE_TARGET_Y,
                13f,
                150f,
                0.42f,
                color,
                3.6f
        ));
        for (int index = 0; index < 15; index++) {
            float angle = effectRandom.nextFloat() * MathUtils.PI2;
            float speed = 50f + effectRandom.nextFloat() * 105f;
            particles.add(new Particle(
                    SCORE_TARGET_X,
                    SCORE_TARGET_Y,
                    MathUtils.cos(angle) * speed,
                    MathUtils.sin(angle) * speed,
                    0.30f + effectRandom.nextFloat() * 0.22f,
                    color,
                    2.2f + effectRandom.nextFloat() * 2.8f,
                    80f,
                    1.2f
            ));
        }
        float pitch = MathUtils.clamp(
                1.02f + Math.min(5, sequence.maxCascadeDepth) * 0.045f,
                1.02f,
                1.28f
        );
        playSound(scoreCollectSound, 0.24f, pitch, -0.55f);
        vibrateIf(settings.vibrateOnScoreCollect(), 22);
    }

    private void spawnDropEffect(float x, float y, int level) {
        Color color = fruitAccent(level);
        rings.add(new Ring(
                x,
                y,
                FruitRules.displayRadius(level) * 0.4f,
                130f,
                0.28f,
                color,
                2.2f
        ));
        for (int index = 0; index < 7; index++) {
            float angle = effectRandom.nextFloat() * MathUtils.PI2;
            float speed = 45f + effectRandom.nextFloat() * 80f;
            particles.add(new Particle(
                    x,
                    y,
                    MathUtils.cos(angle) * speed,
                    MathUtils.sin(angle) * speed,
                    0.25f + effectRandom.nextFloat() * 0.18f,
                    color,
                    2.1f + effectRandom.nextFloat() * 2.4f,
                    150f,
                    0.8f
            ));
        }
    }

    private void spawnMergeEffect(
            float x,
            float y,
            int level,
            int cascadeDepth) {
        Color color = fruitAccent(level);
        float intensity = 1f
                + Math.min(3, Math.max(0, cascadeDepth - 1)) * 0.10f;
        mergeBursts.add(new MergeBurst(
                x,
                y,
                18f + level * 1.6f,
                0.48f,
                color,
                effectRandom.nextFloat() * MathUtils.PI2,
                Math.min(16, 9 + level / 2)
        ));
        rings.add(new Ring(
                x,
                y,
                10f,
                285f * intensity,
                0.54f,
                color,
                4.2f
        ));
        rings.add(new Ring(
                x,
                y,
                6f,
                175f * intensity,
                0.34f,
                Color.WHITE,
                2.7f
        ));
        int count = Math.min(
                44,
                20 + level * 2 + Math.min(cascadeDepth - 1, 3) * 3
        );
        for (int index = 0; index < count; index++) {
            float angle = effectRandom.nextFloat() * MathUtils.PI2;
            float speed = 95f + effectRandom.nextFloat()
                    * (135f + level * 11f) * intensity;
            particles.add(new Particle(
                    x,
                    y,
                    MathUtils.cos(angle) * speed,
                    MathUtils.sin(angle) * speed,
                    0.44f + effectRandom.nextFloat() * 0.34f,
                    color,
                    3f + effectRandom.nextFloat() * 6f,
                    135f,
                    0.55f
            ));
        }
    }

    private float signedEffectRandom(float amplitude) {
        return (effectRandom.nextFloat() * 2f - 1f) * amplitude;
    }

    private void playSound(
            Sound sound,
            float baseVolume,
            float pitch,
            float pan) {
        float volume = baseVolume * settings.soundVolume();
        if (sound != null && volume > 0.001f) {
            sound.play(
                    MathUtils.clamp(volume, 0f, 1f),
                    pitch,
                    pan
            );
        }
    }

    private void vibrateIf(boolean enabled, int milliseconds) {
        if (enabled) {
            aiService.vibrate(milliseconds);
        }
    }

    private Color fruitAccent(int level) {
        switch (MathUtils.clamp(level, FruitRules.MIN_LEVEL, FruitRules.MAX_LEVEL)) {
            case 1: return new Color(0.67f, 0.20f, 0.73f, 1f);
            case 2: return new Color(1.00f, 0.22f, 0.30f, 1f);
            case 3: return new Color(1.00f, 0.48f, 0.10f, 1f);
            case 4: return new Color(1.00f, 0.84f, 0.12f, 1f);
            case 5: return new Color(0.42f, 0.88f, 0.18f, 1f);
            case 6: return new Color(1.00f, 0.35f, 0.48f, 1f);
            case 7: return new Color(1.00f, 0.66f, 0.31f, 1f);
            case 8: return new Color(1.00f, 0.76f, 0.16f, 1f);
            case 9: return new Color(0.88f, 0.81f, 0.64f, 1f);
            case 10: return new Color(1.00f, 0.23f, 0.42f, 1f);
            default: return new Color(0.26f, 0.81f, 0.22f, 1f);
        }
    }

    private void drawGame() {
        ScreenUtils.clear(BACKGROUND_TOP);
        camera.update();
        Gdx.gl.glEnable(GL20.GL_BLEND);
        Gdx.gl.glBlendFunc(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA);

        shapes.setProjectionMatrix(camera.combined);
        shapes.begin(ShapeRenderer.ShapeType.Filled);
        drawBackground();
        drawPanels();
        shapes.end();

        /*
         * 先画真实水果，再把爆浆与冲击环画到水果上方；旧版特效位于水果背后，
         * 即使提高透明度也会被新合成的水果遮住。
         */
        batch.setProjectionMatrix(camera.combined);
        batch.begin();
        drawFruitBodies();
        drawPreviewAndQueue();
        batch.end();

        shapes.begin(ShapeRenderer.ShapeType.Filled);
        drawEffects();
        shapes.end();

        batch.begin();
        drawScoreTokens();
        drawDuelScoreTokens();
        drawText();
        batch.end();

        if (gameMode == GameMode.CLASSIC && !alive) {
            drawGameOverOverlay();
        } else if (gameMode == GameMode.DUEL) {
            drawDuelResultLayer();
        }
        if (overlayPage != OverlayPage.NONE) {
            drawOverlayPage();
        }
    }

    private void drawBackground() {
        // 继续用四角颜色生成无贴图渐变，避免为了整屏背景增加 APK 体积。
        shapes.rect(
                0f,
                0f,
                FruitRules.BOARD_WIDTH,
                FruitRules.BOARD_HEIGHT,
                BACKGROUND_BOTTOM,
                BACKGROUND_BOTTOM,
                BACKGROUND_TOP,
                BACKGROUND_TOP
        );

        // 圆形暖光和低透明叶片只装饰空白边角，不进入水果纹理与物理层。
        shapes.setColor(ORCHARD_GLOW);
        shapes.circle(-9f, toRenderY(31f), 52f, 40);
        shapes.circle(565f, toRenderY(86f), 64f, 40);
        shapes.circle(524f, toRenderY(1084f), 52f, 40);
        drawLeaf(24f, 34f, 48f, 19f, -27f, LEAF_LIGHT);
        drawLeaf(49f, 18f, 42f, 17f, 18f, LEAF_DARK);
        drawLeaf(528f, 32f, 50f, 20f, 32f, LEAF_LIGHT);
        drawLeaf(548f, 59f, 42f, 17f, 67f, LEAF_DARK);
        drawLeaf(18f, 1084f, 48f, 19f, 35f, LEAF_DARK);
        drawLeaf(540f, 1092f, 52f, 20f, -32f, LEAF_LIGHT);
    }

    /**
     * 用两个三角形拼出轻量叶片。这里采用程序绘制而不是背景位图，后续调整主题色时
     * 只需要改调色板，也不会替换用户要求保留的水果素材。
     */
    private void drawLeaf(
            float centerX,
            float centerTopY,
            float width,
            float height,
            float degrees,
            Color color) {
        float centerY = toRenderY(centerTopY);
        float angle = degrees * MathUtils.degreesToRadians;
        float axisX = MathUtils.cos(angle) * width * 0.5f;
        float axisY = MathUtils.sin(angle) * width * 0.5f;
        float sideX = -MathUtils.sin(angle) * height * 0.5f;
        float sideY = MathUtils.cos(angle) * height * 0.5f;
        float startX = centerX - axisX;
        float startY = centerY - axisY;
        float endX = centerX + axisX;
        float endY = centerY + axisY;

        shapes.setColor(color);
        shapes.triangle(
                startX,
                startY,
                centerX + sideX,
                centerY + sideY,
                endX,
                endY
        );
        shapes.triangle(
                startX,
                startY,
                endX,
                endY,
                centerX - sideX,
                centerY - sideY
        );
    }

    private void drawPanels() {
        /*
         * 顶部 HUD 必须在 y=140 前结束。最大可投水果的预览最上缘约为 y=146，
         * 因而 146～248 被完整留给水果，不再放任何会被它遮挡的状态板。
         */
        roundedRectTop(16f, 16f, 528f, 128f, 20f, CARD_SHADOW);
        roundedRectTop(16f, 12f, 528f, 128f, 20f, PANEL_COLOR);

        if (scorePulse > 0f) {
            float expansion = (1f - scorePulse) * 11f;
            scratchColor.set(
                    SCORE_GLOW.r,
                    SCORE_GLOW.g,
                    SCORE_GLOW.b,
                    scorePulse * 0.42f
            );
            roundedRectTop(
                    25f - expansion,
                    61f - expansion,
                    118f + expansion * 2f,
                    70f + expansion * 2f,
                    15f + expansion,
                    scratchColor
            );
        }

        // 用“外层填色 + 内层填色”形成可靠圆角边框，不再调用 Line arc。
        roundedRectTop(28f, 68f, 112f, 64f, 12f, CARD_SHADOW);
        roundedRectTop(28f, 64f, 112f, 64f, 12f, BOARD_FRAME_SOFT);
        roundedRectTop(
                30f,
                66f,
                108f,
                60f,
                10f,
                gameMode == GameMode.DUEL ? PLAYER_TINT_SOFT : SCORE_CARD
        );

        roundedRectTop(150f, 68f, 112f, 64f, 12f, CARD_SHADOW);
        roundedRectTop(150f, 64f, 112f, 64f, 12f, BOARD_FRAME_SOFT);
        roundedRectTop(
                152f,
                66f,
                108f,
                60f,
                10f,
                gameMode == GameMode.DUEL ? AI_TINT_SOFT : SCORE_CARD
        );
        if (gameMode == GameMode.DUEL) {
            shapes.setColor(CARD_SHADOW);
            shapes.circle(145f, toRenderY(99f), 15f, 28);
            shapes.setColor(TEXT_PRIMARY);
            shapes.circle(145f, toRenderY(96f), 13f, 28);
        }

        roundedRectTop(272f, 68f, 260f, 64f, 12f, CARD_SHADOW);
        roundedRectTop(272f, 64f, 260f, 64f, 12f, BOARD_FRAME_SOFT);
        roundedRectTop(274f, 66f, 256f, 60f, 10f, NEXT_CARD);

        // 标题旁的小叶片与队列槽给顶部区域增加果园识别度，但不替换水果本身。
        drawLeaf(168f, 35f, 22f, 9f, -24f, LEAF_ACCENT);
        drawLeaf(181f, 31f, 19f, 8f, 28f, LEAF_ACCENT);
        shapes.setColor(DANGER.r, DANGER.g, DANGER.b, 0.68f);
        shapes.circle(180f, toRenderY(40f), 3.4f, 16);
        float[] queueSlots = {354f, 408f, 462f};
        for (float slotX : queueSlots) {
            shapes.setColor(CARD_SHADOW);
            shapes.circle(slotX, toRenderY(106f), 20f, 28);
            shapes.setColor(BOARD_FRAME_SOFT);
            shapes.circle(slotX, toRenderY(103f), 20f, 28);
            shapes.setColor(SCORE_CARD);
            shapes.circle(slotX, toRenderY(103f), 17.8f, 28);
        }

        roundedRectTop(
                MODE_BUTTON_LEFT,
                MODE_BUTTON_TOP + 4f,
                MODE_BUTTON_WIDTH,
                MODE_BUTTON_HEIGHT,
                MODE_BUTTON_HEIGHT / 2f,
                CARD_SHADOW
        );
        roundedRectTop(
                MODE_BUTTON_LEFT,
                MODE_BUTTON_TOP,
                MODE_BUTTON_WIDTH,
                MODE_BUTTON_HEIGHT,
                MODE_BUTTON_HEIGHT / 2f,
                gameMode == GameMode.DUEL
                        ? PLAYER_TINT_SOFT
                        : NEXT_CARD
        );

        roundedRectTop(
                AI_TOGGLE_LEFT,
                AI_TOGGLE_TOP + 4f,
                AI_TOGGLE_WIDTH,
                AI_TOGGLE_HEIGHT,
                AI_TOGGLE_HEIGHT / 2f,
                CARD_SHADOW
        );
        roundedRectTop(
                AI_TOGGLE_LEFT,
                AI_TOGGLE_TOP,
                AI_TOGGLE_WIDTH,
                AI_TOGGLE_HEIGHT,
                AI_TOGGLE_HEIGHT / 2f,
                gameMode == GameMode.DUEL
                        ? (duelForegroundSide() == DuelMatch.Side.PLAYER
                        ? PLAYER_TINT_SOFT : AI_TINT_SOFT)
                        : (aiEnabled ? ACCENT_SOFT : SWITCH_OFF)
        );

        drawUtilityButton(
                HISTORY_BUTTON_LEFT,
                false
        );
        drawUtilityButton(
                SETTINGS_BUTTON_LEFT,
                true
        );

        /*
         * 棋盘只保留阴影、外框和内底三层。左右/底部不再额外画 wall rect，
         * 避免旧版在 floor 以下多出 20px 线条。
         */
        roundedRectTop(18f, 254f, 524f, 854f, 18f, CARD_SHADOW);
        roundedRectTop(18f, 248f, 524f, 860f, 18f, BOARD_FRAME);
        roundedRectTop(20f, 250f, 520f, 856f, 16f, BOARD_FRAME_SOFT);
        roundedRectTop(22f, 252f, 516f, 848f, 14f, BOARD_COLOR);
        if (gameMode == GameMode.DUEL) {
            Color tint = duelForeground == DuelMatch.Side.PLAYER
                    ? PLAYER_TINT
                    : AI_TINT;
            scratchColor.set(tint.r, tint.g, tint.b, 0.055f);
            roundedRectTop(
                    24f,
                    254f,
                    512f,
                    844f,
                    12f,
                    scratchColor
            );
        }
        shapes.setColor(
                ORCHARD_GLOW.r,
                ORCHARD_GLOW.g,
                ORCHARD_GLOW.b,
                0.045f
        );
        shapes.circle(88f, toRenderY(1018f), 58f, 36);
        shapes.circle(474f, toRenderY(1028f), 50f, 36);

        float visibleDangerSeconds = gameMode == GameMode.DUEL
                ? duelForegroundLane().dangerSeconds()
                : dangerSeconds;
        float dangerAlpha = visibleDangerSeconds <= 0f
                ? 0.48f
                : 0.55f + MathUtils.sin(elapsedSeconds * 12f) * 0.25f;
        shapes.setColor(DANGER.r, DANGER.g, DANGER.b, dangerAlpha);
        float y = toRenderY(FruitRules.SPAWN_Y + 4f);
        for (float x = 30f; x < 530f; x += 20f) {
            shapes.rect(x, y - 1f, 11f, 2f);
        }

        boolean previewVisible = gameMode == GameMode.DUEL
                ? duelForegroundPreviewVisible()
                : waiting;
        if (previewVisible) {
            boolean assistedPreview = gameMode == GameMode.DUEL
                    ? duelForeground == DuelMatch.Side.AI
                    : aiEnabled;
            int visibleLevel = gameMode == GameMode.DUEL
                    ? duelMatch.currentLevel()
                    : currentLevel;
            float visiblePreviewX = gameMode == GameMode.DUEL
                    ? duelForegroundLane().previewX()
                    : previewX;
            float guideAlpha = assistedPreview ? 0.19f : 0.13f;
            shapes.setColor(ACCENT.r, ACCENT.g, ACCENT.b, guideAlpha);
            float guideTop = previewY(visibleLevel)
                    + FruitRules.displayRadius(visibleLevel);
            float guideBottom = FruitRules.FLOOR_Y;
            for (float screenY = guideTop + 14f;
                    screenY < guideBottom;
                    screenY += 26f) {
                shapes.circle(
                        visiblePreviewX,
                        toRenderY(screenY),
                        1.7f,
                        10
                );
            }

            // 实心半透明外圆在水果之前绘制，水果本身会盖住中心形成干净的光环。
            shapes.setColor(
                    ACCENT.r,
                    ACCENT.g,
                    ACCENT.b,
                    assistedPreview ? 0.22f : 0.14f
            );
            shapes.circle(
                    visiblePreviewX,
                    toRenderY(previewY(visibleLevel)),
                    FruitRules.displayRadius(visibleLevel) + 5f,
                    40
            );
        }

        // 经典模式显示 AI 开关；对战模式用同一控件切换前景场景。
        boolean knobOnRight = gameMode == GameMode.DUEL
                ? duelForegroundSide() == DuelMatch.Side.AI
                : aiEnabled;
        float knobCenterX = knobOnRight
                ? AI_TOGGLE_LEFT + AI_TOGGLE_WIDTH - 21f
                : AI_TOGGLE_LEFT + 21f;
        float knobCenterY = AI_TOGGLE_TOP + AI_TOGGLE_HEIGHT / 2f;
        shapes.setColor(CARD_SHADOW);
        shapes.circle(knobCenterX, toRenderY(knobCenterY + 3f), 15f, 28);
        shapes.setColor(SCORE_CARD);
        shapes.circle(knobCenterX, toRenderY(knobCenterY), 15f, 28);
        Color faceColor = gameMode == GameMode.DUEL
                ? (duelForegroundSide() == DuelMatch.Side.PLAYER
                ? PLAYER_TINT : AI_TINT)
                : (aiEnabled ? ACCENT_DARK : TEXT_MUTED);
        shapes.setColor(faceColor);
        shapes.circle(knobCenterX - 4.2f, toRenderY(knobCenterY - 1f), 1.5f, 10);
        shapes.circle(knobCenterX + 4.2f, toRenderY(knobCenterY - 1f), 1.5f, 10);
        shapes.rect(
                knobCenterX - 3.5f,
                toRenderY(knobCenterY + 5f),
                7f,
                1.5f
        );
    }

    private void drawUtilityButton(float left, boolean gear) {
        roundedRectTop(
                left,
                UTILITY_BUTTON_TOP + 4f,
                UTILITY_BUTTON_SIZE,
                UTILITY_BUTTON_SIZE,
                UTILITY_BUTTON_SIZE / 2f,
                CARD_SHADOW
        );
        roundedRectTop(
                left,
                UTILITY_BUTTON_TOP,
                UTILITY_BUTTON_SIZE,
                UTILITY_BUTTON_SIZE,
                UTILITY_BUTTON_SIZE / 2f,
                SCORE_CARD
        );
        float centerX = left + UTILITY_BUTTON_SIZE / 2f;
        float centerY = UTILITY_BUTTON_TOP + UTILITY_BUTTON_SIZE / 2f;
        if (gear) {
            shapes.setColor(TEXT_MUTED);
            for (int tooth = 0; tooth < 8; tooth++) {
                float angle = tooth * MathUtils.PI2 / 8f;
                shapes.circle(
                        centerX + MathUtils.cos(angle) * 10.5f,
                        toRenderY(centerY + MathUtils.sin(angle) * 10.5f),
                        3.2f,
                        10
                );
            }
            shapes.circle(centerX, toRenderY(centerY), 10f, 24);
            shapes.setColor(SCORE_CARD);
            shapes.circle(centerX, toRenderY(centerY), 4.1f, 18);
        } else {
            shapes.setColor(TEXT_MUTED);
            shapes.circle(centerX, toRenderY(centerY), 11f, 28);
            shapes.setColor(SCORE_CARD);
            shapes.circle(centerX, toRenderY(centerY), 7.4f, 24);
            shapes.setColor(TEXT_MUTED);
            shapes.rect(
                    centerX - 1.2f,
                    toRenderY(centerY + 0.5f),
                    2.4f,
                    7f
            );
            shapes.rect(
                    centerX - 0.5f,
                    toRenderY(centerY + 1.8f),
                    6.5f,
                    2.2f
            );
        }
    }

    private void drawEffects() {
        Gdx.gl.glEnable(GL20.GL_BLEND);
        for (MergeBurst burst : mergeBursts) {
            float remaining = MathUtils.clamp(
                    burst.life / burst.maxLife,
                    0f,
                    1f
            );
            float progress = 1f - remaining;
            float renderY = toRenderY(burst.y);
            float radius = burst.radius * (0.76f + progress * 0.54f);
            float alpha = (float) Math.pow(remaining, 0.72f) * 0.86f;
            shapes.setColor(
                    burst.color.r,
                    burst.color.g,
                    burst.color.b,
                    alpha
            );
            for (int ray = 0; ray < burst.rays; ray++) {
                float angle = burst.rotation
                        + ray * MathUtils.PI2 / burst.rays;
                float halfAngle = 0.11f + (ray % 3) * 0.025f;
                float innerRadius = radius * 0.22f;
                float outerRadius = radius
                        * (1.02f + (ray % 4) * 0.075f);
                shapes.triangle(
                        burst.x + MathUtils.cos(angle) * innerRadius,
                        renderY + MathUtils.sin(angle) * innerRadius,
                        burst.x + MathUtils.cos(angle - halfAngle) * outerRadius,
                        renderY + MathUtils.sin(angle - halfAngle) * outerRadius,
                        burst.x + MathUtils.cos(angle + halfAngle) * outerRadius,
                        renderY + MathUtils.sin(angle + halfAngle) * outerRadius
                );
                shapes.circle(
                        burst.x + MathUtils.cos(angle) * outerRadius,
                        renderY + MathUtils.sin(angle) * outerRadius,
                        3.4f + (ray % 3) * 1.5f,
                        12
                );
            }
            float flashAlpha = MathUtils.clamp(
                    (remaining - 0.58f) / 0.42f,
                    0f,
                    1f
            ) * 0.26f;
            shapes.setColor(
                    burst.color.r,
                    burst.color.g,
                    burst.color.b,
                    alpha * 0.24f
            );
            shapes.circle(
                    burst.x,
                    renderY,
                    radius * (0.53f - progress * 0.10f),
                    32
            );
            shapes.setColor(1f, 1f, 0.91f, flashAlpha);
            shapes.circle(
                    burst.x,
                    renderY,
                    radius * (0.24f + remaining * 0.08f),
                    28
            );
        }
        for (Particle particle : particles) {
            float remaining = MathUtils.clamp(
                    particle.life / particle.maxLife,
                    0f,
                    1f
            );
            float alpha = (float) Math.pow(remaining, 0.68f) * 0.96f;
            shapes.setColor(
                    particle.color.r,
                    particle.color.g,
                    particle.color.b,
                    alpha
            );
            shapes.circle(
                    particle.x,
                    toRenderY(particle.y),
                    particle.radius * (0.62f + remaining * 0.38f),
                    12
            );
        }
        for (Ring ring : rings) {
            float remaining = MathUtils.clamp(
                    ring.life / ring.maxLife,
                    0f,
                    1f
            );
            float alpha = (float) Math.pow(remaining, 0.62f) * 0.82f;
            shapes.setColor(ring.color.r, ring.color.g, ring.color.b, alpha);
            float renderY = toRenderY(ring.y);
            float dotRadius = ring.dotRadius
                    * (0.46f + remaining * 0.54f);
            for (int dot = 0; dot < 30; dot++) {
                float angle = dot * MathUtils.PI2 / 30f;
                shapes.circle(
                        ring.x + MathUtils.cos(angle) * ring.radius,
                        renderY + MathUtils.sin(angle) * ring.radius,
                        dotRadius,
                        8
                );
            }
        }
    }

    private void drawScoreTokens() {
        if (gameMode == GameMode.DUEL) {
            return;
        }
        for (ScoreToken token : scoreTokens) {
            String text = "+" + token.value;
            float scale = POPUP_FONT_SCALE * token.scale;
            popupFont.getData().setScale(scale);

            scratchColor.set(
                    token.color.r * 0.42f,
                    token.color.g * 0.42f,
                    token.color.b * 0.42f,
                    token.alpha * 0.66f
            );
            popupFont.setColor(scratchColor);
            drawTextCentered(
                    popupFont,
                    text,
                    token.x + 2.2f,
                    token.y + 2.8f
            );

            scratchColor.set(
                    MathUtils.lerp(token.color.r, 1f, 0.16f),
                    MathUtils.lerp(token.color.g, 1f, 0.16f),
                    MathUtils.lerp(token.color.b, 1f, 0.16f),
                    token.alpha
            );
            popupFont.setColor(scratchColor);
            drawTextCentered(popupFont, text, token.x, token.y);
        }
        popupFont.getData().setScale(POPUP_FONT_SCALE);
    }

    private void drawDuelScoreTokens() {
        if (gameMode != GameMode.DUEL) {
            return;
        }
        for (DuelScoreToken token : duelScoreTokens) {
            String text = "+" + token.value;
            float scale = POPUP_FONT_SCALE * token.scale;
            popupFont.getData().setScale(scale);
            scratchColor.set(
                    token.color.r * 0.42f,
                    token.color.g * 0.42f,
                    token.color.b * 0.42f,
                    token.alpha * 0.66f
            );
            popupFont.setColor(scratchColor);
            drawTextCentered(
                    popupFont,
                    text,
                    token.x + 2.2f,
                    token.y + 2.8f
            );
            scratchColor.set(
                    MathUtils.lerp(token.color.r, 1f, 0.16f),
                    MathUtils.lerp(token.color.g, 1f, 0.16f),
                    MathUtils.lerp(token.color.b, 1f, 0.16f),
                    token.alpha
            );
            popupFont.setColor(scratchColor);
            drawTextCentered(popupFont, text, token.x, token.y);
        }
        popupFont.getData().setScale(POPUP_FONT_SCALE);
    }

    private void drawTextCentered(
            BitmapFont font,
            String text,
            float centerX,
            float centerY) {
        glyphLayout.setText(font, text);
        font.draw(
                batch,
                glyphLayout,
                centerX - glyphLayout.width * 0.5f,
                toRenderY(centerY - glyphLayout.height * 0.5f)
        );
    }

    private void drawFruitBodies() {
        if (gameMode == GameMode.DUEL) {
            drawDuelFruitBodies();
            return;
        }
        for (FruitPhysicsWorld.FruitBody fruit : physics.fruits()) {
            drawFruit(
                    fruit.level,
                    fruit.x(),
                    fruit.y(),
                    fruit.displayRadius * 2f,
                    1f
            );
        }
    }

    private void drawPreviewAndQueue() {
        if (gameMode == GameMode.DUEL) {
            drawDuelPreviewAndQueue();
            return;
        }
        if (waiting) {
            float alpha = dropCooldown > 0f ? 0.55f : 0.96f;
            drawFruit(
                    currentLevel,
                    previewX,
                    previewY(currentLevel),
                    FruitRules.displayRadius(currentLevel) * 2f,
                    alpha
            );
        }

        // 只调整队列卡内的间距；水果贴图、尺寸和透明度保持原实现。
        float[] queueX = {354f, 408f, 462f};
        for (int index = 1; index < Math.min(queue.size, 4); index++) {
            int level = queue.get(index);
            float size = Math.min(34f, FruitRules.displayRadius(level) * 1.05f);
            drawFruit(level, queueX[index - 1], 103f, size, 0.88f);
        }
    }

    private void drawDuelFruitBodies() {
        DuelMatch.Lane background = duelForeground == DuelMatch.Side.PLAYER
                ? duelMatch.aiLane()
                : duelMatch.playerLane();
        DuelMatch.Lane foreground = duelForegroundLane();
        Color backgroundTint = background.side() == DuelMatch.Side.PLAYER
                ? PLAYER_TINT
                : AI_TINT;
        for (FruitPhysicsWorld.FruitBody fruit
                : background.physics().fruits()) {
            float size = fruit.displayRadius * 2f;
            // 三次轻微错位、低饱和绘制形成无需 shader 的柔和虚影。
            drawFruitTinted(
                    fruit.level,
                    fruit.x() - 1.8f,
                    fruit.y(),
                    size,
                    backgroundTint,
                    0.075f
            );
            drawFruitTinted(
                    fruit.level,
                    fruit.x() + 1.8f,
                    fruit.y(),
                    size,
                    backgroundTint,
                    0.075f
            );
            drawFruitTinted(
                    fruit.level,
                    fruit.x(),
                    fruit.y() + 1.4f,
                    size,
                    backgroundTint,
                    0.10f
            );
        }
        for (FruitPhysicsWorld.FruitBody fruit
                : foreground.physics().fruits()) {
            drawFruit(
                    fruit.level,
                    fruit.x(),
                    fruit.y(),
                    fruit.displayRadius * 2f,
                    1f
            );
        }
    }

    private void drawDuelPreviewAndQueue() {
        DuelMatch.Lane background = duelForeground == DuelMatch.Side.PLAYER
                ? duelMatch.aiLane()
                : duelMatch.playerLane();
        DuelMatch.Lane foreground = duelForegroundLane();
        int level = duelMatch.currentLevel();
        if (duelMatch.roundOpen()
                && !background.submittedThisRound()
                && background.alive()) {
            Color tint = background.side() == DuelMatch.Side.PLAYER
                    ? PLAYER_TINT
                    : AI_TINT;
            drawFruitTinted(
                    level,
                    background.previewX(),
                    previewY(level),
                    FruitRules.displayRadius(level) * 2f,
                    tint,
                    0.25f
            );
        }
        if (duelMatch.roundOpen()
                && !foreground.submittedThisRound()
                && foreground.alive()) {
            drawFruit(
                    level,
                    foreground.previewX(),
                    previewY(level),
                    FruitRules.displayRadius(level) * 2f,
                    0.96f
            );
        }
        float[] queueX = {354f, 408f, 462f};
        for (int index = 1;
                index < FruitRules.QUEUE_LENGTH;
                index++) {
            int queuedLevel = duelMatch.queuedLevel(index);
            float size = Math.min(
                    34f,
                    FruitRules.displayRadius(queuedLevel) * 1.05f
            );
            drawFruit(
                    queuedLevel,
                    queueX[index - 1],
                    103f,
                    size,
                    0.88f
            );
        }
    }

    private void drawFruit(int level, float centerX, float centerY, float size, float alpha) {
        batch.setColor(1f, 1f, 1f, alpha);
        batch.draw(
                fruitTextures[level],
                centerX - size / 2f,
                toRenderY(centerY + size / 2f),
                size,
                size
        );
        batch.setColor(Color.WHITE);
    }

    private void drawFruitTinted(
            int level,
            float centerX,
            float centerY,
            float size,
            Color tint,
            float alpha) {
        batch.setColor(tint.r, tint.g, tint.b, alpha);
        batch.draw(
                fruitTextures[level],
                centerX - size / 2f,
                toRenderY(centerY + size / 2f),
                size,
                size
        );
        batch.setColor(Color.WHITE);
    }

    private void drawText() {
        titleFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                titleFont,
                "合成大西瓜",
                28f,
                16f,
                154f,
                42f,
                Align.left
        );

        smallFont.setColor(DANGER);
        drawTextInBox(
                smallFont,
                gameMode == GameMode.DUEL ? "玩家" : "分数",
                30f,
                68f,
                108f,
                20f,
                Align.center
        );
        smallFont.setColor(
                gameMode == GameMode.DUEL ? AI_TINT : DANGER
        );
        drawTextInBox(
                smallFont,
                gameMode == GameMode.DUEL ? "AI" : "最高",
                152f,
                68f,
                108f,
                20f,
                Align.center
        );
        drawTextInBox(smallFont, "下一颗", 284f, 68f, 62f, 20f, Align.left);

        normalFont.setColor(TEXT_PRIMARY);
        normalFont.getData().setScale(0.38f * (1f + scorePulse * 0.10f));
        drawTextInBox(
                normalFont,
                Integer.toString(
                        gameMode == GameMode.DUEL
                                ? duelPlayerScore()
                                : displayedScore
                ),
                30f,
                88f,
                108f,
                34f,
                Align.center
        );
        normalFont.getData().setScale(0.38f);
        drawTextInBox(
                normalFont,
                Integer.toString(
                        gameMode == GameMode.DUEL
                                ? duelAiScore()
                                : displayedBestScore
                ),
                152f,
                88f,
                108f,
                34f,
                Align.center
        );
        if (gameMode == GameMode.DUEL) {
            smallFont.setColor(Color.WHITE);
            drawTextInBox(
                    smallFont,
                    "VS",
                    132f,
                    84f,
                    26f,
                    24f,
                    Align.center
            );
        }

        smallFont.setColor(
                gameMode == GameMode.DUEL
                        ? (duelForegroundSide() == DuelMatch.Side.PLAYER
                        ? PLAYER_TINT : AI_TINT)
                        : (aiEnabled ? ACCENT_DARK : TEXT_MUTED)
        );
        String aiLabel;
        if (gameMode == GameMode.DUEL) {
            aiLabel = duelForegroundSide() == DuelMatch.Side.PLAYER
                    ? "玩家前景"
                    : "AI前景";
        } else {
            aiLabel = aiEnabled ? "AI" + aiState.label : "AI关闭";
        }
        boolean labelOnLeft = gameMode == GameMode.DUEL
                ? duelForegroundSide() == DuelMatch.Side.AI
                : aiEnabled;
        float aiTextLeft = labelOnLeft
                ? AI_TOGGLE_LEFT + 8f
                : AI_TOGGLE_LEFT + 39f;
        float aiTextWidth = 97f;
        drawTextInBox(
                smallFont,
                fitText(smallFont, aiLabel, aiTextWidth),
                aiTextLeft,
                AI_TOGGLE_TOP,
                aiTextWidth,
                AI_TOGGLE_HEIGHT,
                Align.center
        );

        smallFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                smallFont,
                gameMode == GameMode.DUEL ? "对战" : "经典",
                MODE_BUTTON_LEFT,
                MODE_BUTTON_TOP,
                MODE_BUTTON_WIDTH,
                MODE_BUTTON_HEIGHT,
                Align.center
        );

        if (modeSwitchConfirmSeconds > 0f) {
            smallFont.setColor(DANGER);
            drawTextInBox(
                    smallFont,
                    "再次点击切换，本局会重置",
                    120f,
                    1068f,
                    320f,
                    30f,
                    Align.center
            );
        } else if (gameMode == GameMode.DUEL) {
            smallFont.setColor(
                    duelRoundUrgent() ? DANGER : TEXT_MUTED
            );
            drawTextInBox(
                    smallFont,
                    duelRoundLabel(),
                    120f,
                    1068f,
                    320f,
                    30f,
                    Align.center
            );
        } else if (!aiEnabled && alive) {
            smallFont.setColor(MANUAL_HINT);
            drawTextInBox(
                    smallFont,
                    "拖动水果，松手投放",
                    140f,
                    1072f,
                    280f,
                    28f,
                    Align.center
            );
        }
    }

    private void drawGameOverOverlay() {
        shapes.begin(ShapeRenderer.ShapeType.Filled);
        shapes.setColor(0.28f, 0.13f, 0.07f, 0.58f);
        shapes.rect(0f, 0f, FruitRules.BOARD_WIDTH, FruitRules.BOARD_HEIGHT);
        roundedRectTop(85f, 401f, 390f, 275f, 24f, CARD_SHADOW);
        roundedRectTop(85f, 395f, 390f, 275f, 24f, GAME_OVER_PANEL);
        roundedRectTop(145f, 585f, 270f, 58f, 29f, ACCENT);
        shapes.end();

        batch.begin();
        titleFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                titleFont,
                "游戏结束",
                105f,
                420f,
                350f,
                54f,
                Align.center
        );
        normalFont.setColor(TEXT_MUTED);
        drawTextInBox(
                normalFont,
                "本局得分  " + score,
                105f,
                485f,
                350f,
                46f,
                Align.center
        );
        smallFont.setColor(ACCENT_DARK);
        drawTextInBox(
                smallFont,
                "恭喜你已超越 " + soloPercentile + "% 的玩家",
                105f,
                540f,
                350f,
                34f,
                Align.center
        );
        normalFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                normalFont,
                "点击重新开始",
                145f,
                585f,
                270f,
                58f,
                Align.center
        );
        batch.end();
    }

    private void drawDuelResultLayer() {
        if (duelMatch == null
                || duelMatch.outcome()
                == DuelMatch.Outcome.IN_PROGRESS) {
            return;
        }
        if (!duelResultVisible) {
            shapes.begin(ShapeRenderer.ShapeType.Filled);
            shapes.setColor(0.24f, 0.12f, 0.08f, 0.24f);
            shapes.rect(0f, 0f, FruitRules.BOARD_WIDTH, FruitRules.BOARD_HEIGHT);
            roundedRectTop(74f, 294f, 412f, 150f, 22f, CARD_SHADOW);
            roundedRectTop(74f, 288f, 412f, 150f, 22f, PANEL_COLOR);
            shapes.end();

            batch.begin();
            normalFont.setColor(TEXT_PRIMARY);
            drawTextInBox(
                    normalFont,
                    "观察淘汰原因",
                    94f,
                    306f,
                    372f,
                    42f,
                    Align.center
            );
            smallFont.setColor(DANGER);
            drawTextInBox(
                    smallFont,
                    duelResultReason,
                    94f,
                    354f,
                    372f,
                    34f,
                    Align.center
            );
            smallFont.setColor(TEXT_MUTED);
            drawTextInBox(
                    smallFont,
                    formatOneDecimal(duelResultHoldRemaining)
                            + " 秒后结算，可切换场景",
                    94f,
                    394f,
                    372f,
                    30f,
                    Align.center
            );
            batch.end();
            return;
        }

        DuelMatch.Outcome outcome = duelMatch.outcome();
        Color resultColor = outcome == DuelMatch.Outcome.PLAYER_WIN
                ? PLAYER_TINT
                : (outcome == DuelMatch.Outcome.AI_WIN
                ? AI_TINT : ACCENT);
        shapes.begin(ShapeRenderer.ShapeType.Filled);
        shapes.setColor(OVERLAY_DIM);
        shapes.rect(0f, 0f, FruitRules.BOARD_WIDTH, FruitRules.BOARD_HEIGHT);
        roundedRectTop(70f, 350f, 420f, 400f, 26f, CARD_SHADOW);
        roundedRectTop(70f, 344f, 420f, 400f, 26f, GAME_OVER_PANEL);
        roundedRectTop(146f, 658f, 268f, 62f, 31f, resultColor);
        shapes.end();

        batch.begin();
        titleFont.setColor(resultColor);
        drawTextInBox(
                titleFont,
                duelResultTitle(),
                90f,
                372f,
                380f,
                58f,
                Align.center
        );
        normalFont.setColor(PLAYER_TINT);
        drawTextInBox(
                normalFont,
                "玩家 " + duelPlayerScore(),
                104f,
                450f,
                170f,
                46f,
                Align.center
        );
        normalFont.setColor(AI_TINT);
        drawTextInBox(
                normalFont,
                "AI " + duelAiScore(),
                286f,
                450f,
                170f,
                46f,
                Align.center
        );
        smallFont.setColor(TEXT_MUTED);
        drawTextInBox(
                smallFont,
                duelResultReason,
                94f,
                518f,
                372f,
                34f,
                Align.center
        );
        smallFont.setColor(ACCENT_DARK);
        drawTextInBox(
                smallFont,
                "你已超越 " + duelPercentile + "% 的玩家",
                94f,
                568f,
                372f,
                34f,
                Align.center
        );
        normalFont.setColor(Color.WHITE);
        drawTextInBox(
                normalFont,
                "点击再战",
                146f,
                658f,
                268f,
                62f,
                Align.center
        );
        batch.end();
    }

    private String duelResultTitle() {
        if (duelMatch.outcome() == DuelMatch.Outcome.PLAYER_WIN) {
            return "对战胜利";
        }
        if (duelMatch.outcome() == DuelMatch.Outcome.AI_WIN) {
            return "对战失败";
        }
        return "势均力敌";
    }

    private void drawOverlayPage() {
        shapes.begin(ShapeRenderer.ShapeType.Filled);
        shapes.setColor(OVERLAY_DIM);
        shapes.rect(0f, 0f, FruitRules.BOARD_WIDTH, FruitRules.BOARD_HEIGHT);
        roundedRectTop(34f, 104f, 492f, 938f, 26f, CARD_SHADOW);
        roundedRectTop(34f, 98f, 492f, 938f, 26f, PANEL_COLOR);
        roundedRectTop(430f, 124f, 70f, 46f, 23f, ACCENT_SOFT);
        if (overlayPage == OverlayPage.SETTINGS) {
            drawSettingsPageShapes();
        } else {
            drawHistoryPageShapes();
        }
        shapes.end();

        batch.begin();
        if (overlayPage == OverlayPage.SETTINGS) {
            drawSettingsPageText();
        } else {
            drawHistoryPageText();
        }
        batch.end();
    }

    private void drawSettingsPageShapes() {
        float[] rows = {218f, 308f, 398f, 488f, 578f, 668f, 758f};
        for (float row : rows) {
            roundedRectTop(62f, row + 4f, 436f, 68f, 16f, CARD_SHADOW);
            roundedRectTop(62f, row, 436f, 68f, 16f, SCORE_CARD);
        }
        // 数值设置使用同一组“减 / 当前值 / 加”控件。
        for (float row : new float[]{218f, 578f, 668f, 758f}) {
            roundedRectTop(350f, row + 12f, 132f, 44f, 22f, NEXT_CARD);
            shapes.setColor(BOARD_FRAME_SOFT);
            shapes.circle(372f, toRenderY(row + 34f), 18f, 24);
            shapes.circle(460f, toRenderY(row + 34f), 18f, 24);
            shapes.setColor(SCORE_CARD);
            shapes.circle(372f, toRenderY(row + 34f), 15f, 24);
            shapes.circle(460f, toRenderY(row + 34f), 15f, 24);
        }
        for (float row : new float[]{308f, 398f, 488f}) {
            roundedRectTop(
                    370f,
                    row + 13f,
                    108f,
                    42f,
                    21f,
                    ACCENT_SOFT
            );
        }
        roundedRectTop(154f, 886f, 252f, 62f, 31f, BOARD_FRAME_SOFT);
        roundedRectTop(158f, 882f, 244f, 58f, 29f, NEXT_CARD);
    }

    private void drawHistoryPageShapes() {
        float[] rows = {230f, 320f, 410f, 500f, 590f, 680f};
        for (float row : rows) {
            roundedRectTop(62f, row + 4f, 436f, 68f, 16f, CARD_SHADOW);
            roundedRectTop(62f, row, 436f, 68f, 16f, SCORE_CARD);
        }
        roundedRectTop(154f, 838f, 252f, 62f, 31f, BOARD_FRAME_SOFT);
        roundedRectTop(
                158f,
                834f,
                244f,
                58f,
                29f,
                historyResetConfirmSeconds > 0f
                        ? PLAYER_TINT_SOFT
                        : NEXT_CARD
        );
    }

    private void drawSettingsPageText() {
        titleFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                titleFont,
                "游戏设置",
                62f,
                124f,
                340f,
                52f,
                Align.left
        );
        smallFont.setColor(ACCENT_DARK);
        drawTextInBox(
                smallFont,
                "返回",
                430f,
                124f,
                70f,
                46f,
                Align.center
        );
        GameProfileStore.Settings current = settings;
        drawSettingRow("音效音量", Math.round(current.soundVolume() * 100f) + "%", 218f, true);
        drawSettingRow("合成震动", onOff(current.vibrateOnMerge()), 308f, false);
        drawSettingRow("投放震动", onOff(current.vibrateOnDrop()), 398f, false);
        drawSettingRow("收分震动", onOff(current.vibrateOnScoreCollect()), 488f, false);
        drawSettingRow(
                "游戏速度",
                formatOneDecimal(current.gameSpeed()) + "x",
                578f,
                true
        );
        drawSettingRow(
                "对战限时",
                Math.round(current.versusDropSeconds()) + "秒",
                668f,
                true
        );
        drawSettingRow(
                "结算停留",
                Math.round(current.resultHoldSeconds()) + "秒",
                758f,
                true
        );
        normalFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                normalFont,
                "恢复默认",
                158f,
                882f,
                244f,
                58f,
                Align.center
        );
    }

    private void drawSettingRow(
            String label,
            String value,
            float top,
            boolean stepper) {
        normalFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                normalFont,
                label,
                82f,
                top + 10f,
                238f,
                48f,
                Align.left
        );
        smallFont.setColor(stepper ? TEXT_PRIMARY : ACCENT_DARK);
        if (stepper) {
            drawTextInBox(smallFont, "-", 354f, top + 12f, 36f, 44f, Align.center);
            drawTextInBox(smallFont, value, 388f, top + 12f, 56f, 44f, Align.center);
            drawTextInBox(smallFont, "+", 442f, top + 12f, 36f, 44f, Align.center);
        } else {
            drawTextInBox(smallFont, value, 370f, top + 13f, 108f, 42f, Align.center);
        }
    }

    private void drawHistoryPageText() {
        titleFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                titleFont,
                "历史记录",
                62f,
                124f,
                340f,
                52f,
                Align.left
        );
        smallFont.setColor(ACCENT_DARK);
        drawTextInBox(
                smallFont,
                "返回",
                430f,
                124f,
                70f,
                46f,
                Align.center
        );
        GameProfileStore.History history = profileStore.history();
        drawHistoryRow("完成局数", history.totalGames(), 230f);
        drawHistoryRow("历史最高分", history.highScore(), 320f);
        drawHistoryRow(
                "单局最多大西瓜",
                history.maxWatermelonsInGame(),
                410f
        );
        drawHistoryRow("累计合成大西瓜", history.totalWatermelons(), 500f);
        drawHistoryRow(
                "对战胜负平",
                history.versusWins() + " / "
                        + history.versusLosses() + " / "
                        + history.versusDraws(),
                590f
        );
        drawHistoryRow("对战最高分", history.highestVersusScore(), 680f);

        normalFont.setColor(
                historyResetConfirmSeconds > 0f ? DANGER : TEXT_PRIMARY
        );
        drawTextInBox(
                normalFont,
                historyResetConfirmSeconds > 0f
                        ? "再次点击确认重置"
                        : "重置记录",
                158f,
                834f,
                244f,
                58f,
                Align.center
        );
    }

    private void drawHistoryRow(String label, int value, float top) {
        drawHistoryRow(label, Integer.toString(value), top);
    }

    private void drawHistoryRow(String label, String value, float top) {
        normalFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                normalFont,
                label,
                82f,
                top + 10f,
                272f,
                48f,
                Align.left
        );
        normalFont.setColor(TEXT_MUTED);
        drawTextInBox(
                normalFont,
                value,
                350f,
                top + 10f,
                126f,
                48f,
                Align.right
        );
    }

    private String onOff(boolean value) {
        return value ? "开启" : "关闭";
    }

    private String formatOneDecimal(float value) {
        return String.format(java.util.Locale.ROOT, "%.1f", value);
    }

    private void roundedRectTop(
            float x,
            float top,
            float width,
            float height,
            float radius,
            Color color) {
        float y = toRenderY(top + height);
        shapes.setColor(color);
        shapes.rect(x + radius, y, width - 2f * radius, height);
        shapes.rect(x, y + radius, width, height - 2f * radius);
        shapes.circle(x + radius, y + radius, radius, 24);
        shapes.circle(x + width - radius, y + radius, radius, 24);
        shapes.circle(x + radius, y + height - radius, radius, 24);
        shapes.circle(x + width - radius, y + height - radius, radius, 24);
    }

    /**
     * 在逻辑坐标矩形内按可见字形尺寸对齐，不再依赖字体特定的裸 baseline 常数。
     */
    private void drawTextInBox(
            BitmapFont font,
            String text,
            float left,
            float top,
            float width,
            float height,
            int alignment) {
        glyphLayout.setText(font, text == null ? "" : text);
        float x = left;
        if ((alignment & Align.right) != 0) {
            x = left + width - glyphLayout.width;
        } else if ((alignment & Align.left) == 0) {
            x = left + (width - glyphLayout.width) * 0.5f;
        }
        float visibleTop = top + (height - glyphLayout.height) * 0.5f;
        font.draw(batch, glyphLayout, x, toRenderY(visibleTop));
    }

    private float toRenderY(float screenY) {
        return FruitRules.BOARD_HEIGHT - screenY;
    }

    /** 按真实像素宽度截断动态状态，避免不同字宽再次越过 AI 胶囊。 */
    private String fitText(BitmapFont font, String value, float maximumWidth) {
        String safeValue = value == null ? "" : value;
        glyphLayout.setText(font, safeValue);
        if (glyphLayout.width <= maximumWidth) {
            return safeValue;
        }
        String suffix = "...";
        for (int length = safeValue.length() - 1; length > 0; length--) {
            String candidate = safeValue.substring(0, length) + suffix;
            glyphLayout.setText(font, candidate);
            if (glyphLayout.width <= maximumWidth) {
                return candidate;
            }
        }
        return suffix;
    }

    private String sanitizeStatus(String message) {
        if (message == null || message.isEmpty()) {
            return "未知";
        }
        return message.replace('\n', ' ').replace('\r', ' ');
    }

    private boolean isAiRuntimeLoading() {
        String status = aiService.aiRuntimeStatus();
        if (status == null) {
            return false;
        }
        String normalized = status.toLowerCase(java.util.Locale.ROOT);
        return normalized.contains("loading")
                || normalized.contains("starting")
                || normalized.contains("initializing");
    }

    private boolean isInside(float x, float y, float left, float top, float width, float height) {
        return x >= left && x <= left + width && y >= top && y <= top + height;
    }

    private void setAiEnabled(boolean enabled) {
        if (aiEnabled == enabled) {
            return;
        }
        aiEnabled = enabled;
        cancelPendingDecision(enabled ? "AI已开启" : "手动模式");
        aiState = enabled ? AiState.OBSERVING : AiState.MANUAL;
        aiDetail = enabled ? "AI已开启" : "拖动水果，松手投放";
        activeDragPointer = -1;
        stableSeconds = 0f;
        previewAnchorX = previewX;
        if (enabled) {
            aiLoadingSeconds = 0f;
        } else if (!waiting && queue.size > 0) {
            /*
             * AI 刚投下水果后切换人工模式时，也立即提供下一颗的预摆入口。
             * 仍保留极短人工投放冷却，避免切换按钮的同一次触摸误投。
             */
            currentLevel = queue.first();
            waiting = true;
            dropCooldown = Math.min(
                    dropCooldown,
                    MANUAL_DROP_COOLDOWN_SECONDS
            );
            previewX = FruitRules.clampDropX(previewX, currentLevel);
            previewAnchorX = previewX;
        }
    }

    private void openOverlay(OverlayPage page) {
        overlayPage = page;
        activeDragPointer = -1;
        historyResetConfirmSeconds = 0f;
        if (gameMode == GameMode.CLASSIC) {
            cancelPendingDecision("页面暂停");
        } else {
            invalidateDuelDecision();
        }
    }

    private void closeOverlay() {
        overlayPage = OverlayPage.NONE;
        activeDragPointer = -1;
        historyResetConfirmSeconds = 0f;
    }

    private void handleOverlayTouch(float x, float y) {
        if (isInside(x, y, 430f, 124f, 70f, 46f)) {
            closeOverlay();
            return;
        }
        if (overlayPage == OverlayPage.SETTINGS) {
            handleSettingsTouch(x, y);
        } else {
            handleHistoryTouch(x, y);
        }
    }

    private void handleSettingsTouch(float x, float y) {
        if (isInside(x, y, 158f, 882f, 244f, 58f)) {
            profileStore.resetSettings();
            settings = profileStore.settings();
            return;
        }
        if (isInside(x, y, 370f, 321f, 108f, 42f)) {
            profileStore
                    .setVibrateOnMerge(!settings.vibrateOnMerge())
                    .save();
            settings = profileStore.settings();
            return;
        }
        if (isInside(x, y, 370f, 411f, 108f, 42f)) {
            profileStore
                    .setVibrateOnDrop(!settings.vibrateOnDrop())
                    .save();
            settings = profileStore.settings();
            return;
        }
        if (isInside(x, y, 370f, 501f, 108f, 42f)) {
            profileStore
                    .setVibrateOnScoreCollect(
                            !settings.vibrateOnScoreCollect()
                    )
                    .save();
            settings = profileStore.settings();
            return;
        }
        if (isStepperTouch(x, y, 218f)) {
            float delta = x < 416f ? -0.1f : 0.1f;
            profileStore
                    .setSoundVolume(settings.soundVolume() + delta)
                    .save();
            settings = profileStore.settings();
            return;
        }
        if (isStepperTouch(x, y, 578f)) {
            float delta = x < 416f ? -0.25f : 0.25f;
            profileStore
                    .setGameSpeed(settings.gameSpeed() + delta)
                    .save();
            settings = profileStore.settings();
            return;
        }
        if (isStepperTouch(x, y, 668f)) {
            float delta = x < 416f ? -1f : 1f;
            profileStore
                    .setVersusDropSeconds(
                            settings.versusDropSeconds() + delta
                    )
                    .save();
            settings = profileStore.settings();
            return;
        }
        if (isStepperTouch(x, y, 758f)) {
            float delta = x < 416f ? -1f : 1f;
            profileStore
                    .setResultHoldSeconds(
                            settings.resultHoldSeconds() + delta
                    )
                    .save();
            settings = profileStore.settings();
        }
    }

    private boolean isStepperTouch(float x, float y, float top) {
        boolean minus = isInside(x, y, 354f, top + 12f, 36f, 44f);
        boolean plus = isInside(x, y, 442f, top + 12f, 36f, 44f);
        return minus || plus;
    }

    private void handleHistoryTouch(float x, float y) {
        if (!isInside(x, y, 158f, 834f, 244f, 58f)) {
            return;
        }
        if (historyResetConfirmSeconds <= 0f) {
            historyResetConfirmSeconds = 3f;
            return;
        }
        profileStore.resetHistory();
        bestScore = 0;
        displayedBestScore = Math.max(displayedScore, bestScore);
        historyResetConfirmSeconds = 0f;
    }

    private void requestModeSwitch() {
        if (gameMode == GameMode.DUEL
                && duelMatch != null
                && duelMatch.outcome()
                != DuelMatch.Outcome.IN_PROGRESS
                && !duelResultVisible) {
            // 淘汰原因观察期是规则的一部分，不能用模式切换提前跳过。
            return;
        }
        boolean hasProgress = gameMode == GameMode.CLASSIC
                ? stepCount > 0
                : duelHasProgress();
        if (hasProgress && modeSwitchConfirmSeconds <= 0f) {
            modeSwitchConfirmSeconds = 2.6f;
            return;
        }
        modeSwitchConfirmSeconds = 0f;
        activeDragPointer = -1;
        decisionEpoch += 1;
        aiRequestInFlight = false;
        motionPlan = null;
        if (gameMode == GameMode.CLASSIC) {
            gameMode = GameMode.DUEL;
            resetDuelGame();
        } else {
            gameMode = GameMode.CLASSIC;
            disposeDuelGame();
            resetGame();
        }
    }

    private DuelMatch.Side duelForegroundSide() {
        return duelForeground;
    }

    private DuelMatch.Lane duelForegroundLane() {
        return duelMatch.lane(duelForeground);
    }

    private int duelPlayerScore() {
        return duelMatch == null ? 0 : duelMatch.playerLane().score();
    }

    private int duelAiScore() {
        return duelMatch == null ? 0 : duelMatch.aiLane().score();
    }

    private boolean duelRoundUrgent() {
        return duelMatch != null
                && duelMatch.outcome()
                == DuelMatch.Outcome.IN_PROGRESS
                && duelMatch.roundOpen()
                && duelMatch.roundRemainingSeconds() <= 2f;
    }

    private String duelRoundLabel() {
        if (duelMatch == null) {
            return "正在准备对战";
        }
        if (duelMatch.outcome() != DuelMatch.Outcome.IN_PROGRESS) {
            return duelResultVisible
                    ? "对战已结束"
                    : "正在观察淘汰原因";
        }
        if (duelMatch.awaitingNextRound()) {
            return "双方已投放，准备下一颗";
        }
        return "本轮剩余 "
                + formatOneDecimal(duelMatch.roundRemainingSeconds())
                + " 秒";
    }

    private boolean duelForegroundPreviewVisible() {
        if (duelMatch == null || !duelMatch.roundOpen()) {
            return false;
        }
        DuelMatch.Lane lane = duelForegroundLane();
        return lane.alive() && !lane.submittedThisRound();
    }

    private boolean duelHasProgress() {
        return duelMatch != null
                && (duelMatch.roundIndex() > 0
                || duelMatch.playerLane().stepCount() > 0
                || duelMatch.aiLane().stepCount() > 0);
    }

    private void toggleDuelForeground() {
        if (duelMatch == null) {
            return;
        }
        duelForeground = duelForeground == DuelMatch.Side.PLAYER
                ? DuelMatch.Side.AI
                : DuelMatch.Side.PLAYER;
        particles.clear();
        rings.clear();
        mergeBursts.clear();
        duelScoreTokens.clear();
    }

    private boolean duelCanPlayerDrag() {
        return duelMatch != null
                && duelMatch.outcome()
                == DuelMatch.Outcome.IN_PROGRESS
                && duelMatch.roundOpen()
                && duelMatch.playerLane().alive()
                && !duelMatch.playerLane().submittedThisRound();
    }

    private boolean duelCanPlayerDrop() {
        return duelCanPlayerDrag();
    }

    private void setDuelPlayerPreviewX(float x) {
        if (duelMatch != null) {
            duelMatch.setPlayerPreviewX(x);
        }
    }

    private void moveDuelPlayerPreview(float deltaX) {
        if (!duelCanPlayerDrag()) {
            return;
        }
        duelMatch.setPlayerPreviewX(
                duelMatch.playerLane().previewX() + deltaX
        );
    }

    private void dropDuelPlayer() {
        if (!duelCanPlayerDrop()) {
            return;
        }
        if (duelMatch.dropPlayer(
                duelMatch.playerLane().previewX())) {
            spawnDuelDropFeedback(DuelMatch.Side.PLAYER);
        }
    }

    private void dropDuelAiAt(float x) {
        if (duelMatch == null) {
            return;
        }
        if (duelMatch.dropAi(x)) {
            spawnDuelDropFeedback(DuelMatch.Side.AI);
            invalidateDuelDecision();
        }
    }

    private void spawnDuelDropFeedback(DuelMatch.Side side) {
        DuelMatch.Lane lane = duelMatch.lane(side);
        if (side == duelForeground) {
            spawnDropEffect(
                    lane.previewX(),
                    previewY(duelMatch.currentLevel()),
                    duelMatch.currentLevel()
            );
        }
        if (side == DuelMatch.Side.PLAYER) {
            vibrateIf(settings.vibrateOnDrop(), 10);
        }
    }

    private boolean duelResultCanRestart() {
        return duelMatch != null
                && duelMatch.outcome()
                != DuelMatch.Outcome.IN_PROGRESS
                && duelResultVisible;
    }

    private void updateTouchPoint(int screenX, int screenY) {
        touchPoint.set(screenX, screenY, 0f);
        viewport.unproject(touchPoint);
        // viewport 使用 y-up 世界；游戏与模型使用 y-down 屏幕坐标。
        touchPoint.y = FruitRules.BOARD_HEIGHT - touchPoint.y;
    }

    @Override
    public boolean touchDown(int screenX, int screenY, int pointer, int button) {
        if (!touchIsInsideViewport(screenX, screenY)) {
            return false;
        }
        if (activeDragPointer >= 0 && pointer != activeDragPointer) {
            return false;
        }
        updateTouchPoint(screenX, screenY);
        if (overlayPage != OverlayPage.NONE) {
            handleOverlayTouch(touchPoint.x, touchPoint.y);
            return true;
        }
        if (isInside(
                touchPoint.x,
                touchPoint.y,
                SETTINGS_BUTTON_LEFT,
                UTILITY_BUTTON_TOP,
                UTILITY_BUTTON_SIZE,
                UTILITY_BUTTON_SIZE
        )) {
            openOverlay(OverlayPage.SETTINGS);
            return true;
        }
        if (isInside(
                touchPoint.x,
                touchPoint.y,
                HISTORY_BUTTON_LEFT,
                UTILITY_BUTTON_TOP,
                UTILITY_BUTTON_SIZE,
                UTILITY_BUTTON_SIZE
        )) {
            openOverlay(OverlayPage.HISTORY);
            return true;
        }
        if (isInside(
                touchPoint.x,
                touchPoint.y,
                MODE_BUTTON_LEFT,
                MODE_BUTTON_TOP,
                MODE_BUTTON_WIDTH,
                MODE_BUTTON_HEIGHT
        )) {
            requestModeSwitch();
            return true;
        }
        if (isInside(
                touchPoint.x,
                touchPoint.y,
                AI_TOGGLE_LEFT,
                AI_TOGGLE_TOP,
                AI_TOGGLE_WIDTH,
                AI_TOGGLE_HEIGHT
        )) {
            if (gameMode == GameMode.DUEL) {
                toggleDuelForeground();
            } else {
                setAiEnabled(!aiEnabled);
            }
            return true;
        }
        if (gameMode == GameMode.CLASSIC && !alive) {
            resetGame();
            return true;
        }
        if (gameMode == GameMode.DUEL && duelResultCanRestart()) {
            resetDuelGame();
            return true;
        }
        if (gameMode == GameMode.CLASSIC
                && !aiEnabled
                && canManualDragCurrent()
                && touchPoint.y >= MANUAL_INPUT_TOP) {
            activeDragPointer = pointer;
            previewX = FruitRules.clampDropX(touchPoint.x, currentLevel);
            return true;
        }
        if (gameMode == GameMode.DUEL
                && duelCanPlayerDrag()
                && touchPoint.y >= MANUAL_INPUT_TOP) {
            activeDragPointer = pointer;
            setDuelPlayerPreviewX(touchPoint.x);
            return true;
        }
        return false;
    }

    @Override
    public boolean touchDragged(int screenX, int screenY, int pointer) {
        if (pointer != activeDragPointer
                || !touchIsInsideViewport(screenX, screenY)) {
            return false;
        }
        updateTouchPoint(screenX, screenY);
        if (gameMode == GameMode.CLASSIC
                && canManualDragCurrent()) {
            previewX = FruitRules.clampDropX(touchPoint.x, currentLevel);
            return true;
        }
        if (gameMode == GameMode.DUEL && duelCanPlayerDrag()) {
            setDuelPlayerPreviewX(touchPoint.x);
            return true;
        }
        return false;
    }

    @Override
    public boolean touchUp(int screenX, int screenY, int pointer, int button) {
        if (pointer != activeDragPointer) {
            return false;
        }
        activeDragPointer = -1;
        if (!touchIsInsideViewport(screenX, screenY)) {
            return true;
        }
        updateTouchPoint(screenX, screenY);
        if (gameMode == GameMode.CLASSIC
                && !aiEnabled
                && canManualDropCurrent()
                && touchPoint.y >= MANUAL_INPUT_TOP) {
            previewX = FruitRules.clampDropX(touchPoint.x, currentLevel);
            dropCurrent(previewX);
            return true;
        }
        if (gameMode == GameMode.DUEL
                && duelCanPlayerDrop()
                && touchPoint.y >= MANUAL_INPUT_TOP) {
            setDuelPlayerPreviewX(touchPoint.x);
            dropDuelPlayer();
            return true;
        }
        return false;
    }

    @Override
    public boolean touchCancelled(
            int screenX,
            int screenY,
            int pointer,
            int button) {
        if (pointer == activeDragPointer) {
            activeDragPointer = -1;
            return true;
        }
        return false;
    }

    @Override
    public boolean keyDown(int keycode) {
        if (overlayPage != OverlayPage.NONE) {
            if (keycode == Input.Keys.ESCAPE) {
                closeOverlay();
            }
            // 配置/历史页是暂停层，不能让键盘在其背后重开或投放水果。
            return true;
        }
        if (keycode == Input.Keys.R) {
            if (gameMode == GameMode.DUEL) {
                if (duelMatch == null
                        || duelMatch.outcome()
                        == DuelMatch.Outcome.IN_PROGRESS
                        || duelResultVisible) {
                    resetDuelGame();
                }
            } else {
                resetGame();
            }
            return true;
        }
        if (keycode == Input.Keys.SPACE || keycode == Input.Keys.ENTER) {
            if (gameMode == GameMode.DUEL) {
                dropDuelPlayer();
            } else if (!aiEnabled) {
                dropCurrent(previewX);
            }
            return true;
        }
        if (keycode == Input.Keys.A || keycode == Input.Keys.LEFT) {
            if (gameMode == GameMode.DUEL) {
                moveDuelPlayerPreview(-14f);
            } else if (!aiEnabled && waiting) {
                previewX = FruitRules.clampDropX(previewX - 14f, currentLevel);
            }
            return true;
        }
        if (keycode == Input.Keys.D || keycode == Input.Keys.RIGHT) {
            if (gameMode == GameMode.DUEL) {
                moveDuelPlayerPreview(14f);
            } else if (!aiEnabled && waiting) {
                previewX = FruitRules.clampDropX(previewX + 14f, currentLevel);
            }
            return true;
        }
        return false;
    }

    @Override public boolean keyUp(int keycode) { return false; }
    @Override public boolean keyTyped(char character) { return false; }
    @Override public boolean mouseMoved(int screenX, int screenY) { return false; }
    @Override public boolean scrolled(float amountX, float amountY) { return false; }

    private boolean touchIsInsideViewport(int screenX, int screenY) {
        int left = viewport.getScreenX();
        int bottom = viewport.getScreenY();
        int width = viewport.getScreenWidth();
        int height = viewport.getScreenHeight();
        int top = Gdx.graphics.getHeight() - (bottom + height);
        return screenX >= left
                && screenX < left + width
                && screenY >= top
                && screenY < top + height;
    }

    private enum GameMode {
        CLASSIC,
        DUEL
    }

    private enum OverlayPage {
        NONE,
        SETTINGS,
        HISTORY
    }

    private enum AiState {
        MANUAL("手动"),
        OBSERVING("观察中"),
        THINKING("思考中"),
        TESTING("试探中"),
        COMMITTING("投放中"),
        GAME_OVER("已结束");

        private final String label;

        AiState(String label) {
            this.label = label;
        }
    }

    /** 一段不使用光滑贝塞尔曲线的拟人拖动轨迹。 */
    private static final class MotionPlan {
        private final AiDecision decision;
        private final Array<MotionSegment> segments;
        private final Random random;
        private int segmentIndex;
        private float segmentTime;
        private float holdTime;
        private float startX;
        private float currentX;
        private float jitter;
        private float jitterTarget;
        private float renderJitter;
        private float jitterRefresh;
        private float gestureTime;
        private final float wavePhase;

        private MotionPlan(
                AiDecision decision,
                Array<MotionSegment> segments,
                float startX,
                Random random) {
            this.decision = decision;
            this.segments = segments;
            this.startX = startX;
            this.currentX = startX;
            this.random = random;
            this.wavePhase = random.nextFloat() * MathUtils.PI2;
        }

        private static MotionPlan create(
                AiDecision decision,
                float currentX,
                int level,
                Random random) {
            Array<MotionSegment> segments = new Array<>();
            float margin = MathUtils.clamp(decision.normalizedChoiceMargin(), 0f, 0.12f);
            float uncertainty = 1f - margin / 0.12f;

            // 模型前两名接近时先探向备选列；差距明显时只做一次短促修正。
            if (uncertainty > 0.34f
                    && decision.alternativeActionIndex >= 0
                    && decision.alternativeActionIndex < FruitRules.ACTION_COUNT) {
                float alternative = FruitRules.actionDropX(
                        decision.alternativeActionIndex,
                        level
                );
                segments.add(new MotionSegment(
                        FruitRules.clampDropX(
                                alternative + signedRandom(random, 2.4f),
                                level
                        ),
                        0.24f + uncertainty * 0.18f,
                        0.10f + uncertainty * 0.11f,
                        AiState.TESTING,
                        "尝试其他位置"
                ));
            }

            float selected = FruitRules.actionDropX(decision.actionIndex, level);
            float approachDirection = Math.signum(selected - currentX);
            if (approachDirection == 0f) {
                approachDirection = random.nextBoolean() ? 1f : -1f;
            }
            segments.add(new MotionSegment(
                    FruitRules.clampDropX(
                            selected + approachDirection
                                    * (2f + random.nextFloat() * 4.5f),
                            level
                    ),
                    0.22f + random.nextFloat() * 0.16f,
                    0.09f + random.nextFloat() * 0.08f,
                    AiState.TESTING,
                    "微调位置"
            ));
            segments.add(new MotionSegment(
                    selected,
                    0.18f + random.nextFloat() * 0.12f,
                    0.10f + uncertainty * 0.08f,
                    AiState.COMMITTING,
                    "决定完成"
            ));
            return new MotionPlan(decision, segments, currentX, random);
        }

        private MotionSample update(float delta) {
            gestureTime += delta;
            if (holdTime > 0f) {
                MotionSegment heldSegment = segments.get(
                        Math.max(0, segmentIndex - 1)
                );
                holdTime = Math.max(0f, holdTime - delta);
                refreshJitter(
                        delta,
                        heldSegment.state == AiState.COMMITTING ? 0.18f : 0.55f
                );
                return new MotionSample(
                        currentX + renderJitter,
                        heldSegment.state,
                        holdTime <= 0f && segmentIndex >= segments.size
                                ? "投放"
                                : heldSegment.detail,
                        holdTime <= 0f && segmentIndex >= segments.size
                );
            }

            if (segmentIndex >= segments.size) {
                return new MotionSample(
                        currentX,
                        AiState.COMMITTING,
                        "投放",
                        true
                );
            }

            MotionSegment segment = segments.get(segmentIndex);
            segmentTime += delta;
            float progress = MathUtils.clamp(segmentTime / segment.duration, 0f, 1f);
            /*
             * 拖动主体采用轻微 ease-out；手部误差来自低通后的相关噪声和低频波，
             * 不再每隔几十毫秒瞬移到一个全新偏移。
             */
            float eased = 1f - (float) Math.pow(1f - progress, 2.15f);
            float baseX = startX + (segment.targetX - startX) * eased;
            refreshJitter(
                    delta,
                    segment.state == AiState.COMMITTING ? 0.14f : 0.45f
            );
            currentX = baseX;
            if (progress < 1f) {
                return new MotionSample(
                        currentX + renderJitter,
                        segment.state,
                        segment.detail,
                        false
                );
            }

            currentX = segment.targetX;
            holdTime = segment.hold;
            segmentTime = 0f;
            startX = currentX;
            segmentIndex += 1;
            return new MotionSample(
                    currentX + renderJitter,
                    segment.state,
                    segment.detail,
                    segmentIndex >= segments.size && holdTime <= 0f
            );
        }

        private void refreshJitter(float delta, float amplitude) {
            jitterRefresh -= delta;
            if (jitterRefresh <= 0f) {
                jitterTarget = signedRandom(random, amplitude);
                jitterRefresh = 0.12f + random.nextFloat() * 0.10f;
            }
            float response = 1f - (float) Math.exp(-delta * 8.5f);
            jitter += (jitterTarget - jitter) * response;
            renderJitter = jitter
                    + MathUtils.sin(gestureTime * 6.4f + wavePhase)
                    * amplitude * 0.13f;
        }

        private static float signedRandom(Random random, float amplitude) {
            return (random.nextFloat() * 2f - 1f) * amplitude;
        }
    }

    private static final class MotionSegment {
        private final float targetX;
        private final float duration;
        private final float hold;
        private final AiState state;
        private final String detail;

        private MotionSegment(
                float targetX,
                float duration,
                float hold,
                AiState state,
                String detail) {
            this.targetX = targetX;
            this.duration = duration;
            this.hold = hold;
            this.state = state;
            this.detail = detail;
        }
    }

    private static final class MotionSample {
        private final float x;
        private final AiState state;
        private final String detail;
        private final boolean finished;

        private MotionSample(
                float x,
                AiState state,
                String detail,
                boolean finished) {
            this.x = x;
            this.state = state;
            this.detail = detail;
            this.finished = finished;
        }
    }

    private static final class MergeCue {
        private final float x;
        private final float y;
        private final int level;
        private final int scoreDelta;
        private final int cascadeDepth;
        private float delay;
        private final ScoreSequence sequence;

        private MergeCue(
                float x,
                float y,
                int level,
                int scoreDelta,
                int cascadeDepth,
                float delay,
                ScoreSequence sequence) {
            this.x = x;
            this.y = y;
            this.level = level;
            this.scoreDelta = scoreDelta;
            this.cascadeDepth = cascadeDepth;
            this.delay = delay;
            this.sequence = sequence;
        }
    }

    private static final class ScoreSequence {
        private final int id;
        private final Color lastColor = new Color(SCORE_GLOW);
        private final IntIntMap cascadeDepthByFruitId = new IntIntMap();
        private int scoreTarget;
        private int pendingCues;
        private int pendingTokens;
        private int maxCascadeDepth;
        private float secondsSinceLastCue;
        private boolean hasMerges;
        private boolean released;
        private boolean rollQueued;
        private boolean forceRelease;

        private ScoreSequence(int id, int scoreBefore) {
            this.id = id;
            this.scoreTarget = scoreBefore;
        }

        private int registerMerge(
                int sourceFruitIdA,
                int sourceFruitIdB,
                int resultFruitId) {
            int parentDepth = Math.max(
                    cascadeDepthByFruitId.get(sourceFruitIdA, 0),
                    cascadeDepthByFruitId.get(sourceFruitIdB, 0)
            );
            cascadeDepthByFruitId.remove(sourceFruitIdA, 0);
            cascadeDepthByFruitId.remove(sourceFruitIdB, 0);
            int depth = parentDepth + 1;
            cascadeDepthByFruitId.put(resultFruitId, depth);
            maxCascadeDepth = Math.max(maxCascadeDepth, depth);
            return depth;
        }
    }

    private enum TokenPhase {
        POP,
        HOLD,
        FLY
    }

    private static final class ScoreToken {
        private final float originX;
        private final float originY;
        private final int value;
        private final Color color;
        private final ScoreSequence sequence;
        private TokenPhase phase = TokenPhase.POP;
        private float x;
        private float y;
        private float age;
        private float scale = 0.4f;
        private float alpha;
        private float flightAge;
        private float flightStartX;
        private float flightStartY;
        private float curveOffset;
        private float trailClock;

        private ScoreToken(
                float x,
                float y,
                int value,
                Color color,
                ScoreSequence sequence) {
            this.originX = x;
            this.originY = y;
            this.x = x;
            this.y = y;
            this.value = value;
            this.color = new Color(color);
            this.sequence = sequence;
        }
    }

    private static final class DuelScoreToken {
        private final DuelMatch.Side side;
        private final float originX;
        private final float originY;
        private final int value;
        private final Color color;
        private float x;
        private float y;
        private float age;
        private float scale = 0.45f;
        private float alpha;

        private DuelScoreToken(
                DuelMatch.Side side,
                float originX,
                float originY,
                int value,
                Color color) {
            this.side = side;
            this.originX = originX;
            this.originY = originY;
            this.value = value;
            this.color = new Color(color);
            x = originX;
            y = originY;
        }
    }

    private static final class MergeBurst {
        private final float x;
        private final float y;
        private final float radius;
        private final float maxLife;
        private float life;
        private final Color color;
        private final float rotation;
        private final int rays;

        private MergeBurst(
                float x,
                float y,
                float radius,
                float life,
                Color color,
                float rotation,
                int rays) {
            this.x = x;
            this.y = y;
            this.radius = radius;
            this.life = life;
            this.maxLife = life;
            this.color = new Color(color);
            this.rotation = rotation;
            this.rays = rays;
        }
    }

    private static final class Particle {
        private float x;
        private float y;
        private float vx;
        private float vy;
        private float life;
        private final float maxLife;
        private final Color color;
        private final float radius;
        private final float gravity;
        private final float drag;

        private Particle(
                float x,
                float y,
                float vx,
                float vy,
                float life,
                Color color,
                float radius,
                float gravity,
                float drag) {
            this.x = x;
            this.y = y;
            this.vx = vx;
            this.vy = vy;
            this.life = life;
            this.maxLife = life;
            this.color = new Color(color);
            this.radius = radius;
            this.gravity = gravity;
            this.drag = drag;
        }
    }

    private static final class Ring {
        private final float x;
        private final float y;
        private float radius;
        private final float speed;
        private float life;
        private final float maxLife;
        private final Color color;
        private final float dotRadius;

        private Ring(
                float x,
                float y,
                float radius,
                float speed,
                float life,
                Color color,
                float dotRadius) {
            this.x = x;
            this.y = y;
            this.radius = radius;
            this.speed = speed;
            this.life = life;
            this.maxLife = life;
            this.color = new Color(color);
            this.dotRadius = dotRadius;
        }
    }
}
