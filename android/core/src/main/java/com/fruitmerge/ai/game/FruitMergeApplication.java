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
import java.util.EnumMap;
import java.util.List;
import java.util.Random;

/**
 * Android 版《合成大西瓜》的完整游戏循环。
 *
 * <p>画面采用 560×1120 固定逻辑坐标，FitViewport 在不同手机上只增加留边，不改变
 * 训练时的几何。手动模式允许拖动并抬手投放；AI 模式在连续稳定边界上请求一次
 * 模型，并在决策返回后立即投放。</p>
 */
public final class FruitMergeApplication extends ApplicationAdapter
        implements InputProcessor {
    private static final float MANUAL_DROP_COOLDOWN_SECONDS = 0.14f;
    private static final float AI_SLIDE_MIN_SECONDS = 0.12f;
    private static final float AI_SLIDE_MAX_SECONDS = 0.32f;
    private static final float MANUAL_INPUT_TOP = 144f;
    private static final float AI_LOADING_FALLBACK_SECONDS = 12f;
    private static final float AUTOSAVE_INTERVAL_SECONDS = 1.5f;
    private static final float URGENT_EMOTION_TTL_SECONDS = 4f;
    private static final float GAME_BACK_LEFT = 490f;
    private static final float GAME_BACK_TOP = 18f;
    private static final float GAME_BACK_SIZE = 42f;
    // Kept only until the legacy HUD helper is removed; no longer interactive.
    private static final float MODE_BUTTON_LEFT = 188f;
    private static final float MODE_BUTTON_TOP = 18f;
    private static final float MODE_BUTTON_WIDTH = 86f;
    private static final float MODE_BUTTON_HEIGHT = 42f;
    private static final float HISTORY_BUTTON_LEFT = 440f;
    private static final float SETTINGS_BUTTON_LEFT = 490f;
    private static final float UTILITY_BUTTON_TOP = 18f;
    private static final float UTILITY_BUTTON_SIZE = 42f;
    private static final float AI_TOGGLE_LEFT = 418f;
    private static final float AI_TOGGLE_TOP = 18f;
    private static final float AI_TOGGLE_WIDTH = 60f;
    private static final float AI_TOGGLE_HEIGHT = 42f;
    private static final float HOME_CARD_LEFT = 54f;
    private static final float HOME_CARD_WIDTH = 452f;
    private static final float HOME_CARD_HEIGHT = 128f;
    private static final float HOME_SOLO_TOP = 326f;
    private static final float HOME_DUEL_TOP = 476f;
    private static final float HOME_DEMO_TOP = 626f;
    private static final float HOME_UTILITY_TOP = 826f;
    private static final float HOME_UTILITY_WIDTH = 214f;
    private static final float HOME_UTILITY_HEIGHT = 68f;
    private static final float RESULT_BUTTON_LEFT = 145f;
    private static final float RESULT_BUTTON_TOP = 596f;
    private static final float RESULT_BUTTON_WIDTH = 270f;
    private static final float RESULT_BUTTON_HEIGHT = 62f;
    private static final String CONTROL_HOME_SOLO = "home.solo";
    private static final String CONTROL_HOME_DUEL = "home.duel";
    private static final String CONTROL_HOME_DEMO = "home.demo";
    private static final String CONTROL_HOME_RESUME = "home.resume";
    private static final String CONTROL_HOME_SETTINGS = "home.settings";
    private static final String CONTROL_HOME_HISTORY = "home.history";
    private static final String CONTROL_GAME_BACK = "game.back";
    private static final String CONTROL_DUEL_FOREGROUND =
            "game.duel.foreground";
    private static final String CONTROL_RESULT_CONFIRM = "result.confirm";
    private static final String CONTROL_OVERLAY_CLOSE = "overlay.close";
    private static final String CONTROL_SETTINGS_RESET = "settings.reset";
    private static final String CONTROL_HISTORY_RESET = "history.reset";
    private static final String CONTROL_HISTORY_PREVIOUS =
            "history.previous";
    private static final String CONTROL_HISTORY_NEXT = "history.next";
    private static final String CONTROL_EXIT_SAVE = "exit.save";
    private static final String CONTROL_EXIT_ABANDON = "exit.abandon";
    private static final String CONTROL_EXIT_CANCEL = "exit.cancel";
    private static final String CONTROL_NEW_CONFIRM = "new.confirm";
    private static final String CONTROL_NEW_RESUME = "new.resume";
    private static final String CONTROL_NEW_CANCEL = "new.cancel";
    private static final String CONTROL_SETTINGS_SOUND_MINUS =
            "settings.sound.minus";
    private static final String CONTROL_SETTINGS_SOUND_PLUS =
            "settings.sound.plus";
    private static final String CONTROL_SETTINGS_MERGE_VIBRATE =
            "settings.vibrate.merge";
    private static final String CONTROL_SETTINGS_DROP_VIBRATE =
            "settings.vibrate.drop";
    private static final String CONTROL_SETTINGS_SCORE_VIBRATE =
            "settings.vibrate.score";
    private static final String CONTROL_SETTINGS_SPEED_MINUS =
            "settings.speed.minus";
    private static final String CONTROL_SETTINGS_SPEED_PLUS =
            "settings.speed.plus";
    private static final String CONTROL_SETTINGS_TIMER_MINUS =
            "settings.timer.minus";
    private static final String CONTROL_SETTINGS_TIMER_PLUS =
            "settings.timer.plus";
    private static final String CONTROL_SETTINGS_HOLD_MINUS =
            "settings.hold.minus";
    private static final String CONTROL_SETTINGS_HOLD_PLUS =
            "settings.hold.plus";
    private static final float MERGE_CUE_GAP_SECONDS = 0.105f;
    private static final float MERGE_PRESENTATION_SETTLE_SECONDS = 0.28f;
    private static final float MERGE_PRESENTATION_MAX_HOLD_SECONDS = 1.45f;
    private static final float SCORE_TOKEN_POP_SECONDS = 0.34f;
    private static final float SCORE_TOKEN_FLIGHT_SECONDS = 0.48f;
    private static final float SCORE_TARGET_X = 280f;
    private static final float SCORE_TOP_DOCK_Y = 64f;
    private static final float SCORE_BOTTOM_DOCK_Y = 1044f;
    private static final float SCORE_CARD_HEIGHT = 66f;
    private static final float SCORE_TARGET_OFFSET_Y = 37f;
    private static final float BOTTOM_DOCK_SCENE_OFFSET_Y = -66f;
    private static final float HEADER_EXPANDED_HEIGHT = 128f;
    private static final float HEADER_COMPACT_HEIGHT = 52f;
    private static final float POPUP_FONT_SCALE = 0.50f;
    private static final int HISTORY_RECORDS_PER_PAGE = 4;
    private static final float[] QUEUE_SLOT_X = {302f, 333f, 364f, 395f};
    private static final float QUEUE_SLOT_CENTER_Y = 38f;
    private static final float QUEUE_FRUIT_MAX_SIZE = 22f;

    /*
     * “暖色果园”主题只负责外围表现。水果仍从原来的 01.png～11.png 加载，
     * 颜色、半径、队列位置和物理坐标都不参与主题换肤。
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
    private static final Color PLAYER_SCORE_DARK =
            new Color(0.67f, 0.16f, 0.18f, 1f);
    private static final Color AI_SCORE_DARK =
            new Color(0.10f, 0.30f, 0.66f, 1f);
    private static final Color AI_BUBBLE_CARD =
            new Color(1f, 0.92f, 0.76f, 0.99f);
    private static final Color OVERLAY_DIM =
            new Color(0.25f, 0.12f, 0.07f, 0.62f);

    private final AiService aiService;
    /*
     * 内容随机和纯表现随机必须相互独立。否则多画几颗爆浆粒子就会改变
     * 后续水果队列，表现层会意外影响游戏规则与模型输入。
     */
    private final StatefulQueueRandom queueRandom =
            new StatefulQueueRandom();
    private final Random effectRandom = new Random();
    private final Random dialogueRandom = new Random();
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
    private final UiMotionController uiMotion =
            new UiMotionController();
    private final AdaptiveScoreLayoutController scoreLayout =
            new AdaptiveScoreLayoutController();

    private OrthographicCamera camera;
    private FitViewport viewport;
    private SpriteBatch batch;
    private ShapeRenderer shapes;
    private Texture uiFontTexture;
    private BitmapFont smallFont;
    private BitmapFont reactionFont;
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
    private GameSessionStore sessionStore;

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
    private boolean aiEnabled;
    private boolean aiRequestInFlight;
    private int activeDragPointer = -1;
    private float previewX;
    private float dropCooldown;
    private float stableSeconds;
    private float dangerSeconds;
    private final PhysicsDecisionClock physicsDecisionClock =
            new PhysicsDecisionClock();
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
    private int historyPageIndex;
    private float modeSwitchConfirmSeconds;
    private float autosaveRemaining;
    private float screenEnterSeconds;
    private boolean sessionDirty;
    private boolean inMemorySession;
    private boolean soloResultRecorded;
    private boolean discardNextWallDelta;
    private HorizontalSlide aiSlide;
    private ScoreSequence activeScoreSequence;
    private ScoreSequence rollingScoreSequence;
    private DuelMatch duelMatch;
    private DuelMatch.Side duelForeground = DuelMatch.Side.PLAYER;
    private HorizontalSlide duelAiSlide;
    private boolean duelAiRequestInFlight;
    private boolean duelAiArmed;
    private float duelAiArmedX;
    private long duelDecisionEpoch;
    private float duelResultHoldRemaining;
    private boolean duelResultVisible;
    private boolean duelResultRecorded;
    private long currentSessionId;
    private int duelPercentile;
    private String duelResultReason = "";
    private int duelPlayerReactionRound = -1;
    private int duelAiReactionRound = -1;
    private AiReaction aiReaction;
    private AiEmotionPulse aiEmotionPulse;
    private PendingAiEmotion pendingAiEmotion;
    private AiDialogueDirector dialogueDirector;
    private float aiBubbleOcclusion;
    private boolean aiDangerReactionArmed = true;
    private float singleStablePileRatio;
    private float duelPlayerStablePileRatio;
    private float duelAiStablePileRatio;
    private GameMode pendingNewMode;
    private AiState aiState = AiState.OBSERVING;
    private String aiDetail = "启动中";
    private AppScreen appScreen = AppScreen.HOME;
    private GameMode gameMode = GameMode.SOLO;
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
        reactionFont = createFont(0.31f);
        normalFont = createFont(0.38f);
        titleFont = createFont(0.47f);
        popupFont = createFont(POPUP_FONT_SCALE);
        glyphLayout = new GlyphLayout();
        loadFruitTextures();
        loadAudio();
        profileStore = GameProfileStore.open();
        settings = profileStore.settings();
        bestScore = modeHighScore(GameMode.SOLO);
        sessionStore = GameSessionStore.open();
        dialogueDirector = loadDialogueDirector();

        physics = new FruitPhysicsWorld();
        Gdx.input.setCatchKey(Input.Keys.BACK, true);
        Gdx.input.setInputProcessor(this);
        /*
         * 启动时只进入大厅。保存槽存在时由“继续上次”显式恢复，避免用户刚打开
         * 应用就听到上局音效或在尚未看清页面时继续推进物理状态。
         */
        appScreen = AppScreen.HOME;
        screenEnterSeconds = 0f;
    }

    @Override
    public void resize(int width, int height) {
        viewport.update(width, height, true);
    }

    @Override
    public void render() {
        float measuredDelta = Gdx.graphics.getDeltaTime();
        if (!Float.isFinite(measuredDelta) || measuredDelta < 0f) {
            measuredDelta = 0f;
        }
        float wallDelta = discardNextWallDelta
                ? Math.min(measuredDelta, 0.05f)
                : measuredDelta;
        discardNextWallDelta = false;
        float frameDelta = Math.min(wallDelta, 0.05f);
        // UI 动画可以限制单帧跨度，但权威物理不能沿用 50 ms 截断。
        // 否则设备低于 20 FPS 时，每个画面帧都会丢失墙钟时间，游戏必然慢放。
        float simulationDelta = Math.min(wallDelta, 0.25f);
        elapsedSeconds += frameDelta;
        screenEnterSeconds += frameDelta;
        uiMotion.update(frameDelta);
        updateAiReaction(frameDelta);
        historyResetConfirmSeconds = Math.max(
                0f,
                historyResetConfirmSeconds - wallDelta
        );
        modeSwitchConfirmSeconds = Math.max(
                0f,
                modeSwitchConfirmSeconds - wallDelta
        );
        if (appScreen == AppScreen.GAME
                && overlayPage == OverlayPage.NONE) {
            float gameDelta = simulationDelta * settings.gameSpeed();
            if (isSingleBoardMode()) {
                updateGame(gameDelta);
            } else {
                /*
                 * 对战回合限时使用真实前台墙钟 delta；物理仍使用 50ms cap，
                 * DuelMatch 内部还会再限制物理子步。主线程偶发卡顿因此不会把
                 * “8 秒回合”悄悄拉长，Android 从后台恢复的首帧则由 resume guard
                 * 丢弃后台经过的时间。
                 */
                updateDuelGame(wallDelta, gameDelta);
            }
            updateAutosave(wallDelta);
        }
        if (appScreen == AppScreen.GAME) {
            updateAdaptiveScoreLayout(frameDelta);
        }
        drawApp();
    }

    @Override
    public void pause() {
        /*
         * Android 不保证 onDestroy；这里的快照很小且平时已按事件/短周期落盘，
         * pause 只做最后一次同步兜底，不承担首次重型序列化。
         */
        saveCurrentSession(true);
        activeDragPointer = -1;
        uiMotion.cancelAll();
        discardNextWallDelta = true;
    }

    @Override
    public void resume() {
        // 后台停留时间不属于对战回合，也不应瞬间推进拟人物理轨迹。
        discardNextWallDelta = true;
    }

    @Override
    public void dispose() {
        saveCurrentSession(true);
        disposed = true;
        decisionEpoch += 1;
        aiRequestInFlight = false;
        aiSlide = null;
        duelAiSlide = null;
        activeDragPointer = -1;
        uiMotion.cancelAll();
        Gdx.input.setCatchKey(Input.Keys.BACK, false);
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
        if (reactionFont != null) {
            reactionFont.dispose();
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

    private AiDialogueDirector loadDialogueDirector() {
        EnumMap<AiDialogueDirector.Mood, String> texts =
                new EnumMap<>(AiDialogueDirector.Mood.class);
        try {
            for (AiDialogueDirector.Mood mood
                    : AiDialogueDirector.Mood.values()) {
                texts.put(
                        mood,
                        Gdx.files.internal(mood.assetPath())
                                .readString("UTF-8")
                );
            }
            return new AiDialogueDirector(texts, dialogueRandom);
        } catch (RuntimeException error) {
            if (Gdx.app != null) {
                Gdx.app.error(
                        "FruitMerge",
                        "AI dialogue catalog rejected; using safety fallback",
                        error
                );
            }
            return AiDialogueDirector.fallback(dialogueRandom);
        }
    }

    private void loadAudio() {
        // 音效不属于模型或规则契约；迁移首版允许资源缺失并保持游戏可玩。
        if (Gdx.files.internal("audio/merge-pop.ogg").exists()) {
            mergePopSound = Gdx.audio.newSound(
                    Gdx.files.internal("audio/merge-pop.ogg")
            );
        }
        if (Gdx.files.internal("audio/merge-soft.ogg").exists()) {
            mergeSoftSound = Gdx.audio.newSound(
                    Gdx.files.internal("audio/merge-soft.ogg")
            );
        }
        if (Gdx.files.internal("audio/score-collect.ogg").exists()) {
            scoreCollectSound = Gdx.audio.newSound(
                    Gdx.files.internal("audio/score-collect.ogg")
            );
        }
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

    private boolean isSingleBoardMode() {
        return gameMode == GameMode.SOLO
                || gameMode == GameMode.AI_DEMO;
    }

    private boolean isAiControlledSingleBoard() {
        return gameMode == GameMode.AI_DEMO;
    }

    /**
     * Keeps the score close to the player's current point of attention without feeding any visual
     * coordinate back into physics or the model. Only settled fruit contributes to the live envelope:
     * a newly released fruit starts near the spawn line and must not move the HUD by itself.
     */
    private void updateAdaptiveScoreLayout(float deltaSeconds) {
        boolean resultInProgress = currentResultVisible()
                || (gameMode == GameMode.DUEL
                && duelMatch != null
                && duelMatch.outcome()
                != DuelMatch.Outcome.IN_PROGRESS);
        float motionDelta = activeDragPointer >= 0
                ? 0f
                : deltaSeconds;
        if (resultInProgress) {
            // Finish a visible transition, but never create a new result-screen layout decision.
            // Pausing during a drag keeps touch-down and touch-up in the same coordinate frame.
            scoreLayout.update(Float.NaN, motionDelta, false);
            return;
        }
        if (isSingleBoardMode()) {
            singleStablePileRatio = sampleStablePileRatio(
                    physics,
                    singleStablePileRatio
            );
        } else if (duelMatch != null) {
            duelPlayerStablePileRatio = sampleStablePileRatio(
                    duelMatch.playerLane().physics(),
                    duelPlayerStablePileRatio
            );
            duelAiStablePileRatio = sampleStablePileRatio(
                    duelMatch.aiLane().physics(),
                    duelAiStablePileRatio
            );
        }
        scoreLayout.update(
                currentPileRatio(),
                motionDelta,
                !scoreLayoutTransitionBlocked()
        );
    }

    private void resetAdaptiveScoreLayout() {
        singleStablePileRatio = 0f;
        duelPlayerStablePileRatio = 0f;
        duelAiStablePileRatio = 0f;
        scoreLayout.resetBottom();
    }

    private void initializeAdaptiveScoreLayoutFromWorld() {
        if (isSingleBoardMode()) {
            singleStablePileRatio = sampleStablePileRatio(
                    physics,
                    0f
            );
            duelPlayerStablePileRatio = 0f;
            duelAiStablePileRatio = 0f;
        } else if (duelMatch != null) {
            singleStablePileRatio = 0f;
            duelPlayerStablePileRatio = sampleStablePileRatio(
                    duelMatch.playerLane().physics(),
                    0f
            );
            duelAiStablePileRatio = sampleStablePileRatio(
                    duelMatch.aiLane().physics(),
                    0f
            );
        }
        scoreLayout.snapForPileRatio(currentPileRatio());
    }

    private float sampleStablePileRatio(
            FruitPhysicsWorld world,
            float fallback) {
        if (world == null || world.fruits().size == 0) {
            return 0f;
        }
        float highestSettledEdge = FruitRules.FLOOR_Y;
        boolean sampled = false;
        for (FruitPhysicsWorld.FruitBody fruit : world.fruits()) {
            if (fruit.ageFrames() < 12 || !fruit.isStable()) {
                continue;
            }
            sampled = true;
            highestSettledEdge = Math.min(
                    highestSettledEdge,
                    fruit.y() - fruit.displayRadius
            );
        }
        if (!sampled) {
            return MathUtils.clamp(fallback, 0f, 1f);
        }
        float playableHeight = FruitRules.FLOOR_Y - FruitRules.SPAWN_Y;
        return MathUtils.clamp(
                (FruitRules.FLOOR_Y - highestSettledEdge) / playableHeight,
                0f,
                1f
        );
    }

    private float currentPileRatio() {
        if (gameMode == GameMode.DUEL) {
            return Math.max(
                    duelPlayerStablePileRatio,
                    duelAiStablePileRatio
            );
        }
        return singleStablePileRatio;
    }

    private boolean scoreLayoutTransitionBlocked() {
        return overlayPage != OverlayPage.NONE
                || activeDragPointer >= 0
                || uiMotion.hasActiveControl()
                || mergeCues.size > 0
                || scoreTokens.size > 0
                || duelScoreTokens.size > 0
                || activeScoreSequence != null
                || rollingScoreSequence != null
                || scoreRollQueue.size > 0
                || scorePulse > 0.05f;
    }

    private float bottomScoreVisibility() {
        return scoreLayout.bottomProgress();
    }

    private float topScoreVisibility() {
        return 1f - bottomScoreVisibility();
    }

    private float sceneOffsetY() {
        return BOTTOM_DOCK_SCENE_OFFSET_Y * bottomScoreVisibility();
    }

    private float sceneScreenY(float screenY) {
        return screenY + sceneOffsetY();
    }

    private float scoreTargetY() {
        float dockTop = scoreLayout.targetDock()
                == AdaptiveScoreLayoutController.Dock.BOTTOM
                ? SCORE_BOTTOM_DOCK_Y
                : SCORE_TOP_DOCK_Y;
        return dockTop + SCORE_TARGET_OFFSET_Y;
    }

    private float manualInputTop() {
        return MANUAL_INPUT_TOP + sceneOffsetY();
    }

    private float sceneFloorY() {
        return FruitRules.FLOOR_Y + sceneOffsetY();
    }

    private boolean scoreHudContains(float x, float y) {
        return (topScoreVisibility() > 0.08f
                && isInside(
                x,
                y,
                26f,
                SCORE_TOP_DOCK_Y,
                508f,
                SCORE_CARD_HEIGHT
        )) || (bottomScoreVisibility() > 0.08f
                && isInside(
                x,
                y,
                26f,
                SCORE_BOTTOM_DOCK_Y,
                508f,
                SCORE_CARD_HEIGHT
        ));
    }

    /**
     * Keeps the in-game "best" card scoped to the current mode. In particular, an autonomous
     * AI demo score must never be presented as the player's solo best.
     */
    private int modeHighScore(GameMode mode) {
        GameProfileStore.History history = profileStore.history();
        switch (mode) {
            case SOLO:
            case CLASSIC:
                return history.highestSoloScore();
            case AI_DEMO:
                return history.highestAiDemoScore();
            case DUEL:
                return history.highestVersusScore();
            default:
                throw new IllegalStateException("unknown game mode " + mode);
        }
    }

    private void requestStartMode(GameMode requestedMode) {
        if (requestedMode == null) {
            return;
        }
        if (sessionStore.hasSavedSession() || inMemorySession) {
            pendingNewMode = requestedMode;
            overlayPage = OverlayPage.NEW_GAME_CONFIRM;
            uiMotion.cancelAll();
            return;
        }
        startNewMode(requestedMode);
    }

    private void continueSavedSession() {
        pendingNewMode = null;
        overlayPage = OverlayPage.NONE;
        activeDragPointer = -1;
        uiMotion.cancelAll();
        if (inMemorySession && currentSessionId > 0L) {
            appScreen = AppScreen.GAME;
            screenEnterSeconds = 0f;
            if (dialogueDirector != null) {
                dialogueDirector.resetPacing();
            }
            if (gameMode != GameMode.SOLO) {
                showAiReactionImmediately(
                        AiMood.WELCOME,
                        2.0f,
                        4
                );
            }
            return;
        }
        appScreen = AppScreen.HOME;
        resumeSavedSession();
        if (appScreen != AppScreen.GAME) {
            overlayPage = OverlayPage.NONE;
            inMemorySession = false;
        }
    }

    private void startNewMode(GameMode requestedMode) {
        pendingNewMode = null;
        sessionStore.clear();
        inMemorySession = false;
        sessionDirty = false;
        overlayPage = OverlayPage.NONE;
        activeDragPointer = -1;
        uiMotion.cancelAll();
        aiReaction = null;
        aiEmotionPulse = null;
        pendingAiEmotion = null;
        aiBubbleOcclusion = 0f;
        aiDangerReactionArmed = true;
        if (dialogueDirector != null) {
            dialogueDirector.resetPacing();
        }
        gameMode = requestedMode;
        appScreen = AppScreen.GAME;
        screenEnterSeconds = 0f;
        if (requestedMode == GameMode.DUEL) {
            aiEnabled = true;
            resetDuelGame();
            showAiReactionImmediately(
                    AiMood.WELCOME,
                    2.6f,
                    4
            );
        } else {
            disposeDuelGame();
            aiEnabled = requestedMode == GameMode.AI_DEMO;
            resetGame();
            if (requestedMode == GameMode.AI_DEMO) {
                showAiReactionImmediately(
                        AiMood.WELCOME,
                        3.0f,
                        3
                );
            }
        }
        markSessionDirty();
        saveCurrentSession(true);
    }

    private void returnHomeKeepingSession() {
        saveCurrentSession(true);
        cancelPendingDecision("已保存");
        invalidateDuelDecision();
        activeDragPointer = -1;
        uiMotion.cancelAll();
        aiReaction = null;
        aiEmotionPulse = null;
        pendingAiEmotion = null;
        aiBubbleOcclusion = 0f;
        overlayPage = OverlayPage.NONE;
        appScreen = AppScreen.HOME;
        inMemorySession = true;
        screenEnterSeconds = 0f;
    }

    private void abandonCurrentSession() {
        sessionStore.clear();
        inMemorySession = false;
        sessionDirty = false;
        currentSessionId = 0L;
        pendingNewMode = null;
        cancelPendingDecision("本局已放弃");
        invalidateDuelDecision();
        if (gameMode == GameMode.DUEL) {
            disposeDuelGame();
        } else {
            physics.clear();
            queue.clear();
        }
        aiReaction = null;
        aiEmotionPulse = null;
        pendingAiEmotion = null;
        aiBubbleOcclusion = 0f;
        activeDragPointer = -1;
        uiMotion.cancelAll();
        overlayPage = OverlayPage.NONE;
        appScreen = AppScreen.HOME;
        screenEnterSeconds = 0f;
    }

    private void acknowledgeResult() {
        if (!currentResultVisible()) {
            return;
        }
        sessionStore.clear();
        inMemorySession = false;
        sessionDirty = false;
        currentSessionId = 0L;
        pendingNewMode = null;
        if (gameMode == GameMode.DUEL) {
            disposeDuelGame();
        } else {
            physics.clear();
            queue.clear();
        }
        aiReaction = null;
        aiEmotionPulse = null;
        pendingAiEmotion = null;
        aiBubbleOcclusion = 0f;
        activeDragPointer = -1;
        uiMotion.cancelAll();
        overlayPage = OverlayPage.NONE;
        appScreen = AppScreen.HOME;
        screenEnterSeconds = 0f;
    }

    private boolean currentResultVisible() {
        if (appScreen != AppScreen.GAME) {
            return false;
        }
        if (isSingleBoardMode()) {
            return !alive;
        }
        return duelMatch != null
                && duelMatch.outcome()
                != DuelMatch.Outcome.IN_PROGRESS
                && duelResultVisible;
    }

    private void openExitPrompt() {
        if (appScreen != AppScreen.GAME
                || currentResultVisible()
                || (gameMode == GameMode.DUEL
                && duelMatch != null
                && duelMatch.outcome()
                != DuelMatch.Outcome.IN_PROGRESS)) {
            return;
        }
        activeDragPointer = -1;
        uiMotion.cancelAll();
        overlayPage = OverlayPage.EXIT_CONFIRM;
        if (isSingleBoardMode()) {
            cancelPendingDecision("等待退出选择");
        } else {
            invalidateDuelDecision();
        }
        saveCurrentSession(true);
    }

    private void markSessionDirty() {
        if (inMemorySession || appScreen == AppScreen.GAME) {
            sessionDirty = true;
            /*
             * 投放/合成会把下一次落盘提前到 0.18 秒内；同一物理帧的连锁事件只
             * 触发一次双-bank 写入。无事件时仍按 1.5 秒保存运动中的刚体位置。
             */
            autosaveRemaining = Math.min(autosaveRemaining, 0.18f);
        }
    }

    private void updateAutosave(float realDelta) {
        if (!inMemorySession || currentSessionId <= 0L) {
            return;
        }
        autosaveRemaining -= Math.max(0f, realDelta);
        if (autosaveRemaining > 0f) {
            return;
        }
        /*
         * 即使本周期没有投放/合成事件，运动中的物理位置也在变化。每 1.5 秒
         * 保存一次完整小快照，才能做到进程被杀后从接近现场的位置恢复。
         */
        saveCurrentSession(true);
    }

    private void saveCurrentSession(boolean force) {
        if (sessionStore == null
                || physics == null
                || !inMemorySession
                || currentSessionId <= 0L
                || (!force && !sessionDirty)) {
            return;
        }
        try {
            GameSessionStore.Session session = isSingleBoardMode()
                    ? captureSingleSession()
                    : captureDuelSession();
            if (session == null) {
                return;
            }
            sessionStore.save(session);
            sessionDirty = false;
            autosaveRemaining = AUTOSAVE_INTERVAL_SECONDS;
        } catch (RuntimeException exception) {
            /*
             * 保存失败绝不能打断当前局。旧 bank 仍保留为可恢复版本，下一周期会再次
             * 尝试；日志只写类型，避免把设备路径或偏好内容带入输出。
             */
            sessionDirty = true;
            autosaveRemaining = 0.35f;
            if (Gdx.app != null) {
                Gdx.app.error(
                        "FruitMerge",
                        "session save failed: "
                                + exception.getClass().getSimpleName()
                );
            }
        }
    }

    private GameSessionStore.Session captureSingleSession() {
        if (queue.size != FruitRules.QUEUE_LENGTH) {
            return null;
        }
        int savedLevel = queue.first();
        GameSessionStore.SingleState state =
                new GameSessionStore.SingleState(
                        queueRandom.state(),
                        queue.toArray(),
                        savedLevel,
                        score,
                        lastScore,
                        Math.min(displayedScore, score),
                        Math.max(displayedBestScore, displayedScore),
                        stepCount,
                        currentWatermelons,
                        FruitRules.clampDropX(previewX, savedLevel),
                        FruitRules.clampDropX(previewAnchorX, savedLevel),
                        Math.max(0f, dropCooldown),
                        Math.max(0f, stableSeconds),
                        physicsDecisionClock.framesSinceDrop(),
                        Math.max(0f, dangerSeconds),
                        Math.max(0f, aiLoadingSeconds),
                        waiting,
                        alive,
                        soloResultRecorded,
                        soloPercentile,
                        physics.snapshot()
                );
        GameSessionStore.Mode mode = gameMode == GameMode.AI_DEMO
                ? GameSessionStore.Mode.AI_DEMO
                : GameSessionStore.Mode.SOLO;
        return GameSessionStore.Session.single(
                currentSessionId,
                System.currentTimeMillis(),
                mode,
                state
        );
    }

    private GameSessionStore.Session captureDuelSession() {
        if (duelMatch == null) {
            return null;
        }
        GameSessionStore.DuelState state =
                new GameSessionStore.DuelState(
                        duelMatch.snapshot(),
                        duelForeground == DuelMatch.Side.PLAYER
                                ? GameSessionStore.Side.PLAYER
                                : GameSessionStore.Side.AI,
                        Math.max(0f, duelResultHoldRemaining),
                        duelResultVisible,
                        duelResultRecorded,
                        duelPercentile,
                        duelResultReason == null ? "" : duelResultReason,
                        duelAiArmed,
                        FruitRules.clampDropX(
                                duelAiArmed
                                        ? duelAiArmedX
                                        : duelMatch.aiLane().previewX(),
                                duelMatch.currentLevel()
                        )
                );
        return GameSessionStore.Session.duel(
                currentSessionId,
                System.currentTimeMillis(),
                state
        );
    }

    private void resumeSavedSession() {
        GameSessionStore.Session session = sessionStore.load();
        if (session == null) {
            inMemorySession = false;
            return;
        }
        try {
            cancelPendingDecision("正在恢复");
            invalidateDuelDecision();
            if (dialogueDirector != null) {
                dialogueDirector.resetPacing();
            }
            currentSessionId = session.sessionId();
            if (session.mode() == GameSessionStore.Mode.DUEL) {
                restoreDuelSession(session.duel());
            } else {
                restoreSingleSession(
                        session.single(),
                        session.mode()
                                == GameSessionStore.Mode.AI_DEMO
                                ? GameMode.AI_DEMO
                                : GameMode.SOLO
                );
            }
            appScreen = AppScreen.GAME;
            overlayPage = OverlayPage.NONE;
            activeDragPointer = -1;
            inMemorySession = true;
            sessionDirty = false;
            autosaveRemaining = AUTOSAVE_INTERVAL_SECONDS;
            screenEnterSeconds = 0f;
            uiMotion.cancelAll();
            ensureRestoredResultRecorded();
        } catch (RuntimeException exception) {
            sessionStore.clear();
            inMemorySession = false;
            currentSessionId = 0L;
            if (Gdx.app != null) {
                Gdx.app.error(
                        "FruitMerge",
                        "session restore rejected: "
                                + exception.getClass().getSimpleName()
                );
            }
        }
    }

    private void ensureRestoredResultRecorded() {
        if (isSingleBoardMode()) {
            if (alive) {
                return;
            }
            if (soloResultRecorded
                    && profileStore.hasRecordedSession(
                            currentSessionKey())) {
                return;
            }
            soloPercentile = profileStore.resultPercentile(score);
            if (gameMode == GameMode.SOLO) {
                profileStore.recordSoloGame(
                        currentSessionKey(),
                        score,
                        currentWatermelons,
                        stepCount
                );
            } else {
                profileStore.recordAiDemoGame(
                        currentSessionKey(),
                        score,
                        currentWatermelons,
                        stepCount
                );
            }
            soloResultRecorded = true;
        } else {
            if (duelMatch == null
                    || duelMatch.outcome()
                    == DuelMatch.Outcome.IN_PROGRESS
                    || (duelResultRecorded
                    && profileStore.hasRecordedSession(
                            currentSessionKey()))) {
                return;
            }
            DuelMatch.Lane player = duelMatch.playerLane();
            DuelMatch.Lane ai = duelMatch.aiLane();
            duelPercentile = profileStore.resultPercentile(
                    player.score()
            );
            profileStore.recordVersusGame(
                    currentSessionKey(),
                    player.score(),
                    ai.score(),
                    player.watermelonCount(),
                    ai.watermelonCount(),
                    player.stepCount(),
                    toStoredBattleResult(duelMatch.outcome())
            );
            duelResultRecorded = true;
        }
        markSessionDirty();
        saveCurrentSession(true);
    }

    private void restoreSingleSession(
            GameSessionStore.SingleState state,
            GameMode restoredMode) {
        disposeDuelGame();
        physics.restore(state.physics());
        queue.clear();
        for (int level : state.queueLevels()) {
            queue.add(level);
        }
        if (state.queueRandomState()
                != GameSessionStore.NO_RANDOM_STATE) {
            queueRandom.restoreState(state.queueRandomState());
        } else {
            queueRandom.setSeed(
                    System.nanoTime() ^ state.score() ^ state.stepCount()
            );
        }
        gameMode = restoredMode;
        aiEnabled = restoredMode == GameMode.AI_DEMO;
        currentLevel = state.currentLevel();
        score = state.score();
        lastScore = state.lastScore();
        bestScore = modeHighScore(restoredMode);
        /*
         * 浮分、飞行 token 与滚分队列是瞬态表现，不写入存档。恢复时直接让 HUD
         * 追平逻辑分数，避免恰好在吸分动画中保存后出现永久少显示的分数。
         */
        displayedScore = score;
        displayedBestScore = Math.max(bestScore, score);
        stepCount = state.stepCount();
        currentWatermelons = state.watermelonCount();
        previewX = state.previewX();
        previewAnchorX = state.previewAnchorX();
        dropCooldown = aiEnabled ? 0f : state.dropCooldownSeconds();
        stableSeconds = state.stableSeconds();
        dangerSeconds = state.dangerSeconds();
        physicsDecisionClock.restore(
                stableSeconds,
                dangerSeconds,
                state.physicsFramesSinceDrop()
        );
        aiDangerReactionArmed = dangerSeconds < 0.9f;
        aiLoadingSeconds = state.aiLoadingSeconds();
        waiting = state.waiting();
        alive = state.alive();
        soloResultRecorded = state.resultRecorded();
        soloPercentile = state.resultPercentile();
        decisionEpoch += 1L;
        aiRequestInFlight = false;
        aiSlide = null;
        aiState = alive
                ? (aiEnabled ? AiState.OBSERVING : AiState.MANUAL)
                : AiState.GAME_OVER;
        aiDetail = alive
                ? (aiEnabled ? "继续观察局面" : "拖动水果，松手投放")
                : "等待确认结算";
        clearTransientPresentation();
        initializeAdaptiveScoreLayoutFromWorld();
        if (aiEnabled && alive) {
            showAiReactionImmediately(
                    AiMood.WELCOME,
                    2.0f,
                    4
            );
        }
    }

    private void restoreDuelSession(GameSessionStore.DuelState state) {
        disposeDuelGame();
        DuelMatch.Snapshot snapshot = state.match();
        duelMatch = new DuelMatch(
                1L,
                snapshot.roundDurationSeconds(),
                snapshot.nextRoundDelaySeconds()
        );
        duelMatch.restore(snapshot);
        gameMode = GameMode.DUEL;
        aiEnabled = true;
        duelForeground = state.foreground()
                == GameSessionStore.Side.PLAYER
                ? DuelMatch.Side.PLAYER
                : DuelMatch.Side.AI;
        duelResultHoldRemaining =
                state.resultHoldRemainingSeconds();
        duelResultVisible = state.resultVisible();
        duelResultRecorded = state.resultRecorded();
        duelPercentile = state.resultPercentile();
        duelResultReason = state.resultReason();
        duelAiArmed = state.aiArmed()
                && duelMatch.outcome()
                == DuelMatch.Outcome.IN_PROGRESS
                && !duelMatch.aiLane().submittedThisRound();
        duelAiArmedX = state.aiArmedX();
        duelAiRequestInFlight = false;
        duelAiSlide = null;
        duelPlayerReactionRound = -1;
        duelAiReactionRound = -1;
        activeDragPointer = -1;
        clearTransientPresentation();
        initializeAdaptiveScoreLayoutFromWorld();
        showAiReactionImmediately(
                duelAiArmed ? AiMood.READY : AiMood.WELCOME,
                2.0f,
                4
        );
    }

    private void clearTransientPresentation() {
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
        scoreRollElapsed = 0f;
        scoreRollDuration = 0f;
        scoreRollStart = displayedScore;
        scoreRollTarget = displayedScore;
        nextScoreSequenceId = 0;
        aiReaction = null;
        aiEmotionPulse = null;
        pendingAiEmotion = null;
        aiBubbleOcclusion = 0f;
    }

    private long createSessionId() {
        long value = System.nanoTime()
                ^ (System.currentTimeMillis() << 17)
                ^ effectRandom.nextLong();
        value &= Long.MAX_VALUE;
        return value == 0L ? 1L : value;
    }

    private String currentSessionKey() {
        return Long.toString(currentSessionId);
    }

    private void resetGame() {
        decisionEpoch += 1;
        aiRequestInFlight = false;
        aiSlide = null;
        activeDragPointer = -1;
        physics.clear();
        resetAdaptiveScoreLayout();
        queue.clear();
        fillQueue();
        bestScore = modeHighScore(gameMode);
        score = 0;
        displayedScore = 0;
        displayedBestScore = bestScore;
        lastScore = 0;
        stepCount = 0;
        stableSeconds = 0f;
        dangerSeconds = 0f;
        physicsDecisionClock.resetGame();
        aiDangerReactionArmed = true;
        aiLoadingSeconds = 0f;
        dropCooldown = aiEnabled ? 0f : 0.18f;
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
        currentSessionId = createSessionId();
        autosaveRemaining = AUTOSAVE_INTERVAL_SECONDS;
        sessionDirty = true;
        inMemorySession = true;
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
        resetAdaptiveScoreLayout();
        duelForeground = DuelMatch.Side.PLAYER;
        duelResultHoldRemaining = 0f;
        duelResultVisible = false;
        duelResultRecorded = false;
        duelPercentile = 0;
        duelResultReason = "";
        duelPlayerReactionRound = -1;
        duelAiReactionRound = -1;
        currentSessionId = createSessionId();
        duelAiArmed = false;
        duelAiArmedX = FruitRules.BOARD_WIDTH / 2f;
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
        autosaveRemaining = AUTOSAVE_INTERVAL_SECONDS;
        sessionDirty = true;
        inMemorySession = true;
    }

    private void disposeDuelGame() {
        invalidateDuelDecision();
        if (duelMatch != null) {
            duelMatch.dispose();
            duelMatch = null;
        }
        duelScoreTokens.clear();
        duelAiArmed = false;
    }

    private void invalidateDuelDecision() {
        duelDecisionEpoch += 1L;
        duelAiRequestInFlight = false;
        duelAiSlide = null;
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
            duelAiArmed = false;
            activeDragPointer = -1;
            markSessionDirty();
        }

        updateDuelAi(realDelta);
        if (duelMatch.roundOpen()
                && duelMatch.roundRemainingSeconds() <= 0f) {
            if (!duelMatch.playerLane().submittedThisRound()
                    && !duelMatch.aiLane().submittedThisRound()
                    && duelAiArmed) {
                float playerX = duelMatch.playerLane().previewX();
                boolean dropped = duelMatch.timeoutBoth(
                        playerX,
                        duelAiArmedX
                );
                activeDragPointer = -1;
                duelAiArmed = false;
                if (dropped) {
                    spawnDuelDropFeedback(DuelMatch.Side.PLAYER);
                    spawnDuelDropFeedback(DuelMatch.Side.AI);
                    markSessionDirty();
                }
                invalidateDuelDecision();
                return;
            }
            if (!duelMatch.playerLane().submittedThisRound()) {
                boolean dropped = duelMatch.timeoutPlayer();
                activeDragPointer = -1;
                if (dropped) {
                    spawnDuelDropFeedback(DuelMatch.Side.PLAYER);
                    markSessionDirty();
                }
            }
            if (duelAiArmed
                    && !duelMatch.aiLane().submittedThisRound()) {
                dropDuelAiAt(duelAiArmedX);
                duelAiArmed = false;
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
                    markSessionDirty();
                }
                invalidateDuelDecision();
            }
        }
    }

    private void consumeDuelMergeEvents() {
        boolean aiSceneChanged = false;
        int playerMergeCount = 0;
        int aiMergeCount = 0;
        int maxPlayerMergeLevel = 0;
        int maxAiMergeLevel = 0;
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
                playerMergeCount += 1;
                maxPlayerMergeLevel = Math.max(
                        maxPlayerMergeLevel,
                        event.level()
                );
                vibrateIf(
                        settings.vibrateOnMerge(),
                        event.level() >= 8 ? 36 : 18
                );
            } else {
                aiMergeCount += 1;
                maxAiMergeLevel = Math.max(
                        maxAiMergeLevel,
                        event.level()
                );
            }
            scorePulse = 1f;
            markSessionDirty();
        }
        boolean impressivePlayerMerge = playerMergeCount >= 2
                || maxPlayerMergeLevel >= 6;
        int reactionRound = duelMatch.roundIndex();
        if (impressivePlayerMerge
                && duelPlayerReactionRound != reactionRound) {
            duelPlayerReactionRound = reactionRound;
            showAiReaction(
                    AiMood.SURPRISED,
                    2.4f,
                    playerMergeCount >= 4 || maxPlayerMergeLevel >= 9
                            ? 7
                            : 6
            );
        } else if (aiMergeCount > 0
                && duelAiReactionRound != reactionRound) {
            duelAiReactionRound = reactionRound;
            boolean majorAiMerge = aiMergeCount >= 2
                    || maxAiMergeLevel >= 7;
            showAiReaction(
                    AiMood.HAPPY,
                    majorAiMerge ? 2.2f : 1.8f,
                    majorAiMerge ? 6 : 3
            );
        }
        if (aiSceneChanged) {
            if (duelAiArmed
                    && !duelMatch.aiLane().submittedThisRound()) {
                duelAiArmed = false;
                showAiReaction(
                        AiMood.THINKING,
                        1.7f,
                        4
                );
            }
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
        if (duelAiSlide != null) {
            updateDuelAiSlide(realDelta);
            return;
        }
        if (duelAiArmed) {
            return;
        }
        if (remaining <= 0.65f) {
            AiDecision fallback = fallbackDecision(
                    createDuelSnapshot(duelMatch.aiLane())
            );
            invalidateDuelDecision();
            startDuelAiSlide(FruitRules.actionDropX(
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
            startDuelAiSlide(FruitRules.actionDropX(
                    fallback.actionIndex,
                    duelMatch.currentLevel()
            ));
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
        showAiReaction(
                AiMood.THINKING,
                1.8f,
                2
        );
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
            if (decision == null) {
                showAiReaction(
                        AiMood.HESITATING,
                        1.8f,
                        3
                );
            }
            startDuelAiSlide(FruitRules.actionDropX(
                    selected.actionIndex,
                    match.currentLevel()
            ));
        });
    }

    private GameSnapshot createDuelSnapshot(DuelMatch.Lane lane) {
        List<GameSnapshot.FruitSnapshot> fruitSnapshots = new ArrayList<>();
        for (FruitPhysicsWorld.FruitBody fruit : lane.physics().fruits()) {
            fruitSnapshots.add(new GameSnapshot.FruitSnapshot(
                    fruit.slot,
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
                MathUtils.clamp(
                        lane.dangerSeconds() / FruitRules.DANGER_SECONDS,
                        0f,
                        1f
                ),
                lane.physics().hasFruitTopAboveSpawnLine(),
                duelMatch.queueSnapshot().toArray(),
                fruitSnapshots
        );
    }

    private void beginDuelResult() {
        invalidateDuelDecision();
        duelAiArmed = false;
        activeDragPointer = -1;
        uiMotion.cancelAll();
        aiReaction = null;
        aiEmotionPulse = null;
        pendingAiEmotion = null;
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
                    currentSessionKey(),
                    player.score(),
                    ai.score(),
                    player.watermelonCount(),
                    ai.watermelonCount(),
                    player.stepCount(),
                    toStoredBattleResult(duelMatch.outcome())
            );
        }
        markSessionDirty();
        saveCurrentSession(true);
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
                    ? 170f : 390f;
            token.x = MathUtils.lerp(
                    token.originX,
                    targetX,
                    accelerated
            );
            token.y = MathUtils.lerp(
                    token.originY - 30f + sceneOffsetY(),
                    scoreTargetY(),
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
     * <p>它不创建或移动物理水果，只排入三次带间隔的视觉合成事件，便于自动
     * 截图检查爆浆、立体分值、吸附和计分板反馈。</p>
     */
    public void startPresentationShowcase() {
        if (physics == null || disposed) {
            return;
        }
        if (appScreen != AppScreen.GAME || !isSingleBoardMode()) {
            startNewMode(GameMode.SOLO);
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
        if ("home".equals(normalized)) {
            appScreen = AppScreen.HOME;
            overlayPage = OverlayPage.NONE;
            return;
        }
        if ("settings".equals(normalized)) {
            appScreen = AppScreen.HOME;
            overlayPage = OverlayPage.SETTINGS;
            return;
        }
        if ("history".equals(normalized)) {
            profileStore.recordSoloGame(
                    "preview:history:solo",
                    2_680,
                    1,
                    54,
                    1_753_884_600_000L
            );
            profileStore.recordVersusGame(
                    "preview:history:duel",
                    3_420,
                    3_180,
                    2,
                    1,
                    67,
                    GameProfileStore.BattleResult.WIN,
                    1_753_888_200_000L
            );
            profileStore.recordAiDemoGame(
                    "preview:history:demo",
                    4_150,
                    3,
                    78,
                    1_753_891_800_000L
            );
            appScreen = AppScreen.HOME;
            overlayPage = OverlayPage.HISTORY;
            historyPageIndex = 0;
            return;
        }
        if ("score-low".equals(normalized)
                || "score-high".equals(normalized)) {
            startNewMode(GameMode.SOLO);
            boolean highPile = "score-high".equals(normalized);
            FruitPhysicsWorld.FruitState[] states = highPile
                    ? new FruitPhysicsWorld.FruitState[]{
                    new FruitPhysicsWorld.FruitState(
                            1,
                            11,
                            FruitRules.displayRadius(11),
                            FruitRules.mergedPhysicsRadius(11),
                            280f,
                            945f,
                            0f,
                            0f,
                            0f,
                            0f,
                            120
                    ),
                    new FruitPhysicsWorld.FruitState(
                            2,
                            10,
                            FruitRules.displayRadius(10),
                            FruitRules.mergedPhysicsRadius(10),
                            280f,
                            671f,
                            0f,
                            0f,
                            0f,
                            0f,
                            120
                    )
            }
                    : new FruitPhysicsWorld.FruitState[]{
                    new FruitPhysicsWorld.FruitState(
                            1,
                            4,
                            FruitRules.displayRadius(4),
                            FruitRules.mergedPhysicsRadius(4),
                            210f,
                            1055f,
                            0f,
                            0f,
                            0f,
                            0f,
                            120
                    ),
                    new FruitPhysicsWorld.FruitState(
                            2,
                            5,
                            FruitRules.displayRadius(5),
                            FruitRules.mergedPhysicsRadius(5),
                            350f,
                            1043f,
                            0f,
                            0f,
                            0f,
                            0f,
                            120
                    )
            };
            physics.restore(new FruitPhysicsWorld.Snapshot(3, 0f, states));
            initializeAdaptiveScoreLayoutFromWorld();
            return;
        }
        if ("solo".equals(normalized)) {
            startNewMode(GameMode.SOLO);
            return;
        }
        if ("demo".equals(normalized)) {
            startNewMode(GameMode.AI_DEMO);
            return;
        }
        if ("reaction".equals(normalized)
                || "reaction-overlap".equals(normalized)) {
            startNewMode(GameMode.AI_DEMO);
            if ("reaction-overlap".equals(normalized)) {
                physics.addDroppedFruit(4, 280f, 270f);
                showAiReactionImmediately(
                        AiMood.WORRIED,
                        2.8f,
                        7
                );
                // Deterministic visual QA: begin at the fully occluded endpoint while the fruit
                // falls through the card, instead of depending on capture-frame timing.
                aiBubbleOcclusion = 1f;
            }
            return;
        }
        if ("exit".equals(normalized)) {
            startNewMode(GameMode.SOLO);
            openExitPrompt();
            return;
        }
        if ("new".equals(normalized)) {
            startNewMode(GameMode.SOLO);
            appScreen = AppScreen.HOME;
            requestStartMode(GameMode.DUEL);
            return;
        }
        if ("result".equals(normalized)) {
            startNewMode(GameMode.SOLO);
            score = 3140;
            displayedScore = score;
            bestScore = Math.max(bestScore, score);
            displayedBestScore = bestScore;
            soloPercentile = profileStore.resultPercentile(score);
            soloResultRecorded = true;
            alive = false;
            return;
        }
        if (!"duel".equals(normalized)) {
            return;
        }
        gameMode = GameMode.DUEL;
        appScreen = AppScreen.GAME;
        overlayPage = OverlayPage.NONE;
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

        physics.step(delta, (stable, overDangerLine) -> {
            physicsDecisionClock.afterFrame(stable, overDangerLine);
            // 决策边界只用于抓取模型快照，不能暂停正在显示的物理世界。
            return !physicsDecisionClock.failed();
        });
        consumeMergeEvents();
        updateEffects(delta);
        stableSeconds = physicsDecisionClock.stableSeconds();
        dangerSeconds = physicsDecisionClock.dangerSeconds();
        updateDangerTimer();
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
        int maxMergedLevel = 0;
        ScoreSequence reactionSequence = null;
        if (events.size > 0) {
            // 合成会改变图拓扑，即使新刚体瞬时速度为零，也必须重新累计稳定窗口，
            // 并让已经发出的异步决策失效。
            physicsDecisionClock.resetStable();
            stableSeconds = 0f;
            cancelPendingDecision("局面变化");
        }
        for (FruitPhysicsWorld.MergeEvent event : events) {
            int visualLevel = event.level > 0 ? event.level : event.sourceLevel;
            maxMergedLevel = Math.max(maxMergedLevel, visualLevel);
            ScoreSequence sequence = ensureActiveScoreSequence();
            reactionSequence = sequence;
            lastScore = score;
            score += event.scoreDelta;
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
                    visualLevel,
                    event.scoreDelta,
                    cascadeDepth,
                    delay,
                    sequence
            ));
            markSessionDirty();
        }
        if (gameMode == GameMode.AI_DEMO
                && reactionSequence != null
                && !reactionSequence.aiReactionOffered) {
            reactionSequence.aiReactionOffered = true;
            boolean majorMerge = events.size >= 2 || maxMergedLevel >= 8;
            showAiReaction(
                    AiMood.HAPPY,
                    majorMerge ? 2.3f : 1.8f,
                    majorMerge ? 6 : 3
            );
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

    private void updateDangerTimer() {
        boolean danger = physicsDecisionClock.dangerFrames() > 0;
        if (!danger) {
            aiDangerReactionArmed = true;
        }
        if (gameMode == GameMode.AI_DEMO
                && aiDangerReactionArmed
                && dangerSeconds >= 0.9f) {
            showAiReaction(
                    AiMood.WORRIED,
                    1.8f,
                    7
            );
            aiDangerReactionArmed = false;
        }
        if (physicsDecisionClock.failed()) {
            alive = false;
            activeDragPointer = -1;
            uiMotion.cancelAll();
            cancelPendingDecision("游戏结束");
            finishScorePresentationAtGameOver();
            aiState = AiState.GAME_OVER;
            aiDetail = "等待确认结算";
            if (!soloResultRecorded) {
                soloResultRecorded = true;
                soloPercentile = profileStore.resultPercentile(score);
                if (gameMode == GameMode.SOLO) {
                    profileStore.recordSoloGame(
                            currentSessionKey(),
                            score,
                            currentWatermelons,
                            stepCount
                    );
                    bestScore = modeHighScore(GameMode.SOLO);
                } else {
                    profileStore.recordAiDemoGame(
                            currentSessionKey(),
                            score,
                            currentWatermelons,
                            stepCount
                    );
                    bestScore = modeHighScore(GameMode.AI_DEMO);
                }
                settings = profileStore.settings();
            }
            markSessionDirty();
            saveCurrentSession(true);
        }
    }

    private void updateAi(float delta) {
        if (aiSlide != null) {
            updateAiSlide(delta);
            return;
        }
        if (aiRequestInFlight) {
            aiState = AiState.THINKING;
            return;
        }
        if (!modelDecisionBoundaryReady()) {
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
            applyAiDecision(fallbackDecision(snapshot));
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
                        applyAiDecision(fallbackDecision(createSnapshot()));
                    } else {
                        applyAiDecision(decision);
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
                    applyAiDecision(fallbackDecision(createSnapshot()));
                });
            }
        });
    }

    private GameSnapshot createSnapshot() {
        List<GameSnapshot.FruitSnapshot> fruitSnapshots = new ArrayList<>();
        for (FruitPhysicsWorld.FruitBody fruit : physics.fruits()) {
            fruitSnapshots.add(new GameSnapshot.FruitSnapshot(
                    fruit.slot,
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
                MathUtils.clamp(
                        dangerSeconds / FruitRules.DANGER_SECONDS,
                        0f,
                        1f
                ),
                physics.hasFruitTopAboveSpawnLine(),
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

    private void applyAiDecision(AiDecision decision) {
        float targetX = FruitRules.clampDropX(
                FruitRules.actionDropX(decision.actionIndex, currentLevel),
                currentLevel
        );
        aiState = AiState.COMMITTING;
        if (Math.abs(targetX - previewX) <= 0.25f) {
            previewX = targetX;
            previewAnchorX = targetX;
            aiDetail = "投放";
            dropCurrent(targetX);
            return;
        }
        aiSlide = new HorizontalSlide(previewX, targetX);
        aiDetail = "移动至投放位置";
    }

    private void updateAiSlide(float delta) {
        HorizontalSlide slide = aiSlide;
        if (slide == null) {
            return;
        }
        if (!alive || !waiting || !aiEnabled) {
            aiSlide = null;
            return;
        }
        previewX = FruitRules.clampDropX(
                slide.advance(delta),
                currentLevel
        );
        if (!slide.finished()) {
            return;
        }
        float targetX = FruitRules.clampDropX(
                slide.targetX,
                currentLevel
        );
        aiSlide = null;
        previewX = targetX;
        previewAnchorX = targetX;
        aiState = AiState.COMMITTING;
        aiDetail = "投放";
        dropCurrent(targetX);
    }


    private void cancelPendingDecision(String detail) {
        if (!aiRequestInFlight && aiSlide == null) {
            return;
        }
        if (waiting) {
            previewAnchorX = FruitRules.clampDropX(
                    previewX,
                    currentLevel
            );
        }
        decisionEpoch += 1;
        aiRequestInFlight = false;
        aiSlide = null;
        aiState = aiEnabled ? AiState.OBSERVING : AiState.MANUAL;
        aiDetail = detail;
    }

    private void dropCurrent(float x) {
        if (!canDropCurrent()) {
            return;
        }
        aiSlide = null;
        x = FruitRules.clampDropX(x, currentLevel);
        /*
         * 表现序列在第一笔真实合成发生时懒创建。手动玩家可以在上一颗仍运动时
         * 连续投放，因此这里绝不能覆盖尚未释放的合成序列；连续运动窗口内产生的
         * 合成会自然聚合为同一组浮分，稳定后再统一吸入 HUD。
         */
        float y = FruitRules.SPAWN_Y;
        physics.addDroppedFruit(currentLevel, x, y);
        spawnDropEffect(x, y, currentLevel);
        vibrateIf(settings.vibrateOnDrop(), 10);

        if (queue.size > 0) {
            queue.removeIndex(0);
        }
        fillQueue();
        stepCount += 1;
        dropCooldown = aiEnabled ? 0f : MANUAL_DROP_COOLDOWN_SECONDS;
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
        physicsDecisionClock.resetForDrop();
        decisionEpoch += 1;
        aiRequestInFlight = false;
        aiState = aiEnabled ? AiState.OBSERVING : AiState.MANUAL;
        previewAnchorX = previewX;
        markSessionDirty();
    }

    private boolean canDropCurrent() {
        return aiEnabled ? canAiDropCurrent() : canManualDropCurrent();
    }

    private boolean isBaseDropReady() {
        return alive && waiting && dropCooldown <= 0f;
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
                && modelDecisionBoundaryReady();
    }

    private boolean modelDecisionBoundaryReady() {
        return (stepCount == 0 && physics.fruits().size == 0)
                || physicsDecisionClock.decisionReady();
    }

    private float previewY(int level) {
        // 与物理水果的实际生成中心一致，让拖动到下落只发生状态切换而不瞬移。
        return FruitRules.SPAWN_Y;
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
         * 游戏结束后物理会停止推进，因此不能再等待 physics.isStable()。
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
                    token.flightStartY + sceneOffsetY(),
                    scoreTargetY(),
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
                        4f,
                        false
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
        float targetY = scoreTargetY();
        rings.add(new Ring(
                SCORE_TARGET_X,
                targetY,
                13f,
                150f,
                0.42f,
                color,
                3.6f,
                false
        ));
        for (int index = 0; index < 15; index++) {
            float angle = effectRandom.nextFloat() * MathUtils.PI2;
            float speed = 50f + effectRandom.nextFloat() * 105f;
            particles.add(new Particle(
                    SCORE_TARGET_X,
                    targetY,
                    MathUtils.cos(angle) * speed,
                    MathUtils.sin(angle) * speed,
                    0.30f + effectRandom.nextFloat() * 0.22f,
                    color,
                    2.2f + effectRandom.nextFloat() * 2.8f,
                    80f,
                    1.2f,
                    false
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
                2.2f,
                true
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
                    0.8f,
                    true
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
                4.2f,
                true
        ));
        rings.add(new Ring(
                x,
                y,
                6f,
                175f * intensity,
                0.34f,
                Color.WHITE,
                2.7f,
                true
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
                    0.55f,
                    true
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

    private void updateAiReaction(float delta) {
        if (dialogueDirector != null) {
            dialogueDirector.update(delta);
        }
        if (aiReaction != null) {
            aiReaction.age += delta;
            if (aiReaction.age >= aiReaction.duration) {
                aiReaction = null;
            }
        }
        if (aiEmotionPulse != null) {
            aiEmotionPulse.age += delta;
            if (aiEmotionPulse.age >= aiEmotionPulse.duration) {
                aiEmotionPulse = null;
            }
        }
        if (pendingAiEmotion != null) {
            pendingAiEmotion.age += delta;
            if (pendingAiEmotion.age >= URGENT_EMOTION_TTL_SECONDS
                    || appScreen != AppScreen.GAME
                    || currentResultVisible()) {
                pendingAiEmotion = null;
            } else {
                boolean mayPreemptSpeech = aiReaction != null
                        && pendingAiEmotion.priority >= 7
                        && aiReaction.age >= 0.85f;
                if ((aiReaction == null || mayPreemptSpeech)
                        && aiEmotionPulse == null
                        && dialogueDirector != null) {
                    AiDialogueDirector.Mood pendingMood =
                            AiDialogueDirector.Mood.valueOf(
                                    pendingAiEmotion.mood.name()
                            );
                    String emoticon =
                            dialogueDirector.offerDeferredUrgentEmoticon(
                                    pendingMood,
                                    pendingAiEmotion.priority
                            );
                    if (emoticon != null) {
                        if (mayPreemptSpeech) {
                            aiReaction = null;
                        }
                        pulseAiEmotion(
                                pendingAiEmotion.mood,
                                emoticon,
                                pendingAiEmotion.priority
                        );
                        pendingAiEmotion = null;
                    }
                }
            }
        }
        float targetOcclusion = (aiReaction != null
                || aiEmotionPulse != null)
                && appScreen == AppScreen.GAME
                && reactionOverlapsVisibleFruit()
                ? 1f
                : 0f;
        float response = targetOcclusion > aiBubbleOcclusion ? 9f : 5f;
        float blend = 1f - (float) Math.exp(-response * delta);
        aiBubbleOcclusion = MathUtils.lerp(
                aiBubbleOcclusion,
                targetOcclusion,
                MathUtils.clamp(blend, 0f, 1f)
        );
    }

    private boolean showAiReaction(
            AiMood mood,
            float duration,
            int priority) {
        return showAiReaction(mood, duration, priority, false);
    }

    private boolean showAiReactionImmediately(
            AiMood mood,
            float duration,
            int priority) {
        return showAiReaction(mood, duration, priority, true);
    }

    private boolean showAiReaction(
            AiMood mood,
            float duration,
            int priority,
            boolean force) {
        if (gameMode == GameMode.SOLO) {
            return false;
        }
        int activePriority = aiReaction == null ? -1 : aiReaction.priority;
        float activeAge = aiReaction == null ? -1f : aiReaction.age;
        AiDialogueDirector.Mood directorMood =
                AiDialogueDirector.Mood.valueOf(mood.name());
        AiDialogueDirector.Line line = dialogueDirector == null
                ? null
                : dialogueDirector.offer(
                        directorMood,
                        duration,
                        priority,
                        force,
                        activePriority,
                        activeAge
                );
        if (line == null) {
            // Keep the face stable while a sentence is still on screen. Silent reactions are a
            // fallback for otherwise quiet moments, not an animation layered over active speech.
            if (aiReaction != null) {
                queueUrgentEmotion(mood, priority);
                return false;
            }
            String pulseEmoticon = dialogueDirector == null
                    ? null
                    : dialogueDirector.offerEmoticon(
                            directorMood,
                            priority
                    );
            if (pulseEmoticon != null) {
                pulseAiEmotion(mood, pulseEmoticon, priority);
            } else {
                queueUrgentEmotion(mood, priority);
            }
            return false;
        }
        // The full speech bubble already carries its own face. Clearing a stale standalone pulse
        // prevents a normal merge reaction from briefly replacing the newly spoken expression.
        aiEmotionPulse = null;
        pendingAiEmotion = null;
        aiReaction = new AiReaction(
                line.text(),
                line.emoticon(),
                mood,
                line.duration(),
                line.priority()
        );
        return true;
    }

    private void queueUrgentEmotion(AiMood mood, int priority) {
        if (priority < 6) {
            return;
        }
        if (pendingAiEmotion == null
                || priority > pendingAiEmotion.priority) {
            pendingAiEmotion = new PendingAiEmotion(mood, priority);
        }
    }

    private void pulseAiEmotion(
            AiMood mood,
            String emoticon,
            int priority) {
        if (aiEmotionPulse != null
                && aiEmotionPulse.age < 0.22f
                && priority < aiEmotionPulse.priority) {
            return;
        }
        aiEmotionPulse = new AiEmotionPulse(
                emoticon,
                mood,
                MathUtils.clamp(0.66f + priority * 0.035f, 0.66f, 0.94f),
                priority
        );
    }

    private void drawAiReaction() {
        if ((aiReaction == null && aiEmotionPulse == null)
                || appScreen != AppScreen.GAME
                || currentResultVisible()
                || (gameMode == GameMode.DUEL
                && duelMatch != null
                && duelMatch.outcome()
                != DuelMatch.Outcome.IN_PROGRESS)) {
            return;
        }
        if (aiReaction == null) {
            drawAiEmotionPulseChip();
            return;
        }
        float enter = MathUtils.clamp(aiReaction.age / 0.18f, 0f, 1f);
        float leave = MathUtils.clamp(
                (aiReaction.duration - aiReaction.age) / 0.24f,
                0f,
                1f
        );
        float alpha = Math.min(enter, leave);
        float pop = 0.84f
                + (1f - (float) Math.pow(1f - enter, 3f)) * 0.16f;
        float bob = MathUtils.sin(elapsedSeconds * 3.6f) * 1.25f;
        float width = 512f * pop;
        float height = 90f * pop;
        float left = 24f + (512f - width) * 0.5f;
        float top = 270f + (90f - height) * 0.5f + bob;
        boolean pulsing = aiEmotionPulse != null;
        AiMood faceMood = pulsing
                ? aiEmotionPulse.mood
                : aiReaction.mood;
        String faceEmoticon = pulsing
                ? aiEmotionPulse.emoticon
                : aiReaction.emoticon;
        float facePulse = pulsing
                ? 1f + 0.10f * (float) Math.sin(
                MathUtils.clamp(
                        aiEmotionPulse.age / aiEmotionPulse.duration,
                        0f,
                        1f
                ) * MathUtils.PI)
                : 1f;
        Color accent = reactionAccent(aiReaction.mood);
        Color faceAccent = reactionAccent(faceMood);
        Color foreground = reactionForeground(aiReaction.mood);
        Color faceForeground = reactionForeground(faceMood);
        float panelAlpha = alpha * (1f - 0.42f * aiBubbleOcclusion);
        float contentAlpha = alpha * (1f - 0.06f * aiBubbleOcclusion);

        // ShapeRenderer does not own blend state. Re-enable it here because several sprite/effect
        // passes run between drawGame()'s initial setup and this late overlay.
        Gdx.gl.glEnable(GL20.GL_BLEND);
        Gdx.gl.glBlendFunc(
                GL20.GL_SRC_ALPHA,
                GL20.GL_ONE_MINUS_SRC_ALPHA
        );
        shapes.begin(ShapeRenderer.ShapeType.Filled);
        scratchColor.set(CARD_SHADOW);
        scratchColor.a = 0.34f * panelAlpha;
        roundedRectTop(
                left,
                top + 7f,
                width,
                height,
                24f * pop,
                scratchColor
        );
        scratchColor.set(accent);
        scratchColor.a = 0.96f * panelAlpha;
        roundedRectTop(
                left,
                top,
                width,
                height,
                24f * pop,
                scratchColor
        );
        scratchColor.set(AI_BUBBLE_CARD);
        scratchColor.a = 0.98f * panelAlpha;
        roundedRectTop(
                left + 3f,
                top + 3f,
                width - 6f,
                height - 6f,
                21f * pop,
                scratchColor
        );
        scratchColor.set(faceAccent);
        scratchColor.lerp(AI_BUBBLE_CARD, 0.68f);
        scratchColor.a = 0.98f * panelAlpha;
        roundedRectTop(
                left + 12f,
                top + 13f,
                108f * pop,
                height - 26f,
                17f * pop,
                scratchColor
        );
        shapes.end();

        batch.begin();
        normalFont.setColor(
                faceForeground.r,
                faceForeground.g,
                faceForeground.b,
                contentAlpha
        );
        drawTextInBox(
                normalFont,
                fitText(
                        normalFont,
                        faceEmoticon,
                        98f * pop * facePulse
                ),
                left + 17f - 4f * (facePulse - 1f),
                top + 14f - 3f * (facePulse - 1f),
                98f * pop * facePulse,
                (height - 28f) * facePulse,
                Align.center
        );
        smallFont.setColor(
                foreground.r,
                foreground.g,
                foreground.b,
                contentAlpha
        );
        drawTextInBox(
                smallFont,
                reactionLabel(aiReaction.mood),
                left + 136f,
                top + 8f,
                width - 154f,
                28f,
                Align.left
        );
        reactionFont.setColor(
                TEXT_PRIMARY.r,
                TEXT_PRIMARY.g,
                TEXT_PRIMARY.b,
                contentAlpha
        );
        drawWrappedTextInBox(
                reactionFont,
                aiReaction.text,
                left + 136f,
                top + 34f,
                width - 154f,
                height - 40f,
                Align.left
        );
        batch.end();
    }

    /**
     * Silent emotion channel: an event can change the kaomoji immediately even when the text
     * scheduler deliberately refuses another sentence.
     */
    private void drawAiEmotionPulseChip() {
        AiEmotionPulse pulse = aiEmotionPulse;
        if (pulse == null) {
            return;
        }
        float enter = MathUtils.clamp(pulse.age / 0.10f, 0f, 1f);
        float leave = MathUtils.clamp(
                (pulse.duration - pulse.age) / 0.18f,
                0f,
                1f
        );
        float alpha = Math.min(enter, leave);
        float pop = 0.82f
                + (1f - (float) Math.pow(1f - enter, 3f)) * 0.18f;
        float width = 132f * pop;
        float height = 66f * pop;
        float left = 24f + (132f - width) * 0.5f;
        float top = 282f + (66f - height) * 0.5f;
        Color accent = reactionAccent(pulse.mood);
        Color foreground = reactionForeground(pulse.mood);
        float panelAlpha = alpha * (1f - 0.42f * aiBubbleOcclusion);
        float contentAlpha = alpha * (1f - 0.06f * aiBubbleOcclusion);

        Gdx.gl.glEnable(GL20.GL_BLEND);
        Gdx.gl.glBlendFunc(
                GL20.GL_SRC_ALPHA,
                GL20.GL_ONE_MINUS_SRC_ALPHA
        );
        shapes.begin(ShapeRenderer.ShapeType.Filled);
        scratchColor.set(CARD_SHADOW);
        scratchColor.a = 0.32f * panelAlpha;
        roundedRectTop(
                left,
                top + 5f,
                width,
                height,
                21f * pop,
                scratchColor
        );
        scratchColor.set(accent);
        scratchColor.a = 0.96f * panelAlpha;
        roundedRectTop(
                left,
                top,
                width,
                height,
                21f * pop,
                scratchColor
        );
        scratchColor.set(AI_BUBBLE_CARD);
        scratchColor.a = 0.98f * panelAlpha;
        roundedRectTop(
                left + 3f,
                top + 3f,
                width - 6f,
                height - 6f,
                18f * pop,
                scratchColor
        );
        shapes.end();

        batch.begin();
        normalFont.setColor(
                foreground.r,
                foreground.g,
                foreground.b,
                contentAlpha
        );
        drawTextInBox(
                normalFont,
                fitText(normalFont, pulse.emoticon, width - 18f),
                left + 9f,
                top + 5f,
                width - 18f,
                height - 10f,
                Align.center
        );
        batch.end();
    }

    private boolean reactionOverlapsVisibleFruit() {
        float left = 24f;
        float top = aiReaction == null ? 282f : 270f;
        float right = aiReaction == null ? 156f : 536f;
        float bottom = aiReaction == null ? 348f : 360f;
        if (gameMode == GameMode.DUEL && duelMatch != null) {
            if (laneOverlapsReaction(
                    duelMatch.lane(duelForeground),
                    left,
                    top,
                    right,
                    bottom
            )) {
                return true;
            }
            DuelMatch.Side background = duelForeground
                    == DuelMatch.Side.PLAYER
                    ? DuelMatch.Side.AI
                    : DuelMatch.Side.PLAYER;
            return laneOverlapsReaction(
                    duelMatch.lane(background),
                    left + 8f,
                    top + 8f,
                    right - 8f,
                    bottom - 8f
            );
        }
        for (FruitPhysicsWorld.FruitBody fruit : physics.fruits()) {
            if (circleIntersectsRectangle(
                    fruit.x(),
                    sceneScreenY(fruit.y()),
                    fruit.displayRadius + 4f,
                    left,
                    top,
                    right,
                    bottom
            )) {
                return true;
            }
        }
        return false;
    }

    private boolean laneOverlapsReaction(
            DuelMatch.Lane lane,
            float left,
            float top,
            float right,
            float bottom) {
        for (FruitPhysicsWorld.FruitBody fruit
                : lane.physics().fruits()) {
            if (circleIntersectsRectangle(
                    fruit.x(),
                    sceneScreenY(fruit.y()),
                    fruit.displayRadius + 4f,
                    left,
                    top,
                    right,
                    bottom
            )) {
                return true;
            }
        }
        return false;
    }

    static boolean circleIntersectsRectangle(
            float centerX,
            float centerY,
            float radius,
            float left,
            float top,
            float right,
            float bottom) {
        float nearestX = MathUtils.clamp(centerX, left, right);
        float nearestY = MathUtils.clamp(centerY, top, bottom);
        float dx = centerX - nearestX;
        float dy = centerY - nearestY;
        return dx * dx + dy * dy <= radius * radius;
    }

    private Color reactionAccent(AiMood mood) {
        if (mood == AiMood.SURPRISED) {
            return DANGER;
        }
        if (mood == AiMood.HAPPY) {
            return ACCENT_DARK;
        }
        if (mood == AiMood.WORRIED) {
            return PLAYER_TINT;
        }
        return AI_TINT;
    }

    private Color reactionForeground(AiMood mood) {
        if (mood == AiMood.SURPRISED
                || mood == AiMood.WORRIED) {
            return PLAYER_SCORE_DARK;
        }
        if (mood == AiMood.HAPPY) {
            return ACCENT_DARK;
        }
        return AI_SCORE_DARK;
    }

    private String reactionLabel(AiMood mood) {
        return AiDialogueDirector.Mood.valueOf(mood.name()).label();
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

    private void drawApp() {
        if (appScreen == AppScreen.HOME) {
            drawHome();
        } else {
            drawGame();
        }
    }

    private void drawHome() {
        ScreenUtils.clear(BACKGROUND_TOP);
        camera.update();
        Gdx.gl.glEnable(GL20.GL_BLEND);
        Gdx.gl.glBlendFunc(
                GL20.GL_SRC_ALPHA,
                GL20.GL_ONE_MINUS_SRC_ALPHA
        );

        shapes.setProjectionMatrix(camera.combined);
        shapes.begin(ShapeRenderer.ShapeType.Filled);
        drawBackground();
        roundedRectTop(24f, 30f, 512f, 962f, 30f, CARD_SHADOW);
        roundedRectTop(24f, 24f, 512f, 962f, 30f, PANEL_COLOR);

        scratchColor.set(ACCENT_SOFT);
        scratchColor.a = 0.56f;
        roundedRectTop(52f, 188f, 456f, 102f, 24f, scratchColor);
        shapes.setColor(ORCHARD_GLOW);
        shapes.circle(92f, toRenderY(128f), 54f, 42);
        shapes.circle(474f, toRenderY(130f), 46f, 42);
        drawLeaf(73f, 107f, 55f, 21f, -24f, LEAF_LIGHT);
        drawLeaf(492f, 108f, 49f, 19f, 31f, LEAF_DARK);

        drawHomeModeCardShapes(
                CONTROL_HOME_SOLO,
                HOME_SOLO_TOP,
                ACCENT_SOFT,
                ACCENT_DARK,
                0
        );
        drawHomeModeCardShapes(
                CONTROL_HOME_DUEL,
                HOME_DUEL_TOP,
                PLAYER_TINT_SOFT,
                AI_TINT,
                1
        );
        drawHomeModeCardShapes(
                CONTROL_HOME_DEMO,
                HOME_DEMO_TOP,
                AI_TINT_SOFT,
                AI_SCORE_DARK,
                2
        );
        drawAnimatedButton(
                CONTROL_HOME_HISTORY,
                HOME_CARD_LEFT,
                HOME_UTILITY_TOP,
                HOME_UTILITY_WIDTH,
                HOME_UTILITY_HEIGHT,
                25f,
                NEXT_CARD
        );
        drawAnimatedButton(
                CONTROL_HOME_SETTINGS,
                HOME_CARD_LEFT + 238f,
                HOME_UTILITY_TOP,
                HOME_UTILITY_WIDTH,
                HOME_UTILITY_HEIGHT,
                25f,
                ACCENT_SOFT
        );
        if (sessionStore.hasSavedSession() || inMemorySession) {
            drawAnimatedButton(
                    CONTROL_HOME_RESUME,
                    298f,
                    210f,
                    184f,
                    58f,
                    24f,
                    ACCENT
            );
        }
        shapes.end();

        batch.setProjectionMatrix(camera.combined);
        batch.begin();
        titleFont.setColor(TEXT_PRIMARY);
        titleFont.getData().setScale(0.56f);
        drawTextInBox(
                titleFont,
                "合成大西瓜",
                54f,
                64f,
                452f,
                60f,
                Align.center
        );
        titleFont.getData().setScale(0.47f);
        smallFont.setColor(TEXT_MUTED);
        drawTextInBox(
                smallFont,
                "今天想怎样和水果们玩？",
                54f,
                132f,
                452f,
                34f,
                Align.center
        );

        normalFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                normalFont,
                sessionStore.hasSavedSession() || inMemorySession
                        ? "上次的水果还在等你"
                        : "进度会自动保存，不怕突然离开",
                76f,
                202f,
                210f,
                38f,
                Align.left
        );
        smallFont.setColor(TEXT_MUTED);
        drawTextInBox(
                smallFont,
                sessionStore.hasSavedSession() || inMemorySession
                        ? "可以继续，也可以选择新模式"
                        : "选择一种模式开始吧",
                76f,
                242f,
                214f,
                28f,
                Align.left
        );
        if (sessionStore.hasSavedSession() || inMemorySession) {
            normalFont.setColor(Color.WHITE);
            drawAnimatedTextInBox(
                    CONTROL_HOME_RESUME,
                    normalFont,
                    "继续上次",
                    298f,
                    210f,
                    184f,
                    58f,
                    Align.center
            );
        }

        drawHomeModeCardText(
                CONTROL_HOME_SOLO,
                HOME_SOLO_TOP,
                "单人模式",
                "完全由你掌控，可连续快速投放",
                "自己来",
                ACCENT_DARK
        );
        drawHomeModeCardText(
                CONTROL_HOME_DUEL,
                HOME_DUEL_TOP,
                "挑战 AI",
                "相同水果序列，看看谁坚持更久",
                "VS",
                PLAYER_SCORE_DARK
        );
        drawHomeModeCardText(
                CONTROL_HOME_DEMO,
                HOME_DEMO_TOP,
                "AI 演示",
                "不用操作，看 AI 独立完成整局",
                "AI",
                AI_SCORE_DARK
        );

        normalFont.setColor(TEXT_PRIMARY);
        drawAnimatedTextInBox(
                CONTROL_HOME_HISTORY,
                normalFont,
                "历史记录",
                HOME_CARD_LEFT,
                HOME_UTILITY_TOP,
                HOME_UTILITY_WIDTH,
                HOME_UTILITY_HEIGHT,
                Align.center
        );
        normalFont.setColor(ACCENT_DARK);
        drawAnimatedTextInBox(
                CONTROL_HOME_SETTINGS,
                normalFont,
                "游戏设置",
                HOME_CARD_LEFT + 238f,
                HOME_UTILITY_TOP,
                HOME_UTILITY_WIDTH,
                HOME_UTILITY_HEIGHT,
                Align.center
        );
        smallFont.setColor(TEXT_MUTED);
        drawTextInBox(
                smallFont,
                aiService.isAiReady()
                        ? "离线 AI 已准备好"
                        : "AI 正在准备，单人模式可立即游玩",
                54f,
                924f,
                452f,
                32f,
                Align.center
        );
        batch.end();

        if (overlayPage != OverlayPage.NONE) {
            drawOverlayPage();
        }
    }

    private void drawHomeModeCardShapes(
            String controlId,
            float top,
            Color background,
            Color accent,
            int iconType) {
        drawAnimatedButton(
                controlId,
                HOME_CARD_LEFT,
                top,
                HOME_CARD_WIDTH,
                HOME_CARD_HEIGHT,
                24f,
                background
        );
        UiMotionController.Visual visual = uiMotion.visual(controlId);
        float iconX = HOME_CARD_LEFT + 68f + visual.offsetX;
        float iconY = top + HOME_CARD_HEIGHT / 2f + visual.offsetY;
        shapes.setColor(SCORE_CARD);
        shapes.circle(iconX, toRenderY(iconY), 37f, 36);
        shapes.setColor(accent);
        shapes.circle(iconX, toRenderY(iconY), 30f, 36);
        shapes.setColor(SCORE_CARD);
        if (iconType == 0) {
            shapes.circle(iconX - 8f, toRenderY(iconY - 3f), 3f, 14);
            shapes.circle(iconX + 8f, toRenderY(iconY - 3f), 3f, 14);
            shapes.rect(iconX - 9f, toRenderY(iconY + 10f), 18f, 3f);
        } else if (iconType == 1) {
            shapes.circle(iconX - 10f, toRenderY(iconY), 11f, 24);
            shapes.circle(iconX + 10f, toRenderY(iconY), 11f, 24);
            shapes.setColor(accent);
            shapes.rect(iconX - 2f, toRenderY(iconY + 9f), 4f, 18f);
        } else {
            shapes.circle(iconX - 8f, toRenderY(iconY - 3f), 3f, 14);
            shapes.circle(iconX + 8f, toRenderY(iconY - 3f), 3f, 14);
            for (int dot = 0; dot < 5; dot++) {
                float angle = MathUtils.PI * dot / 4f;
                shapes.circle(
                        iconX + MathUtils.cos(angle) * 12f,
                        toRenderY(iconY + 6f + MathUtils.sin(angle) * 8f),
                        1.8f,
                        10
                );
            }
        }
    }

    private void drawHomeModeCardText(
            String controlId,
            float top,
            String title,
            String description,
            String badge,
            Color accent) {
        normalFont.setColor(TEXT_PRIMARY);
        drawAnimatedTextInBox(
                controlId,
                normalFont,
                title,
                HOME_CARD_LEFT + 126f,
                top + 23f,
                244f,
                42f,
                Align.left
        );
        smallFont.setColor(TEXT_MUTED);
        drawAnimatedTextInBox(
                controlId,
                smallFont,
                description,
                HOME_CARD_LEFT + 126f,
                top + 70f,
                286f,
                34f,
                Align.left
        );
        smallFont.setColor(accent);
        drawAnimatedTextInBox(
                controlId,
                smallFont,
                badge,
                HOME_CARD_LEFT + 370f,
                top + 24f,
                56f,
                38f,
                Align.center
        );
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
        drawEffects(true);
        shapes.end();

        batch.begin();
        drawScoreTokens();
        drawDuelScoreTokens();
        batch.end();

        /*
         * 底部停靠时 HUD 必须盖在棋盘和水果之上；屏幕空间的收分冲击再画到卡片
         * 上方。这样既不会让水果遮住总分，也不会把计分闪光压在卡片背后。
         */
        shapes.begin(ShapeRenderer.ShapeType.Filled);
        drawScorePanels();
        drawEffects(false);
        shapes.end();

        batch.begin();
        drawText();
        batch.end();

        if (isSingleBoardMode() && !alive) {
            drawGameOverOverlay();
        } else if (gameMode == GameMode.DUEL) {
            drawDuelResultLayer();
        }
        drawAiReaction();
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
        float headerHeight = MathUtils.lerp(
                HEADER_EXPANDED_HEIGHT,
                HEADER_COMPACT_HEIGHT,
                bottomScoreVisibility()
        );
        roundedRectTop(16f, 16f, 528f, headerHeight, 20f, CARD_SHADOW);
        roundedRectTop(16f, 12f, 528f, headerHeight, 20f, PANEL_COLOR);

        drawAnimatedButton(
                CONTROL_GAME_BACK,
                GAME_BACK_LEFT,
                GAME_BACK_TOP,
                GAME_BACK_SIZE,
                GAME_BACK_SIZE,
                GAME_BACK_SIZE / 2f,
                SCORE_CARD
        );
        if (gameMode == GameMode.DUEL) {
            drawAnimatedButton(
                    CONTROL_DUEL_FOREGROUND,
                    AI_TOGGLE_LEFT,
                    AI_TOGGLE_TOP,
                    AI_TOGGLE_WIDTH,
                    AI_TOGGLE_HEIGHT,
                    AI_TOGGLE_HEIGHT / 2f,
                    duelForegroundSide() == DuelMatch.Side.PLAYER
                            ? PLAYER_TINT_SOFT
                            : AI_TINT_SOFT
            );
        }
        drawQueueSlots();

        // 棋盘几何保持不变；HUD 重排绝不移动训练/推理使用的 spawn_y。
        float sceneOffset = sceneOffsetY();
        roundedRectTop(
                18f,
                254f + sceneOffset,
                524f,
                854f,
                18f,
                CARD_SHADOW
        );
        roundedRectTop(
                18f,
                248f + sceneOffset,
                524f,
                860f,
                18f,
                BOARD_FRAME
        );
        roundedRectTop(
                20f,
                250f + sceneOffset,
                520f,
                856f,
                16f,
                BOARD_FRAME_SOFT
        );
        roundedRectTop(
                22f,
                252f + sceneOffset,
                516f,
                848f,
                14f,
                BOARD_COLOR
        );
        if (gameMode == GameMode.DUEL) {
            Color tint = duelForeground == DuelMatch.Side.PLAYER
                    ? PLAYER_TINT
                    : AI_TINT;
            scratchColor.set(tint.r, tint.g, tint.b, 0.055f);
            roundedRectTop(
                    24f,
                    254f + sceneOffset,
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
        shapes.circle(88f, toRenderY(sceneScreenY(1018f)), 58f, 36);
        shapes.circle(474f, toRenderY(sceneScreenY(1028f)), 50f, 36);

        float visibleDangerSeconds = gameMode == GameMode.DUEL
                ? duelForegroundLane().dangerSeconds()
                : dangerSeconds;
        float dangerAlpha = visibleDangerSeconds <= 0f
                ? 0.48f
                : 0.55f + MathUtils.sin(elapsedSeconds * 12f) * 0.25f;
        shapes.setColor(DANGER.r, DANGER.g, DANGER.b, dangerAlpha);
        float dangerY = toRenderY(sceneScreenY(FruitRules.SPAWN_Y + 4f));
        for (float x = 30f; x < 530f; x += 20f) {
            shapes.rect(x, dangerY - 1f, 11f, 2f);
        }

        boolean previewVisible = gameMode == GameMode.DUEL
                ? duelForegroundPreviewVisible()
                : waiting;
        if (previewVisible) {
            boolean assistedPreview = gameMode == GameMode.DUEL
                    ? duelForeground == DuelMatch.Side.AI
                    : isAiControlledSingleBoard();
            int visibleLevel = gameMode == GameMode.DUEL
                    ? duelMatch.currentLevel()
                    : currentLevel;
            float visiblePreviewX = gameMode == GameMode.DUEL
                    ? duelForegroundLane().previewX()
                    : previewX;
            shapes.setColor(
                    ACCENT.r,
                    ACCENT.g,
                    ACCENT.b,
                    assistedPreview ? 0.19f : 0.13f
            );
            float guideTop = previewY(visibleLevel)
                    + FruitRules.displayRadius(visibleLevel);
            for (float screenY = guideTop + 14f;
                    screenY < FruitRules.FLOOR_Y;
                    screenY += 26f) {
                shapes.circle(
                        visiblePreviewX,
                        toRenderY(sceneScreenY(screenY)),
                        1.7f,
                        10
                );
            }
            shapes.setColor(
                    ACCENT.r,
                    ACCENT.g,
                    ACCENT.b,
                    assistedPreview ? 0.22f : 0.14f
            );
            shapes.circle(
                    visiblePreviewX,
                    toRenderY(sceneScreenY(previewY(visibleLevel))),
                    FruitRules.displayRadius(visibleLevel) + 5f,
                    40
            );
        }
    }

    private void drawQueueSlots() {
        for (float slotX : QUEUE_SLOT_X) {
            shapes.setColor(CARD_SHADOW);
            shapes.circle(
                    slotX,
                    toRenderY(QUEUE_SLOT_CENTER_Y + 2.5f),
                    14f,
                    28
            );
            shapes.setColor(BOARD_FRAME_SOFT);
            shapes.circle(
                    slotX,
                    toRenderY(QUEUE_SLOT_CENTER_Y),
                    14f,
                    28
            );
            shapes.setColor(SCORE_CARD);
            shapes.circle(
                    slotX,
                    toRenderY(QUEUE_SLOT_CENTER_Y),
                    12.2f,
                    28
            );
        }
        drawQueueFocusReticle(QUEUE_SLOT_X[0], QUEUE_SLOT_CENTER_Y);
    }

    private void drawScorePanels() {
        drawScorePanelAt(SCORE_TOP_DOCK_Y, topScoreVisibility());
        drawScorePanelAt(SCORE_BOTTOM_DOCK_Y, bottomScoreVisibility());
    }

    /**
     * Two fixed docks cross-fade instead of dragging a full-width opaque card through the fruit
     * pile. The subtle scale change still makes the relocation readable, while the board itself
     * provides the continuous spatial motion requested by the adaptive layout.
     */
    private void drawScorePanelAt(float top, float visibility) {
        float alpha = MathUtils.clamp(visibility, 0f, 1f);
        if (alpha <= 0.002f) {
            return;
        }
        float scale = 0.94f + alpha * 0.06f;
        float scaledHeight = SCORE_CARD_HEIGHT * scale;
        float scaledTop = top + (SCORE_CARD_HEIGHT - scaledHeight) * 0.5f;

        if (scorePulse > 0f) {
            float expansion = (1f - scorePulse) * 10f;
            scratchColor.set(
                    SCORE_GLOW.r,
                    SCORE_GLOW.g,
                    SCORE_GLOW.b,
                    scorePulse * 0.34f * alpha
            );
            roundedRectTop(
                    22f - expansion,
                    scaledTop + 1f - expansion,
                    516f + expansion * 2f,
                    scaledHeight + 4f + expansion * 2f,
                    18f + expansion,
                    scratchColor
            );
        }

        if (gameMode == GameMode.DUEL) {
            float cardWidth = 230f * scale;
            float playerLeft = 26f + (230f - cardWidth) * 0.5f;
            float aiLeft = 304f + (230f - cardWidth) * 0.5f;

            scratchColor.set(CARD_SHADOW);
            scratchColor.a *= alpha;
            roundedRectTop(
                    playerLeft,
                    scaledTop + 4f,
                    cardWidth,
                    scaledHeight,
                    16f * scale,
                    scratchColor
            );
            scratchColor.set(PLAYER_TINT_SOFT);
            scratchColor.a *= alpha;
            roundedRectTop(
                    playerLeft,
                    scaledTop,
                    cardWidth,
                    scaledHeight,
                    16f * scale,
                    scratchColor
            );

            scratchColor.set(CARD_SHADOW);
            scratchColor.a *= alpha;
            roundedRectTop(
                    aiLeft,
                    scaledTop + 4f,
                    cardWidth,
                    scaledHeight,
                    16f * scale,
                    scratchColor
            );
            scratchColor.set(AI_TINT_SOFT);
            scratchColor.a *= alpha;
            roundedRectTop(
                    aiLeft,
                    scaledTop,
                    cardWidth,
                    scaledHeight,
                    16f * scale,
                    scratchColor
            );

            float centerY = top + SCORE_CARD_HEIGHT * 0.53f;
            shapes.setColor(
                    TEXT_PRIMARY.r,
                    TEXT_PRIMARY.g,
                    TEXT_PRIMARY.b,
                    alpha
            );
            shapes.circle(280f, toRenderY(centerY), 24f * scale, 32);
            shapes.setColor(
                    SCORE_CARD.r,
                    SCORE_CARD.g,
                    SCORE_CARD.b,
                    alpha
            );
            shapes.circle(280f, toRenderY(centerY), 20f * scale, 32);
            return;
        }

        float cardWidth = 508f * scale;
        float cardLeft = 26f + (508f - cardWidth) * 0.5f;
        Color scoreBackground = gameMode == GameMode.AI_DEMO
                ? AI_TINT_SOFT
                : PLAYER_TINT_SOFT;
        scratchColor.set(CARD_SHADOW);
        scratchColor.a *= alpha;
        roundedRectTop(
                cardLeft,
                scaledTop + 4f,
                cardWidth,
                scaledHeight,
                16f * scale,
                scratchColor
        );
        scratchColor.set(scoreBackground);
        scratchColor.a *= alpha;
        roundedRectTop(
                cardLeft,
                scaledTop,
                cardWidth,
                scaledHeight,
                16f * scale,
                scratchColor
        );
    }

    @SuppressWarnings("unused")
    private void drawLegacyPanels() {
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
        drawQueueSlots();

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

    private void drawEffects(boolean boardPass) {
        Gdx.gl.glEnable(GL20.GL_BLEND);
        if (boardPass) {
            for (MergeBurst burst : mergeBursts) {
                float remaining = MathUtils.clamp(
                        burst.life / burst.maxLife,
                        0f,
                        1f
                );
                float progress = 1f - remaining;
                float renderY = toRenderY(sceneScreenY(burst.y));
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
                            burst.x + MathUtils.cos(angle - halfAngle)
                                    * outerRadius,
                            renderY + MathUtils.sin(angle - halfAngle)
                                    * outerRadius,
                            burst.x + MathUtils.cos(angle + halfAngle)
                                    * outerRadius,
                            renderY + MathUtils.sin(angle + halfAngle)
                                    * outerRadius
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
        }
        for (Particle particle : particles) {
            if (particle.boardAnchored != boardPass) {
                continue;
            }
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
                    toRenderY(
                            particle.boardAnchored
                                    ? sceneScreenY(particle.y)
                                    : particle.y
                    ),
                    particle.radius * (0.62f + remaining * 0.38f),
                    12
            );
        }
        for (Ring ring : rings) {
            if (ring.boardAnchored != boardPass) {
                continue;
            }
            float remaining = MathUtils.clamp(
                    ring.life / ring.maxLife,
                    0f,
                    1f
            );
            float alpha = (float) Math.pow(remaining, 0.62f) * 0.82f;
            shapes.setColor(ring.color.r, ring.color.g, ring.color.b, alpha);
            float renderY = toRenderY(
                    ring.boardAnchored
                            ? sceneScreenY(ring.y)
                            : ring.y
            );
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
            float displayY = token.phase == TokenPhase.FLY
                    ? token.y
                    : sceneScreenY(token.y);
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
                    displayY + 2.8f
            );

            scratchColor.set(
                    MathUtils.lerp(token.color.r, 1f, 0.16f),
                    MathUtils.lerp(token.color.g, 1f, 0.16f),
                    MathUtils.lerp(token.color.b, 1f, 0.16f),
                    token.alpha
            );
            popupFont.setColor(scratchColor);
            drawTextCentered(popupFont, text, token.x, displayY);
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
            float displayY = token.age > 0.50f
                    ? token.y
                    : sceneScreenY(token.y);
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
                    displayY + 2.8f
            );
            scratchColor.set(
                    MathUtils.lerp(token.color.r, 1f, 0.16f),
                    MathUtils.lerp(token.color.g, 1f, 0.16f),
                    MathUtils.lerp(token.color.b, 1f, 0.16f),
                    token.alpha
            );
            popupFont.setColor(scratchColor);
            drawTextCentered(popupFont, text, token.x, displayY);
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
                    sceneScreenY(fruit.y()),
                    fruit.displayRadius * 2f,
                    fruit.angle(),
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
                    sceneScreenY(previewY(currentLevel)),
                    FruitRules.displayRadius(currentLevel) * 2f,
                    alpha
            );
        }

        // q0 也显示在队列中；动态准星明确标记它就是当前正在投放的水果。
        for (int index = 0;
                index < Math.min(queue.size, FruitRules.QUEUE_LENGTH);
                index++) {
            int level = queue.get(index);
            float size = Math.min(
                    QUEUE_FRUIT_MAX_SIZE,
                    FruitRules.displayRadius(level) * 0.82f
            );
            drawFruit(
                    level,
                    QUEUE_SLOT_X[index],
                    QUEUE_SLOT_CENTER_Y,
                    size,
                    index == 0 ? 1f : 0.88f
            );
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
                    sceneScreenY(fruit.y()),
                    size,
                    backgroundTint,
                    fruit.angle(),
                    0.075f
            );
            drawFruitTinted(
                    fruit.level,
                    fruit.x() + 1.8f,
                    sceneScreenY(fruit.y()),
                    size,
                    backgroundTint,
                    fruit.angle(),
                    0.075f
            );
            drawFruitTinted(
                    fruit.level,
                    fruit.x(),
                    sceneScreenY(fruit.y() + 1.4f),
                    size,
                    backgroundTint,
                    fruit.angle(),
                    0.10f
            );
        }
        for (FruitPhysicsWorld.FruitBody fruit
                : foreground.physics().fruits()) {
            drawFruit(
                    fruit.level,
                    fruit.x(),
                    sceneScreenY(fruit.y()),
                    fruit.displayRadius * 2f,
                    fruit.angle(),
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
                    sceneScreenY(previewY(level)),
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
                    sceneScreenY(previewY(level)),
                    FruitRules.displayRadius(level) * 2f,
                    0.96f
            );
        }
        for (int index = 0;
                index < FruitRules.QUEUE_LENGTH;
                index++) {
            int queuedLevel = duelMatch.queuedLevel(index);
            float size = Math.min(
                    QUEUE_FRUIT_MAX_SIZE,
                    FruitRules.displayRadius(queuedLevel) * 0.82f
            );
            drawFruit(
                    queuedLevel,
                    QUEUE_SLOT_X[index],
                    QUEUE_SLOT_CENTER_Y,
                    size,
                    index == 0 ? 1f : 0.88f
            );
        }
    }

    /** 用呼吸、轻摆与环绕光点标记队列中的当前水果。 */
    private void drawQueueFocusReticle(float centerX, float centerTopY) {
        float phase = elapsedSeconds * 2.75f;
        float radius = 15.6f + MathUtils.sin(phase) * 1.05f;
        float arm = 5.6f + MathUtils.sin(phase * 0.68f + 0.9f) * 0.65f;
        float rotation = MathUtils.sin(phase * 0.43f) * 0.10f;
        float cosine = MathUtils.cos(rotation);
        float sine = MathUtils.sin(rotation);
        float centerY = toRenderY(centerTopY);
        float alpha = 0.72f + MathUtils.sin(phase) * 0.15f;
        scratchColor.set(ACCENT.r, ACCENT.g, ACCENT.b, alpha);
        shapes.setColor(scratchColor);
        for (int xSign = -1; xSign <= 1; xSign += 2) {
            for (int ySign = -1; ySign <= 1; ySign += 2) {
                float localX = xSign * radius;
                float localY = ySign * radius;
                float cornerX = centerX + localX * cosine - localY * sine;
                float cornerY = centerY + localX * sine + localY * cosine;
                float xArmX = cornerX - xSign * arm * cosine;
                float xArmY = cornerY - xSign * arm * sine;
                float yArmX = cornerX + ySign * arm * sine;
                float yArmY = cornerY - ySign * arm * cosine;
                shapes.rectLine(cornerX, cornerY, xArmX, xArmY, 1.8f);
                shapes.rectLine(cornerX, cornerY, yArmX, yArmY, 1.8f);
            }
        }
        float orbit = phase * 0.62f;
        float orbitRadius = radius + 2.7f;
        shapes.circle(
                centerX + MathUtils.cos(orbit) * orbitRadius,
                centerY + MathUtils.sin(orbit) * orbitRadius,
                1.65f,
                12
        );
        shapes.circle(
                centerX - MathUtils.cos(orbit) * orbitRadius,
                centerY - MathUtils.sin(orbit) * orbitRadius,
                1.05f,
                12
        );
    }
    private void drawFruit(int level, float centerX, float centerY, float size, float alpha) {
        drawFruit(level, centerX, centerY, size, 0f, alpha);
    }

    private void drawFruit(
            int level,
            float centerX,
            float centerY,
            float size,
            float angleRadians,
            float alpha) {
        batch.setColor(1f, 1f, 1f, alpha);
        batch.draw(
                fruitTextures[level],
                centerX - size / 2f,
                toRenderY(centerY + size / 2f),
                size / 2f,
                size / 2f,
                size,
                size,
                1f,
                1f,
                -angleRadians * MathUtils.radiansToDegrees
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
        drawFruitTinted(level, centerX, centerY, size, tint, 0f, alpha);
    }

    private void drawFruitTinted(
            int level,
            float centerX,
            float centerY,
            float size,
            Color tint,
            float angleRadians,
            float alpha) {
        batch.setColor(tint.r, tint.g, tint.b, alpha);
        batch.draw(
                fruitTextures[level],
                centerX - size / 2f,
                toRenderY(centerY + size / 2f),
                size / 2f,
                size / 2f,
                size,
                size,
                1f,
                1f,
                -angleRadians * MathUtils.radiansToDegrees
        );
        batch.setColor(Color.WHITE);
    }

    private void drawText() {
        titleFont.setColor(TEXT_PRIMARY);
        titleFont.getData().setScale(0.38f);
        drawTextInBox(
                titleFont,
                modeTitle(),
                72f,
                16f,
                142f,
                42f,
                Align.left
        );
        titleFont.getData().setScale(0.47f);

        smallFont.setColor(TEXT_MUTED);
        drawTextInBox(
                smallFont,
                "投放序列",
                218f,
                18f,
                66f,
                40f,
                Align.center
        );
        smallFont.setColor(TEXT_PRIMARY);
        drawAnimatedTextInBox(
                CONTROL_GAME_BACK,
                smallFont,
                "返回",
                GAME_BACK_LEFT,
                GAME_BACK_TOP,
                GAME_BACK_SIZE,
                GAME_BACK_SIZE,
                Align.center
        );

        if (gameMode == GameMode.DUEL) {
            smallFont.setColor(
                    duelForegroundSide() == DuelMatch.Side.PLAYER
                            ? PLAYER_SCORE_DARK
                            : AI_SCORE_DARK
            );
            drawAnimatedTextInBox(
                    CONTROL_DUEL_FOREGROUND,
                    smallFont,
                    duelForegroundSide() == DuelMatch.Side.PLAYER
                            ? "玩家"
                            : "AI",
                    AI_TOGGLE_LEFT,
                    AI_TOGGLE_TOP,
                    AI_TOGGLE_WIDTH,
                    AI_TOGGLE_HEIGHT,
                    Align.center
            );
        } else if (gameMode == GameMode.AI_DEMO) {
            smallFont.setColor(AI_SCORE_DARK);
            drawTextInBox(
                    smallFont,
                    aiState.label,
                    AI_TOGGLE_LEFT,
                    AI_TOGGLE_TOP,
                    AI_TOGGLE_WIDTH,
                    42f,
                    Align.center
            );
        }

        drawScoreTextAt(SCORE_TOP_DOCK_Y, topScoreVisibility());
        drawScoreTextAt(SCORE_BOTTOM_DOCK_Y, bottomScoreVisibility());

        if (gameMode == GameMode.DUEL) {
            smallFont.setColor(duelRoundUrgent() ? DANGER : TEXT_MUTED);
            drawTextInBox(
                    smallFont,
                    duelAiArmed && !duelMatch.playerLane().submittedThisRound()
                            ? "AI 已就位，等你一起投放"
                            : duelRoundLabel(),
                    100f,
                    sceneScreenY(1068f),
                    360f,
                    28f,
                    Align.center
            );
        } else if (gameMode == GameMode.SOLO && alive) {
            smallFont.setColor(MANUAL_HINT);
            drawTextInBox(
                    smallFont,
                    "拖动水果，松手投放",
                    140f,
                    sceneScreenY(1070f),
                    280f,
                    28f,
                    Align.center
            );
        } else if (gameMode == GameMode.AI_DEMO && alive) {
            smallFont.setColor(AI_SCORE_DARK);
            drawTextInBox(
                    smallFont,
                    "AI 正在独立游玩",
                    140f,
                    sceneScreenY(1070f),
                    280f,
                    28f,
                    Align.center
            );
        }
    }

    private void drawScoreTextAt(float dockTop, float visibility) {
        float alpha = MathUtils.clamp(visibility, 0f, 1f);
        if (alpha <= 0.01f) {
            return;
        }
        if (gameMode == GameMode.DUEL) {
            smallFont.setColor(
                    PLAYER_SCORE_DARK.r,
                    PLAYER_SCORE_DARK.g,
                    PLAYER_SCORE_DARK.b,
                    alpha
            );
            drawTextInBox(
                    smallFont,
                    "玩家",
                    38f,
                    dockTop + 6f,
                    58f,
                    24f,
                    Align.left
            );
            normalFont.getData().setScale(
                    0.62f * (1f + scorePulse * 0.08f)
            );
            normalFont.setColor(
                    PLAYER_SCORE_DARK.r,
                    PLAYER_SCORE_DARK.g,
                    PLAYER_SCORE_DARK.b,
                    alpha
            );
            drawTextInBox(
                    normalFont,
                    Integer.toString(duelPlayerScore()),
                    88f,
                    dockTop + 6f,
                    154f,
                    54f,
                    Align.center
            );
            normalFont.setColor(
                    AI_SCORE_DARK.r,
                    AI_SCORE_DARK.g,
                    AI_SCORE_DARK.b,
                    alpha
            );
            drawTextInBox(
                    normalFont,
                    Integer.toString(duelAiScore()),
                    318f,
                    dockTop + 6f,
                    154f,
                    54f,
                    Align.center
            );
            normalFont.getData().setScale(0.38f);
            smallFont.setColor(
                    AI_SCORE_DARK.r,
                    AI_SCORE_DARK.g,
                    AI_SCORE_DARK.b,
                    alpha
            );
            drawTextInBox(
                    smallFont,
                    "AI",
                    472f,
                    dockTop + 6f,
                    50f,
                    24f,
                    Align.right
            );
            smallFont.setColor(
                    TEXT_PRIMARY.r,
                    TEXT_PRIMARY.g,
                    TEXT_PRIMARY.b,
                    alpha
            );
            drawTextInBox(
                    smallFont,
                    "VS",
                    256f,
                    dockTop + 18f,
                    48f,
                    34f,
                    Align.center
            );
            return;
        }

        Color scoreColor = gameMode == GameMode.AI_DEMO
                ? AI_SCORE_DARK
                : PLAYER_SCORE_DARK;
        smallFont.setColor(
                scoreColor.r,
                scoreColor.g,
                scoreColor.b,
                alpha
        );
        drawTextInBox(
                smallFont,
                gameMode == GameMode.AI_DEMO ? "AI 得分" : "本局得分",
                42f,
                dockTop + 6f,
                88f,
                24f,
                Align.left
        );
        normalFont.setColor(
                scoreColor.r,
                scoreColor.g,
                scoreColor.b,
                alpha
        );
        normalFont.getData().setScale(
                0.68f * (1f + scorePulse * 0.08f)
        );
        drawTextInBox(
                normalFont,
                Integer.toString(displayedScore),
                130f,
                dockTop + 3f,
                300f,
                58f,
                Align.center
        );
        normalFont.getData().setScale(0.38f);
        smallFont.setColor(
                scoreColor.r,
                scoreColor.g,
                scoreColor.b,
                alpha
        );
        drawTextInBox(
                smallFont,
                "最高 " + displayedBestScore,
                430f,
                dockTop + 6f,
                88f,
                44f,
                Align.right
        );
    }

    private String modeTitle() {
        if (gameMode == GameMode.DUEL) {
            return "挑战 AI";
        }
        if (gameMode == GameMode.AI_DEMO) {
            return "AI 演示";
        }
        return "单人模式";
    }

    @SuppressWarnings("unused")
    private void drawLegacyText() {
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
        drawTextInBox(smallFont, "投放序列", 284f, 68f, 78f, 20f, Align.left);

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
        drawAnimatedButton(
                CONTROL_RESULT_CONFIRM,
                RESULT_BUTTON_LEFT,
                RESULT_BUTTON_TOP,
                RESULT_BUTTON_WIDTH,
                RESULT_BUTTON_HEIGHT,
                31f,
                ACCENT
        );
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
                gameMode == GameMode.AI_DEMO
                        ? "这次 AI 演示超越 "
                        + soloPercentile + "% 的玩家"
                        : "恭喜你已超越 "
                        + soloPercentile + "% 的玩家",
                105f,
                540f,
                350f,
                34f,
                Align.center
        );
        normalFont.setColor(TEXT_PRIMARY);
        drawAnimatedTextInBox(
                CONTROL_RESULT_CONFIRM,
                normalFont,
                "确认结算",
                RESULT_BUTTON_LEFT,
                RESULT_BUTTON_TOP,
                RESULT_BUTTON_WIDTH,
                RESULT_BUTTON_HEIGHT,
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
        drawAnimatedButton(
                CONTROL_RESULT_CONFIRM,
                146f,
                658f,
                268f,
                62f,
                31f,
                resultColor
        );
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
        drawAnimatedTextInBox(
                CONTROL_RESULT_CONFIRM,
                normalFont,
                "确认结算",
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
        if (overlayPage == OverlayPage.EXIT_CONFIRM
                || overlayPage == OverlayPage.NEW_GAME_CONFIRM) {
            drawConfirmationOverlay();
            return;
        }
        shapes.begin(ShapeRenderer.ShapeType.Filled);
        shapes.setColor(OVERLAY_DIM);
        shapes.rect(0f, 0f, FruitRules.BOARD_WIDTH, FruitRules.BOARD_HEIGHT);
        roundedRectTop(34f, 104f, 492f, 938f, 26f, CARD_SHADOW);
        roundedRectTop(34f, 98f, 492f, 938f, 26f, PANEL_COLOR);
        drawAnimatedButton(
                CONTROL_OVERLAY_CLOSE,
                430f,
                124f,
                70f,
                46f,
                23f,
                ACCENT_SOFT
        );
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

    private void drawConfirmationOverlay() {
        boolean exiting = overlayPage == OverlayPage.EXIT_CONFIRM;
        shapes.begin(ShapeRenderer.ShapeType.Filled);
        shapes.setColor(OVERLAY_DIM);
        shapes.rect(0f, 0f, FruitRules.BOARD_WIDTH, FruitRules.BOARD_HEIGHT);
        roundedRectTop(62f, 340f, 436f, 420f, 28f, CARD_SHADOW);
        roundedRectTop(62f, 334f, 436f, 420f, 28f, PANEL_COLOR);
        if (exiting) {
            drawAnimatedButton(
                    CONTROL_EXIT_SAVE,
                    104f,
                    522f,
                    352f,
                    64f,
                    28f,
                    ACCENT
            );
            drawAnimatedButton(
                    CONTROL_EXIT_ABANDON,
                    104f,
                    608f,
                    352f,
                    64f,
                    28f,
                    PLAYER_TINT_SOFT
            );
            drawAnimatedButton(
                    CONTROL_EXIT_CANCEL,
                    168f,
                    696f,
                    224f,
                    48f,
                    22f,
                    NEXT_CARD
            );
        } else {
            drawAnimatedButton(
                    CONTROL_NEW_RESUME,
                    104f,
                    522f,
                    352f,
                    64f,
                    28f,
                    ACCENT
            );
            drawAnimatedButton(
                    CONTROL_NEW_CONFIRM,
                    104f,
                    608f,
                    352f,
                    64f,
                    28f,
                    PLAYER_TINT
            );
            drawAnimatedButton(
                    CONTROL_NEW_CANCEL,
                    168f,
                    696f,
                    224f,
                    48f,
                    22f,
                    NEXT_CARD
            );
        }
        shapes.end();

        batch.begin();
        titleFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                titleFont,
                exiting ? "暂时离开吗？" : "开始新的游戏？",
                92f,
                372f,
                376f,
                58f,
                Align.center
        );
        smallFont.setColor(TEXT_MUTED);
        drawTextInBox(
                smallFont,
                exiting
                        ? "本局可保存并在大厅继续，也可以直接放弃"
                        : "新游戏会覆盖当前保存的进度，历史成绩不受影响",
                92f,
                446f,
                376f,
                46f,
                Align.center
        );
        if (exiting) {
            normalFont.setColor(Color.WHITE);
            drawAnimatedTextInBox(
                    CONTROL_EXIT_SAVE,
                    normalFont,
                    "保存并返回大厅",
                    104f,
                    522f,
                    352f,
                    64f,
                    Align.center
            );
            normalFont.setColor(PLAYER_SCORE_DARK);
            drawAnimatedTextInBox(
                    CONTROL_EXIT_ABANDON,
                    normalFont,
                    "放弃本局",
                    104f,
                    608f,
                    352f,
                    64f,
                    Align.center
            );
            smallFont.setColor(TEXT_PRIMARY);
            drawAnimatedTextInBox(
                    CONTROL_EXIT_CANCEL,
                    smallFont,
                    "继续游玩",
                    168f,
                    696f,
                    224f,
                    48f,
                    Align.center
            );
        } else {
            normalFont.setColor(Color.WHITE);
            drawAnimatedTextInBox(
                    CONTROL_NEW_RESUME,
                    normalFont,
                    "继续上次游戏",
                    104f,
                    522f,
                    352f,
                    64f,
                    Align.center
            );
            normalFont.setColor(Color.WHITE);
            drawAnimatedTextInBox(
                    CONTROL_NEW_CONFIRM,
                    normalFont,
                    "覆盖进度并开始",
                    104f,
                    608f,
                    352f,
                    64f,
                    Align.center
            );
            smallFont.setColor(TEXT_PRIMARY);
            drawAnimatedTextInBox(
                    CONTROL_NEW_CANCEL,
                    smallFont,
                    "返回大厅",
                    168f,
                    696f,
                    224f,
                    48f,
                    Align.center
            );
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
            drawAnimatedButton(
                    settingMinusControl(row),
                    354f,
                    row + 12f,
                    36f,
                    44f,
                    18f,
                    SCORE_CARD
            );
            drawAnimatedButton(
                    settingPlusControl(row),
                    442f,
                    row + 12f,
                    36f,
                    44f,
                    18f,
                    SCORE_CARD
            );
        }
        drawAnimatedButton(
                CONTROL_SETTINGS_MERGE_VIBRATE,
                370f, 321f, 108f, 42f, 21f, ACCENT_SOFT
        );
        drawAnimatedButton(
                CONTROL_SETTINGS_DROP_VIBRATE,
                370f, 411f, 108f, 42f, 21f, ACCENT_SOFT
        );
        drawAnimatedButton(
                CONTROL_SETTINGS_SCORE_VIBRATE,
                370f, 501f, 108f, 42f, 21f, ACCENT_SOFT
        );
        drawAnimatedButton(
                CONTROL_SETTINGS_RESET,
                158f, 882f, 244f, 58f, 29f, NEXT_CARD
        );
    }

    private void drawHistoryPageShapes() {
        roundedRectTop(62f, 196f, 436f, 92f, 18f, CARD_SHADOW);
        roundedRectTop(62f, 192f, 436f, 92f, 18f, SCORE_CARD);
        roundedRectTop(62f, 300f, 436f, 70f, 16f, CARD_SHADOW);
        roundedRectTop(62f, 296f, 436f, 70f, 16f, NEXT_CARD);

        List<GameProfileStore.GameRecord> records =
                profileStore.gameRecords();
        int pageCount = historyPageCount(records.size());
        historyPageIndex = MathUtils.clamp(
                historyPageIndex,
                0,
                pageCount - 1
        );
        int first = historyPageIndex * HISTORY_RECORDS_PER_PAGE;
        int visible = Math.min(
                HISTORY_RECORDS_PER_PAGE,
                records.size() - first
        );
        if (visible <= 0) {
            roundedRectTop(62f, 424f, 436f, 164f, 18f, CARD_SHADOW);
            roundedRectTop(62f, 420f, 436f, 164f, 18f, SCORE_CARD);
        } else {
            for (int row = 0; row < visible; row++) {
                float top = 420f + row * 90f;
                roundedRectTop(
                        62f,
                        top + 4f,
                        436f,
                        80f,
                        16f,
                        CARD_SHADOW
                );
                roundedRectTop(
                        62f,
                        top,
                        436f,
                        80f,
                        16f,
                        historyRecordColor(
                                records.get(first + row).mode())
                );
            }
        }
        drawAnimatedButton(
                CONTROL_HISTORY_PREVIOUS,
                154f,
                798f,
                110f,
                48f,
                22f,
                NEXT_CARD
        );
        drawAnimatedButton(
                CONTROL_HISTORY_NEXT,
                296f,
                798f,
                110f,
                48f,
                22f,
                NEXT_CARD
        );
        drawAnimatedButton(
                CONTROL_HISTORY_RESET,
                158f,
                900f,
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
        drawAnimatedTextInBox(
                CONTROL_OVERLAY_CLOSE,
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
        drawAnimatedTextInBox(
                CONTROL_SETTINGS_RESET,
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
            drawAnimatedTextInBox(
                    settingMinusControl(top),
                    smallFont,
                    "-",
                    354f,
                    top + 12f,
                    36f,
                    44f,
                    Align.center
            );
            drawTextInBox(smallFont, value, 388f, top + 12f, 56f, 44f, Align.center);
            drawAnimatedTextInBox(
                    settingPlusControl(top),
                    smallFont,
                    "+",
                    442f,
                    top + 12f,
                    36f,
                    44f,
                    Align.center
            );
        } else {
            drawAnimatedTextInBox(
                    settingToggleControl(top),
                    smallFont,
                    value,
                    370f,
                    top + 13f,
                    108f,
                    42f,
                    Align.center
            );
        }
    }

    private String settingMinusControl(float top) {
        if (top == 218f) {
            return CONTROL_SETTINGS_SOUND_MINUS;
        }
        if (top == 578f) {
            return CONTROL_SETTINGS_SPEED_MINUS;
        }
        if (top == 668f) {
            return CONTROL_SETTINGS_TIMER_MINUS;
        }
        return CONTROL_SETTINGS_HOLD_MINUS;
    }

    private String settingPlusControl(float top) {
        if (top == 218f) {
            return CONTROL_SETTINGS_SOUND_PLUS;
        }
        if (top == 578f) {
            return CONTROL_SETTINGS_SPEED_PLUS;
        }
        if (top == 668f) {
            return CONTROL_SETTINGS_TIMER_PLUS;
        }
        return CONTROL_SETTINGS_HOLD_PLUS;
    }

    private String settingToggleControl(float top) {
        if (top == 308f) {
            return CONTROL_SETTINGS_MERGE_VIBRATE;
        }
        if (top == 398f) {
            return CONTROL_SETTINGS_DROP_VIBRATE;
        }
        return CONTROL_SETTINGS_SCORE_VIBRATE;
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
        drawAnimatedTextInBox(
                CONTROL_OVERLAY_CLOSE,
                smallFont,
                "返回",
                430f,
                124f,
                70f,
                46f,
                Align.center
        );
        GameProfileStore.History history = profileStore.history();
        smallFont.setColor(TEXT_MUTED);
        drawTextInBox(
                smallFont,
                "最佳成绩",
                82f,
                198f,
                120f,
                26f,
                Align.left
        );
        normalFont.setColor(PLAYER_SCORE_DARK);
        drawTextInBox(
                normalFont,
                "全模式最高  " + history.highScore(),
                82f,
                220f,
                396f,
                34f,
                Align.left
        );
        smallFont.setColor(TEXT_PRIMARY);
        String modeBests = "单人 " + history.highestSoloScore()
                + "  ·  对战 " + history.highestVersusScore()
                + "  ·  演示 " + history.highestAiDemoScore();
        drawTextInBox(
                smallFont,
                fitText(smallFont, modeBests, 396f),
                82f,
                252f,
                396f,
                26f,
                Align.left
        );

        smallFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                smallFont,
                "累计 " + history.totalGames()
                        + " 局  ·  单局最多大西瓜 "
                        + history.maxWatermelonsInGame(),
                82f,
                302f,
                396f,
                28f,
                Align.left
        );
        drawTextInBox(
                smallFont,
                "对战  " + history.versusWins() + "胜  "
                        + history.versusLosses() + "负  "
                        + history.versusDraws() + "平",
                82f,
                332f,
                396f,
                26f,
                Align.left
        );

        List<GameProfileStore.GameRecord> records =
                profileStore.gameRecords();
        int pageCount = historyPageCount(records.size());
        historyPageIndex = MathUtils.clamp(
                historyPageIndex,
                0,
                pageCount - 1
        );
        smallFont.setColor(TEXT_MUTED);
        drawTextInBox(
                smallFont,
                "对局列表  " + (historyPageIndex + 1)
                        + " / " + pageCount,
                72f,
                380f,
                416f,
                30f,
                Align.left
        );

        int first = historyPageIndex * HISTORY_RECORDS_PER_PAGE;
        int visible = Math.min(
                HISTORY_RECORDS_PER_PAGE,
                records.size() - first
        );
        if (visible <= 0) {
            normalFont.setColor(TEXT_MUTED);
            drawTextInBox(
                    normalFont,
                    "完成新对局后会逐条显示在这里",
                    82f,
                    420f,
                    396f,
                    164f,
                    Align.center
            );
        } else {
            for (int row = 0; row < visible; row++) {
                drawGameRecord(
                        records.get(first + row),
                        420f + row * 90f
                );
            }
        }

        boolean hasPrevious = historyPageIndex > 0;
        boolean hasNext = historyPageIndex + 1 < pageCount;
        smallFont.setColor(hasPrevious ? TEXT_PRIMARY : TEXT_MUTED);
        drawAnimatedTextInBox(
                CONTROL_HISTORY_PREVIOUS,
                smallFont,
                "上一页",
                154f,
                798f,
                110f,
                48f,
                Align.center
        );
        smallFont.setColor(hasNext ? TEXT_PRIMARY : TEXT_MUTED);
        drawAnimatedTextInBox(
                CONTROL_HISTORY_NEXT,
                smallFont,
                "下一页",
                296f,
                798f,
                110f,
                48f,
                Align.center
        );

        normalFont.setColor(
                historyResetConfirmSeconds > 0f ? DANGER : TEXT_PRIMARY
        );
        drawAnimatedTextInBox(
                CONTROL_HISTORY_RESET,
                normalFont,
                historyResetConfirmSeconds > 0f
                        ? "再次点击确认重置"
                        : "重置记录",
                158f,
                900f,
                244f,
                58f,
                Align.center
        );
    }

    private int historyPageCount(int recordCount) {
        return Math.max(
                1,
                (Math.max(0, recordCount) + HISTORY_RECORDS_PER_PAGE - 1)
                        / HISTORY_RECORDS_PER_PAGE
        );
    }

    private Color historyRecordColor(GameProfileStore.GameMode mode) {
        if (mode == GameProfileStore.GameMode.SOLO) {
            return PLAYER_TINT_SOFT;
        }
        if (mode == GameProfileStore.GameMode.DUEL) {
            return AI_TINT_SOFT;
        }
        return ACCENT_SOFT;
    }

    private void drawGameRecord(
            GameProfileStore.GameRecord record,
            float top) {
        normalFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                normalFont,
                gameRecordTitle(record),
                82f,
                top + 7f,
                220f,
                30f,
                Align.left
        );
        normalFont.setColor(
                record.mode() == GameProfileStore.GameMode.DUEL
                        ? AI_SCORE_DARK
                        : PLAYER_SCORE_DARK
        );
        drawTextInBox(
                normalFont,
                gameRecordScore(record),
                306f,
                top + 7f,
                172f,
                30f,
                Align.right
        );
        smallFont.setColor(TEXT_MUTED);
        String detail = formatHistoryTime(
                record.completedAtEpochMillis())
                + "  ·  大西瓜 " + record.watermelonsCreated()
                + "  ·  投放 " + record.dropCount();
        drawTextInBox(
                smallFont,
                fitText(smallFont, detail, 396f),
                82f,
                top + 43f,
                396f,
                26f,
                Align.left
        );
    }

    private String gameRecordTitle(
            GameProfileStore.GameRecord record) {
        if (record.mode() == GameProfileStore.GameMode.SOLO) {
            return "单人模式";
        }
        if (record.mode() == GameProfileStore.GameMode.AI_DEMO) {
            return "AI演示";
        }
        switch (record.result()) {
            case WIN:
                return "挑战AI · 胜";
            case LOSS:
                return "挑战AI · 负";
            case DRAW:
                return "挑战AI · 平";
            default:
                return "挑战AI";
        }
    }

    private String gameRecordScore(
            GameProfileStore.GameRecord record) {
        if (record.mode() == GameProfileStore.GameMode.DUEL) {
            return record.score() + " : " + record.opponentScore();
        }
        return record.score() + "分";
    }

    private String formatHistoryTime(long epochMillis) {
        if (epochMillis <= 0L) {
            return "旧版本记录";
        }
        java.text.SimpleDateFormat format =
                new java.text.SimpleDateFormat(
                        "MM-dd HH:mm",
                        java.util.Locale.CHINA
                );
        return format.format(new java.util.Date(epochMillis));
    }

    private String onOff(boolean value) {
        return value ? "开启" : "关闭";
    }

    private String formatOneDecimal(float value) {
        return String.format(java.util.Locale.ROOT, "%.1f", value);
    }

    private void drawAnimatedButton(
            String controlId,
            float left,
            float top,
            float width,
            float height,
            float radius,
            Color color) {
        UiMotionController.Visual visual = uiMotion.visual(controlId);
        float scaledWidth = width * visual.scale;
        float scaledHeight = height * visual.scale;
        float animatedLeft = left
                + (width - scaledWidth) * 0.5f
                + visual.offsetX;
        float animatedTop = top
                + (height - scaledHeight) * 0.5f
                + visual.offsetY;
        float animatedRadius = Math.min(
                radius * visual.scale,
                Math.min(scaledWidth, scaledHeight) * 0.5f
        );
        roundedRectTop(
                animatedLeft,
                animatedTop + 5f - visual.pressure * 3f,
                scaledWidth,
                scaledHeight,
                animatedRadius,
                CARD_SHADOW
        );
        scratchColor.set(color);
        scratchColor.lerp(
                TEXT_PRIMARY,
                visual.pressure * 0.075f
        );
        roundedRectTop(
                animatedLeft,
                animatedTop,
                scaledWidth,
                scaledHeight,
                animatedRadius,
                scratchColor
        );
        if (visual.releasePulse > 0f) {
            scratchColor.set(
                    1f,
                    1f,
                    1f,
                    visual.releasePulse * 0.16f
            );
            shapes.circle(
                    animatedLeft + scaledWidth * 0.5f,
                    toRenderY(animatedTop + scaledHeight * 0.5f),
                    Math.min(scaledWidth, scaledHeight)
                            * (0.22f + (1f - visual.releasePulse) * 0.26f),
                    28
            );
        }
    }

    private void drawAnimatedTextInBox(
            String controlId,
            BitmapFont font,
            String text,
            float left,
            float top,
            float width,
            float height,
            int alignment) {
        UiMotionController.Visual visual = uiMotion.visual(controlId);
        drawTextInBox(
                font,
                text,
                left + visual.offsetX,
                top + visual.offsetY,
                width,
                height,
                alignment
        );
    }

    private void roundedRectTop(
            float x,
            float top,
            float width,
            float height,
            float radius,
            Color color) {
        float y = toRenderY(top + height);
        float safeRadius = MathUtils.clamp(
                radius,
                0f,
                Math.min(width, height) * 0.5f
        );
        shapes.setColor(color);
        if (safeRadius <= 0f) {
            shapes.rect(x, y, width, height);
            return;
        }
        /*
         * These primitives meet only at their boundaries. The previous two full rectangles plus
         * four full circles overlapped across the card, which made a nominally 60%-opaque bubble
         * composite several times and appear almost solid.
         */
        shapes.rect(
                x + safeRadius,
                y,
                width - 2f * safeRadius,
                height
        );
        shapes.rect(
                x,
                y + safeRadius,
                safeRadius,
                height - 2f * safeRadius
        );
        shapes.rect(
                x + width - safeRadius,
                y + safeRadius,
                safeRadius,
                height - 2f * safeRadius
        );
        shapes.arc(
                x + safeRadius,
                y + safeRadius,
                safeRadius,
                180f,
                90f,
                10
        );
        shapes.arc(
                x + width - safeRadius,
                y + safeRadius,
                safeRadius,
                270f,
                90f,
                10
        );
        shapes.arc(
                x + width - safeRadius,
                y + height - safeRadius,
                safeRadius,
                0f,
                90f,
                10
        );
        shapes.arc(
                x + safeRadius,
                y + height - safeRadius,
                safeRadius,
                90f,
                90f,
                10
        );
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

    private void drawWrappedTextInBox(
            BitmapFont font,
            String text,
            float left,
            float top,
            float width,
            float height,
            int alignment) {
        glyphLayout.setText(
                font,
                text == null ? "" : text,
                font.getColor(),
                width,
                alignment,
                true
        );
        float visibleTop = top + (height - glyphLayout.height) * 0.5f;
        font.draw(batch, glyphLayout, left, toRenderY(visibleTop));
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
        physicsDecisionClock.resetStable();
        previewAnchorX = previewX;
        if (enabled) {
            aiLoadingSeconds = 0f;
            dropCooldown = 0f;
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
        if (appScreen != AppScreen.HOME
                || (page != OverlayPage.SETTINGS
                && page != OverlayPage.HISTORY)) {
            return;
        }
        overlayPage = page;
        activeDragPointer = -1;
        historyResetConfirmSeconds = 0f;
        if (page == OverlayPage.HISTORY) {
            historyPageIndex = 0;
        }
        uiMotion.cancelAll();
    }

    private void closeOverlay() {
        overlayPage = OverlayPage.NONE;
        activeDragPointer = -1;
        historyResetConfirmSeconds = 0f;
        uiMotion.cancelAll();
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
        if (!isInside(x, y, 158f, 900f, 244f, 58f)) {
            return;
        }
        if (historyResetConfirmSeconds <= 0f) {
            historyResetConfirmSeconds = 3f;
            return;
        }
        profileStore.resetHistory();
        clearCompletedSavedSessionAfterHistoryReset();
        historyPageIndex = 0;
        bestScore = 0;
        displayedBestScore = Math.max(displayedScore, bestScore);
        historyResetConfirmSeconds = 0f;
    }

    /**
     * A completed draft may survive the tiny window between durable history recording and the
     * player's result acknowledgement. Once the user explicitly resets history, retaining that
     * draft would allow a later "continue" to recreate the deleted row and aggregates.
     * In-progress saves remain untouched.
     */
    private void clearCompletedSavedSessionAfterHistoryReset() {
        GameSessionStore.Session saved = sessionStore.load();
        if (saved == null) {
            return;
        }
        boolean completed;
        if (saved.mode() == GameSessionStore.Mode.DUEL) {
            completed = saved.duel().resultRecorded()
                    || saved.duel().match().outcome()
                    != DuelMatch.Outcome.IN_PROGRESS;
        } else {
            completed = saved.single().resultRecorded()
                    || !saved.single().alive();
        }
        if (completed) {
            sessionStore.clear();
            if (saved.sessionId() == currentSessionId) {
                inMemorySession = false;
            }
        }
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
        aiSlide = null;
        duelAiSlide = null;
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
        float playerX = duelMatch.playerLane().previewX();
        if (duelAiArmed
                && !duelMatch.aiLane().submittedThisRound()) {
            boolean dropped = duelMatch.dropBoth(
                    playerX,
                    duelAiArmedX
            );
            duelAiArmed = false;
            if (dropped) {
                spawnDuelDropFeedback(DuelMatch.Side.PLAYER);
                spawnDuelDropFeedback(DuelMatch.Side.AI);
                showAiReaction(
                        AiMood.READY,
                        1.6f,
                        4
                );
                invalidateDuelDecision();
                markSessionDirty();
            }
            return;
        }
        if (duelMatch.dropPlayer(playerX)) {
            spawnDuelDropFeedback(DuelMatch.Side.PLAYER);
            markSessionDirty();
        }
    }

    /**
     * AI 先完成决策时只把悬浮水果移到目标列。玩家尚未提交则进入“就位等待”，
     * 由玩家抬手触发 dropBoth；玩家已经先投时则按 AI 自己的轨迹时序正常落下。
     */
    private void startDuelAiSlide(float x) {
        if (duelMatch == null
                || !duelMatch.roundOpen()
                || duelMatch.aiLane().submittedThisRound()) {
            return;
        }
        float target = FruitRules.clampDropX(
                x,
                duelMatch.currentLevel()
        );
        float start = duelMatch.aiLane().previewX();
        if (Math.abs(target - start) <= 0.25f) {
            armOrDropDuelAi(target);
            return;
        }
        duelAiSlide = new HorizontalSlide(start, target);
    }

    private void updateDuelAiSlide(float delta) {
        HorizontalSlide slide = duelAiSlide;
        if (slide == null || duelMatch == null) {
            duelAiSlide = null;
            return;
        }
        if (!duelMatch.roundOpen()
                || duelMatch.aiLane().submittedThisRound()
                || duelMatch.outcome()
                != DuelMatch.Outcome.IN_PROGRESS) {
            duelAiSlide = null;
            return;
        }
        duelMatch.setAiPreviewX(slide.advance(delta));
        if (!slide.finished()) {
            return;
        }
        float target = FruitRules.clampDropX(
                slide.targetX,
                duelMatch.currentLevel()
        );
        duelAiSlide = null;
        duelMatch.setAiPreviewX(target);
        armOrDropDuelAi(target);
    }

    private void armOrDropDuelAi(float x) {
        if (duelMatch == null
                || !duelMatch.roundOpen()
                || duelMatch.aiLane().submittedThisRound()) {
            return;
        }
        float target = FruitRules.clampDropX(
                x,
                duelMatch.currentLevel()
        );
        duelMatch.setAiPreviewX(target);
        if (duelMatch.playerLane().submittedThisRound()) {
            dropDuelAiAt(target);
            return;
        }
        duelAiArmed = true;
        duelAiArmedX = target;
        showAiReaction(
                AiMood.READY,
                2.2f,
                4
        );
        markSessionDirty();
    }

    private void dropDuelAiAt(float x) {
        if (duelMatch == null) {
            return;
        }
        if (duelMatch.dropAi(x)) {
            spawnDuelDropFeedback(DuelMatch.Side.AI);
            duelAiArmed = false;
            invalidateDuelDecision();
            markSessionDirty();
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

    private boolean beginControl(
            String id,
            int pointer,
            float x,
            float y,
            float left,
            float top,
            float width,
            float height) {
        return uiMotion.begin(
                id,
                new UiMotionController.Bounds(
                        left,
                        top,
                        width,
                        height
                ),
                pointer,
                x,
                y
        );
    }

    private boolean tryBeginUiControl(
            int pointer,
            float x,
            float y) {
        if (overlayPage == OverlayPage.EXIT_CONFIRM) {
            return beginControl(
                    CONTROL_EXIT_SAVE, pointer, x, y,
                    104f, 522f, 352f, 64f
            ) || beginControl(
                    CONTROL_EXIT_ABANDON, pointer, x, y,
                    104f, 608f, 352f, 64f
            ) || beginControl(
                    CONTROL_EXIT_CANCEL, pointer, x, y,
                    168f, 696f, 224f, 48f
            );
        }
        if (overlayPage == OverlayPage.NEW_GAME_CONFIRM) {
            return beginControl(
                    CONTROL_NEW_RESUME, pointer, x, y,
                    104f, 522f, 352f, 64f
            ) || beginControl(
                    CONTROL_NEW_CONFIRM, pointer, x, y,
                    104f, 608f, 352f, 64f
            ) || beginControl(
                    CONTROL_NEW_CANCEL, pointer, x, y,
                    168f, 696f, 224f, 48f
            );
        }
        if (overlayPage == OverlayPage.SETTINGS) {
            return beginControl(
                    CONTROL_OVERLAY_CLOSE, pointer, x, y,
                    430f, 124f, 70f, 46f
            ) || beginControl(
                    CONTROL_SETTINGS_SOUND_MINUS, pointer, x, y,
                    354f, 230f, 36f, 44f
            ) || beginControl(
                    CONTROL_SETTINGS_SOUND_PLUS, pointer, x, y,
                    442f, 230f, 36f, 44f
            ) || beginControl(
                    CONTROL_SETTINGS_MERGE_VIBRATE, pointer, x, y,
                    370f, 321f, 108f, 42f
            ) || beginControl(
                    CONTROL_SETTINGS_DROP_VIBRATE, pointer, x, y,
                    370f, 411f, 108f, 42f
            ) || beginControl(
                    CONTROL_SETTINGS_SCORE_VIBRATE, pointer, x, y,
                    370f, 501f, 108f, 42f
            ) || beginControl(
                    CONTROL_SETTINGS_SPEED_MINUS, pointer, x, y,
                    354f, 590f, 36f, 44f
            ) || beginControl(
                    CONTROL_SETTINGS_SPEED_PLUS, pointer, x, y,
                    442f, 590f, 36f, 44f
            ) || beginControl(
                    CONTROL_SETTINGS_TIMER_MINUS, pointer, x, y,
                    354f, 680f, 36f, 44f
            ) || beginControl(
                    CONTROL_SETTINGS_TIMER_PLUS, pointer, x, y,
                    442f, 680f, 36f, 44f
            ) || beginControl(
                    CONTROL_SETTINGS_HOLD_MINUS, pointer, x, y,
                    354f, 770f, 36f, 44f
            ) || beginControl(
                    CONTROL_SETTINGS_HOLD_PLUS, pointer, x, y,
                    442f, 770f, 36f, 44f
            ) || beginControl(
                    CONTROL_SETTINGS_RESET, pointer, x, y,
                    158f, 882f, 244f, 58f
            );
        }
        if (overlayPage == OverlayPage.HISTORY) {
            return beginControl(
                    CONTROL_OVERLAY_CLOSE, pointer, x, y,
                    430f, 124f, 70f, 46f
            ) || beginControl(
                    CONTROL_HISTORY_PREVIOUS, pointer, x, y,
                    154f, 798f, 110f, 48f
            ) || beginControl(
                    CONTROL_HISTORY_NEXT, pointer, x, y,
                    296f, 798f, 110f, 48f
            ) || beginControl(
                    CONTROL_HISTORY_RESET, pointer, x, y,
                    158f, 900f, 244f, 58f
            );
        }
        if (appScreen == AppScreen.HOME) {
            boolean hasResume =
                    sessionStore.hasSavedSession() || inMemorySession;
            return (hasResume && beginControl(
                    CONTROL_HOME_RESUME, pointer, x, y,
                    298f, 210f, 184f, 58f
            )) || beginControl(
                    CONTROL_HOME_SOLO, pointer, x, y,
                    HOME_CARD_LEFT, HOME_SOLO_TOP,
                    HOME_CARD_WIDTH, HOME_CARD_HEIGHT
            ) || beginControl(
                    CONTROL_HOME_DUEL, pointer, x, y,
                    HOME_CARD_LEFT, HOME_DUEL_TOP,
                    HOME_CARD_WIDTH, HOME_CARD_HEIGHT
            ) || beginControl(
                    CONTROL_HOME_DEMO, pointer, x, y,
                    HOME_CARD_LEFT, HOME_DEMO_TOP,
                    HOME_CARD_WIDTH, HOME_CARD_HEIGHT
            ) || beginControl(
                    CONTROL_HOME_HISTORY, pointer, x, y,
                    HOME_CARD_LEFT, HOME_UTILITY_TOP,
                    HOME_UTILITY_WIDTH, HOME_UTILITY_HEIGHT
            ) || beginControl(
                    CONTROL_HOME_SETTINGS, pointer, x, y,
                    HOME_CARD_LEFT + 238f, HOME_UTILITY_TOP,
                    HOME_UTILITY_WIDTH, HOME_UTILITY_HEIGHT
            );
        }
        if (currentResultVisible()) {
            if (gameMode == GameMode.DUEL) {
                return beginControl(
                        CONTROL_RESULT_CONFIRM, pointer, x, y,
                        146f, 658f, 268f, 62f
                );
            }
            return beginControl(
                    CONTROL_RESULT_CONFIRM, pointer, x, y,
                    RESULT_BUTTON_LEFT, RESULT_BUTTON_TOP,
                    RESULT_BUTTON_WIDTH, RESULT_BUTTON_HEIGHT
            );
        }
        if (gameMode == GameMode.DUEL
                && duelMatch != null
                && duelMatch.outcome()
                != DuelMatch.Outcome.IN_PROGRESS) {
            return beginControl(
                    CONTROL_DUEL_FOREGROUND, pointer, x, y,
                    AI_TOGGLE_LEFT, AI_TOGGLE_TOP,
                    AI_TOGGLE_WIDTH, AI_TOGGLE_HEIGHT
            );
        }
        return beginControl(
                CONTROL_GAME_BACK, pointer, x, y,
                GAME_BACK_LEFT, GAME_BACK_TOP,
                GAME_BACK_SIZE, GAME_BACK_SIZE
        ) || (gameMode == GameMode.DUEL && beginControl(
                CONTROL_DUEL_FOREGROUND, pointer, x, y,
                AI_TOGGLE_LEFT, AI_TOGGLE_TOP,
                AI_TOGGLE_WIDTH, AI_TOGGLE_HEIGHT
        ));
    }

    private void dispatchControl(String controlId) {
        if (controlId == null) {
            return;
        }
        if (CONTROL_HOME_SOLO.equals(controlId)) {
            requestStartMode(GameMode.SOLO);
        } else if (CONTROL_HOME_DUEL.equals(controlId)) {
            requestStartMode(GameMode.DUEL);
        } else if (CONTROL_HOME_DEMO.equals(controlId)) {
            requestStartMode(GameMode.AI_DEMO);
        } else if (CONTROL_HOME_RESUME.equals(controlId)) {
            continueSavedSession();
        } else if (CONTROL_HOME_SETTINGS.equals(controlId)) {
            openOverlay(OverlayPage.SETTINGS);
        } else if (CONTROL_HOME_HISTORY.equals(controlId)) {
            openOverlay(OverlayPage.HISTORY);
        } else if (CONTROL_GAME_BACK.equals(controlId)) {
            openExitPrompt();
        } else if (CONTROL_DUEL_FOREGROUND.equals(controlId)) {
            toggleDuelForeground();
        } else if (CONTROL_RESULT_CONFIRM.equals(controlId)) {
            acknowledgeResult();
        } else if (CONTROL_OVERLAY_CLOSE.equals(controlId)) {
            closeOverlay();
        } else if (CONTROL_EXIT_SAVE.equals(controlId)) {
            returnHomeKeepingSession();
        } else if (CONTROL_EXIT_ABANDON.equals(controlId)) {
            abandonCurrentSession();
        } else if (CONTROL_EXIT_CANCEL.equals(controlId)) {
            closeOverlay();
        } else if (CONTROL_NEW_CONFIRM.equals(controlId)) {
            GameMode requested = pendingNewMode;
            if (requested != null) {
                startNewMode(requested);
            }
        } else if (CONTROL_NEW_RESUME.equals(controlId)) {
            continueSavedSession();
        } else if (CONTROL_NEW_CANCEL.equals(controlId)) {
            pendingNewMode = null;
            closeOverlay();
        } else if (CONTROL_SETTINGS_RESET.equals(controlId)) {
            profileStore.resetSettings();
            settings = profileStore.settings();
        } else if (CONTROL_HISTORY_PREVIOUS.equals(controlId)) {
            historyPageIndex = Math.max(0, historyPageIndex - 1);
        } else if (CONTROL_HISTORY_NEXT.equals(controlId)) {
            int pageCount = historyPageCount(
                    profileStore.gameRecords().size()
            );
            historyPageIndex = Math.min(
                    pageCount - 1,
                    historyPageIndex + 1
            );
        } else if (CONTROL_HISTORY_RESET.equals(controlId)) {
            handleHistoryResetAction();
        } else {
            dispatchSettingsControl(controlId);
        }
    }

    private void dispatchSettingsControl(String controlId) {
        if (CONTROL_SETTINGS_SOUND_MINUS.equals(controlId)
                || CONTROL_SETTINGS_SOUND_PLUS.equals(controlId)) {
            float delta = CONTROL_SETTINGS_SOUND_MINUS.equals(controlId)
                    ? -0.1f : 0.1f;
            profileStore.setSoundVolume(
                    settings.soundVolume() + delta
            ).save();
        } else if (CONTROL_SETTINGS_MERGE_VIBRATE.equals(controlId)) {
            profileStore.setVibrateOnMerge(
                    !settings.vibrateOnMerge()
            ).save();
        } else if (CONTROL_SETTINGS_DROP_VIBRATE.equals(controlId)) {
            profileStore.setVibrateOnDrop(
                    !settings.vibrateOnDrop()
            ).save();
        } else if (CONTROL_SETTINGS_SCORE_VIBRATE.equals(controlId)) {
            profileStore.setVibrateOnScoreCollect(
                    !settings.vibrateOnScoreCollect()
            ).save();
        } else if (CONTROL_SETTINGS_SPEED_MINUS.equals(controlId)
                || CONTROL_SETTINGS_SPEED_PLUS.equals(controlId)) {
            float delta = CONTROL_SETTINGS_SPEED_MINUS.equals(controlId)
                    ? -0.25f : 0.25f;
            profileStore.setGameSpeed(
                    settings.gameSpeed() + delta
            ).save();
        } else if (CONTROL_SETTINGS_TIMER_MINUS.equals(controlId)
                || CONTROL_SETTINGS_TIMER_PLUS.equals(controlId)) {
            float delta = CONTROL_SETTINGS_TIMER_MINUS.equals(controlId)
                    ? -1f : 1f;
            profileStore.setVersusDropSeconds(
                    settings.versusDropSeconds() + delta
            ).save();
        } else if (CONTROL_SETTINGS_HOLD_MINUS.equals(controlId)
                || CONTROL_SETTINGS_HOLD_PLUS.equals(controlId)) {
            float delta = CONTROL_SETTINGS_HOLD_MINUS.equals(controlId)
                    ? -1f : 1f;
            profileStore.setResultHoldSeconds(
                    settings.resultHoldSeconds() + delta
            ).save();
        } else {
            return;
        }
        settings = profileStore.settings();
    }

    private void handleHistoryResetAction() {
        if (historyResetConfirmSeconds <= 0f) {
            historyResetConfirmSeconds = 3f;
            return;
        }
        profileStore.resetHistory();
        clearCompletedSavedSessionAfterHistoryReset();
        historyPageIndex = 0;
        bestScore = 0;
        displayedBestScore = Math.max(displayedScore, bestScore);
        historyResetConfirmSeconds = 0f;
    }

    @Override
    public boolean touchDown(int screenX, int screenY, int pointer, int button) {
        if (!touchIsInsideViewport(screenX, screenY)) {
            return false;
        }
        if (uiMotion.hasActiveControl()
                || (activeDragPointer >= 0
                && pointer != activeDragPointer)) {
            return false;
        }
        updateTouchPoint(screenX, screenY);
        if (tryBeginUiControl(pointer, touchPoint.x, touchPoint.y)) {
            return true;
        }
        if (overlayPage != OverlayPage.NONE) {
            return true;
        }
        if (appScreen == AppScreen.HOME || currentResultVisible()) {
            return true;
        }
        if (gameMode == GameMode.DUEL
                && duelMatch != null
                && duelMatch.outcome()
                != DuelMatch.Outcome.IN_PROGRESS) {
            return true;
        }
        if (scoreHudContains(touchPoint.x, touchPoint.y)) {
            // The bottom score card is display-only but must not become a hidden drop surface.
            return true;
        }
        if (gameMode == GameMode.SOLO
                && canManualDragCurrent()
                && touchPoint.y >= manualInputTop()
                && touchPoint.y <= sceneFloorY()) {
            activeDragPointer = pointer;
            previewX = FruitRules.clampDropX(touchPoint.x, currentLevel);
            return true;
        }
        if (gameMode == GameMode.DUEL
                && duelCanPlayerDrag()
                && touchPoint.y >= manualInputTop()
                && touchPoint.y <= sceneFloorY()) {
            activeDragPointer = pointer;
            setDuelPlayerPreviewX(touchPoint.x);
            return true;
        }
        // AI 演示模式明确禁止人工接管，吞掉棋盘触摸但不改变预览位置。
        return gameMode == GameMode.AI_DEMO;
    }

    @Override
    public boolean touchDragged(int screenX, int screenY, int pointer) {
        if (uiMotion.hasActiveControl()
                && pointer == uiMotion.activePointer()) {
            updateTouchPoint(screenX, screenY);
            uiMotion.drag(pointer, touchPoint.x, touchPoint.y);
            return true;
        }
        if (pointer != activeDragPointer
                || !touchIsInsideViewport(screenX, screenY)) {
            return false;
        }
        updateTouchPoint(screenX, screenY);
        if (gameMode == GameMode.SOLO && canManualDragCurrent()) {
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
        if (uiMotion.hasActiveControl()
                && pointer == uiMotion.activePointer()) {
            updateTouchPoint(screenX, screenY);
            String action = uiMotion.release(
                    pointer,
                    touchPoint.x,
                    touchPoint.y
            );
            dispatchControl(action);
            return true;
        }
        if (pointer != activeDragPointer) {
            return false;
        }
        activeDragPointer = -1;
        if (!touchIsInsideViewport(screenX, screenY)) {
            return true;
        }
        updateTouchPoint(screenX, screenY);
        if (gameMode == GameMode.SOLO
                && canManualDropCurrent()
                && touchPoint.y >= manualInputTop()
                && touchPoint.y <= sceneFloorY()
                && !scoreHudContains(touchPoint.x, touchPoint.y)) {
            previewX = FruitRules.clampDropX(touchPoint.x, currentLevel);
            dropCurrent(previewX);
            return true;
        }
        if (gameMode == GameMode.DUEL
                && duelCanPlayerDrop()
                && touchPoint.y >= manualInputTop()
                && touchPoint.y <= sceneFloorY()
                && !scoreHudContains(touchPoint.x, touchPoint.y)) {
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
        if (uiMotion.cancel(pointer)) {
            return true;
        }
        if (pointer == activeDragPointer) {
            activeDragPointer = -1;
            return true;
        }
        return false;
    }

    @Override
    public boolean keyDown(int keycode) {
        if (keycode == Input.Keys.BACK
                || keycode == Input.Keys.ESCAPE) {
            if (overlayPage == OverlayPage.NEW_GAME_CONFIRM) {
                pendingNewMode = null;
                closeOverlay();
            } else if (overlayPage != OverlayPage.NONE) {
                closeOverlay();
            } else if (appScreen == AppScreen.HOME) {
                Gdx.app.exit();
            } else if (currentResultVisible()) {
                // 结算必须点击画面中的“确认结算”，系统返回键不代替确认。
                uiMotion.cancelAll();
                return true;
            } else if (gameMode == GameMode.DUEL
                    && duelMatch != null
                    && duelMatch.outcome()
                    != DuelMatch.Outcome.IN_PROGRESS) {
                uiMotion.cancelAll();
                return true;
            } else {
                openExitPrompt();
            }
            return true;
        }
        if (overlayPage != OverlayPage.NONE) {
            return true;
        }
        if (appScreen != AppScreen.GAME || currentResultVisible()) {
            return false;
        }
        if (keycode == Input.Keys.SPACE || keycode == Input.Keys.ENTER) {
            if (gameMode == GameMode.DUEL) {
                dropDuelPlayer();
            } else if (gameMode == GameMode.SOLO) {
                dropCurrent(previewX);
            }
            return true;
        }
        if (keycode == Input.Keys.A || keycode == Input.Keys.LEFT) {
            if (gameMode == GameMode.DUEL) {
                moveDuelPlayerPreview(-14f);
            } else if (gameMode == GameMode.SOLO && waiting) {
                previewX = FruitRules.clampDropX(previewX - 14f, currentLevel);
            }
            return true;
        }
        if (keycode == Input.Keys.D || keycode == Input.Keys.RIGHT) {
            if (gameMode == GameMode.DUEL) {
                moveDuelPlayerPreview(14f);
            } else if (gameMode == GameMode.SOLO && waiting) {
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

    /**
     * 与 java.util.Random 相同的 48 位 LCG，但公开可持久化内部状态。
     * 单人存档因此不仅恢复当前可见队列，后续随机水果也与保存前完全一致。
     */
    private static final class StatefulQueueRandom extends Random {
        private static final long serialVersionUID = 1L;
        private static final long MULTIPLIER = 0x5DEECE66DL;
        private static final long ADDEND = 0xBL;
        private static final long MASK = (1L << 48) - 1L;

        private long state;

        private StatefulQueueRandom() {
            this(System.nanoTime() ^ System.currentTimeMillis());
        }

        private StatefulQueueRandom(long seed) {
            super(0L);
            setSeed(seed);
        }

        @Override
        public synchronized void setSeed(long seed) {
            state = (seed ^ MULTIPLIER) & MASK;
        }

        @Override
        protected synchronized int next(int bits) {
            state = (state * MULTIPLIER + ADDEND) & MASK;
            return (int) (state >>> (48 - bits));
        }

        private synchronized long state() {
            return state;
        }

        private synchronized void restoreState(long savedState) {
            if ((savedState & ~MASK) != 0L) {
                throw new IllegalArgumentException(
                        "queue random state must fit in 48 bits"
                );
            }
            state = savedState;
        }
    }

    private enum GameMode {
        SOLO,
        AI_DEMO,
        // Kept temporarily for legacy preview helpers; new sessions never use it.
        CLASSIC,
        DUEL
    }

    private enum AppScreen {
        HOME,
        GAME
    }

    private enum OverlayPage {
        NONE,
        SETTINGS,
        HISTORY,
        EXIT_CONFIRM,
        NEW_GAME_CONFIRM
    }

    private enum AiMood {
        THINKING,
        HESITATING,
        WELCOME,
        READY,
        HAPPY,
        SURPRISED,
        WORRIED
    }

    private enum AiState {
        MANUAL("手动"),
        OBSERVING("观察中"),
        THINKING("思考中"),
        COMMITTING("投放中"),
        GAME_OVER("已结束");

        private final String label;

        AiState(String label) {
            this.label = label;
        }
    }

    private static final class AiReaction {
        private final String text;
        private final String emoticon;
        private final AiMood mood;
        private final float duration;
        private final int priority;
        private float age;

        private AiReaction(
                String text,
                String emoticon,
                AiMood mood,
                float duration,
                int priority) {
            this.text = text;
            this.emoticon = emoticon;
            this.mood = mood;
            this.duration = duration;
            this.priority = priority;
        }
    }

    /**
     * 模型选定目标后使用的单段横移动画。这里只做确定性的平滑过渡，
     * 不引入停顿、试探、回拉或随机抖动，动画结束即投放。
     */
    private static final class HorizontalSlide {
        private final float startX;
        private final float targetX;
        private final float duration;
        private float elapsed;

        private HorizontalSlide(float startX, float targetX) {
            this.startX = startX;
            this.targetX = targetX;
            float normalizedDistance = MathUtils.clamp(
                    Math.abs(targetX - startX) / FruitRules.BOARD_WIDTH,
                    0f,
                    1f
            );
            this.duration = MathUtils.lerp(
                    AI_SLIDE_MIN_SECONDS,
                    AI_SLIDE_MAX_SECONDS,
                    (float) Math.sqrt(normalizedDistance)
            );
        }

        private float advance(float delta) {
            elapsed = Math.min(
                    duration,
                    elapsed + Math.max(0f, delta)
            );
            float progress = duration <= 0f ? 1f : elapsed / duration;
            float eased = progress * progress * (3f - 2f * progress);
            return MathUtils.lerp(startX, targetX, eased);
        }

        private boolean finished() {
            return elapsed >= duration;
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

    private static final class AiEmotionPulse {
        private final String emoticon;
        private final AiMood mood;
        private final float duration;
        private final int priority;
        private float age;

        private AiEmotionPulse(
                String emoticon,
                AiMood mood,
                float duration,
                int priority) {
            this.emoticon = emoticon;
            this.mood = mood;
            this.duration = duration;
            this.priority = priority;
        }
    }

    private static final class PendingAiEmotion {
        private final AiMood mood;
        private final int priority;
        private float age;

        private PendingAiEmotion(AiMood mood, int priority) {
            this.mood = mood;
            this.priority = priority;
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
        private boolean aiReactionOffered;

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
        private final boolean boardAnchored;

        private Particle(
                float x,
                float y,
                float vx,
                float vy,
                float life,
                Color color,
                float radius,
                float gravity,
                float drag,
                boolean boardAnchored) {
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
            this.boardAnchored = boardAnchored;
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
        private final boolean boardAnchored;

        private Ring(
                float x,
                float y,
                float radius,
                float speed,
                float life,
                Color color,
                float dotRadius,
                boolean boardAnchored) {
            this.x = x;
            this.y = y;
            this.radius = radius;
            this.speed = speed;
            this.life = life;
            this.maxLife = life;
            this.color = new Color(color);
            this.dotRadius = dotRadius;
            this.boardAnchored = boardAnchored;
        }
    }
}
