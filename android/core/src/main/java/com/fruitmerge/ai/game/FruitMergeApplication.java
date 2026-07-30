package com.fruitmerge.ai.game;

import com.badlogic.gdx.ApplicationAdapter;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.InputProcessor;
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
    private static final float DROP_COOLDOWN_SECONDS = 0.36f;
    private static final float GAME_OVER_SECONDS = 2f;
    private static final float AI_LOADING_FALLBACK_SECONDS = 12f;
    private static final float AI_TOGGLE_LEFT = 378f;
    private static final float AI_TOGGLE_TOP = 18f;
    private static final float AI_TOGGLE_WIDTH = 154f;
    private static final float AI_TOGGLE_HEIGHT = 42f;

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

    private final AiService aiService;
    private final Random random = new Random();
    private final IntArray queue = new IntArray();
    private final Array<Particle> particles = new Array<>();
    private final Array<Ring> rings = new Array<>();
    private final Vector3 touchPoint = new Vector3();

    private OrthographicCamera camera;
    private FitViewport viewport;
    private SpriteBatch batch;
    private ShapeRenderer shapes;
    private BitmapFont smallFont;
    private BitmapFont normalFont;
    private BitmapFont titleFont;
    private GlyphLayout glyphLayout;
    private TextureRegion[] fruitTextures;
    private FruitPhysicsWorld physics;

    private int score;
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
    private long decisionEpoch;
    private MotionPlan motionPlan;
    private AiState aiState = AiState.OBSERVING;
    private String aiDetail = "starting";

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
         */
        smallFont = createFont(0.25f);
        normalFont = createFont(0.38f);
        titleFont = createFont(0.47f);
        glyphLayout = new GlyphLayout();
        loadFruitTextures();

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
        float delta = Math.min(Gdx.graphics.getDeltaTime(), 0.05f);
        elapsedSeconds += delta;
        updateGame(delta);
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
        if (batch != null) {
            batch.dispose();
        }
        if (shapes != null) {
            shapes.dispose();
        }
    }

    private BitmapFont createFont(float scale) {
        BitmapFont font = new BitmapFont(
                Gdx.files.internal("fonts/ui-nunito.fnt")
        );
        font.getData().setScale(scale);
        font.setColor(TEXT_PRIMARY);
        font.setUseIntegerPositions(false);
        font.getRegion().getTexture().setFilter(
                Texture.TextureFilter.Linear,
                Texture.TextureFilter.Linear
        );
        return font;
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
        lastScore = 0;
        stepCount = 0;
        stableSeconds = 0f;
        dangerSeconds = 0f;
        aiLoadingSeconds = 0f;
        dropCooldown = 0.18f;
        previewX = FruitRules.BOARD_WIDTH / 2f;
        alive = true;
        waiting = true;
        currentLevel = queue.first();
        aiState = aiEnabled ? AiState.OBSERVING : AiState.MANUAL;
        aiDetail = aiService.isAiReady()
                ? "model ready"
                : sanitizeStatus(aiService.aiRuntimeStatus());
        particles.clear();
        rings.clear();
    }

    private void fillQueue() {
        while (queue.size < FruitRules.QUEUE_LENGTH) {
            queue.add(
                    FruitRules.SPAWN_MIN_LEVEL
                            + random.nextInt(
                            FruitRules.SPAWN_MAX_LEVEL
                                    - FruitRules.SPAWN_MIN_LEVEL + 1)
            );
        }
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
            stableSeconds = 0f;
        }

        boolean stable = physics.isStable();
        if (stable) {
            stableSeconds += delta;
        } else {
            stableSeconds = 0f;
            cancelPendingDecision("board moving");
        }

        if (!waiting || dropCooldown > 0f) {
            return;
        }
        if (aiEnabled) {
            updateAi(delta);
        } else {
            aiState = AiState.MANUAL;
            aiDetail = "drag and release";
        }
    }

    private void consumeMergeEvents() {
        Array<FruitPhysicsWorld.MergeEvent> events = physics.drainMergeEvents();
        if (events.size > 0) {
            // 合成会改变图拓扑，即使新刚体瞬时速度为零，也必须重新累计稳定窗口，
            // 并让已经发出的异步决策失效。
            stableSeconds = 0f;
            cancelPendingDecision("board changed");
        }
        for (FruitPhysicsWorld.MergeEvent event : events) {
            lastScore = score;
            score += event.scoreDelta;
            bestScore = Math.max(bestScore, score);
            spawnMergeEffect(event.x, event.y, event.level);
            aiService.vibrate(event.level >= 8 ? 36 : 18);
        }
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
            cancelPendingDecision("game over");
            aiState = AiState.GAME_OVER;
            aiDetail = "tap to restart";
            aiService.vibrate(90);
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
        if (stableSeconds < FruitRules.STABLE_WINDOW_SECONDS) {
            aiState = AiState.OBSERVING;
            aiDetail = "waiting for a stable board";
            // 等待期间仍保留很轻的手指颤动，但不会改变最终离散动作。
            previewX = FruitRules.clampDropX(
                    previewX + tremor(0.32f),
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
            aiDetail = "loading AI model";
            previewX = FruitRules.clampDropX(
                    previewX + tremor(0.22f),
                    currentLevel
            );
            return;
        }
        requestAiDecision();
    }

    private void requestAiDecision() {
        aiRequestInFlight = true;
        aiState = AiState.THINKING;
        aiDetail = aiService.isAiReady() ? "reading the board" : "safe fallback";
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
                        aiDetail = "fallback: empty model decision";
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
                    aiDetail = "fallback: " + sanitizeStatus(message);
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
        float currentRadius = FruitRules.droppedPhysicsRadius(currentLevel);
        for (int action = 0; action < FruitRules.ACTION_COUNT; action++) {
            float x = FruitRules.actionDropX(action, currentLevel);
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
            float mergeUtility = firstLevel == currentLevel ? 0.8f : 0f;
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
                random
        );
        aiState = AiState.THINKING;
        aiDetail = "considering " + (decision.actionIndex + 1) + "/21";
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
        decisionEpoch += 1;
        motionPlan = null;
        aiRequestInFlight = false;
        aiState = aiEnabled ? AiState.OBSERVING : AiState.MANUAL;
        aiDetail = detail;
    }

    private void dropCurrent(float x) {
        if (!alive || !waiting || dropCooldown > 0f) {
            return;
        }
        x = FruitRules.clampDropX(x, currentLevel);
        float y = previewY(currentLevel);
        physics.addDroppedFruit(currentLevel, x, y);
        spawnDropEffect(x, y, currentLevel);
        aiService.vibrate(10);

        if (queue.size > 0) {
            queue.removeIndex(0);
        }
        fillQueue();
        waiting = false;
        stepCount += 1;
        dropCooldown = DROP_COOLDOWN_SECONDS;
        stableSeconds = 0f;
        decisionEpoch += 1;
        aiRequestInFlight = false;
        motionPlan = null;
        aiState = aiEnabled ? AiState.OBSERVING : AiState.MANUAL;
    }

    private float previewY(int level) {
        return FruitRules.SPAWN_Y
                - FruitRules.displayRadius(level)
                - PREVIEW_GAP
                + MathUtils.sin(elapsedSeconds * 5f) * 2f;
    }

    private float tremor(float amplitude) {
        // 每帧独立白噪声会显得像故障；两组低频正弦叠加少量随机跳变更像手指微颤。
        return MathUtils.sin(elapsedSeconds * 31f) * amplitude
                + MathUtils.sin(elapsedSeconds * 17f + 0.7f) * amplitude * 0.55f
                + (random.nextFloat() - 0.5f) * amplitude * 0.2f;
    }

    private void updateEffects(float delta) {
        for (int index = particles.size - 1; index >= 0; index--) {
            Particle particle = particles.get(index);
            particle.life -= delta;
            particle.x += particle.vx * delta;
            particle.y += particle.vy * delta;
            particle.vy += 180f * delta;
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
    }

    private void spawnDropEffect(float x, float y, int level) {
        Color color = fruitAccent(level);
        rings.add(new Ring(x, y, FruitRules.displayRadius(level) * 0.4f, 130f, 0.28f, color));
        for (int index = 0; index < 7; index++) {
            float angle = random.nextFloat() * MathUtils.PI2;
            float speed = 45f + random.nextFloat() * 80f;
            particles.add(new Particle(
                    x,
                    y,
                    MathUtils.cos(angle) * speed,
                    MathUtils.sin(angle) * speed,
                    0.25f + random.nextFloat() * 0.18f,
                    color
            ));
        }
    }

    private void spawnMergeEffect(float x, float y, int level) {
        Color color = fruitAccent(level);
        rings.add(new Ring(x, y, 12f, 260f, 0.52f, color));
        int count = Math.min(24, 10 + level);
        for (int index = 0; index < count; index++) {
            float angle = random.nextFloat() * MathUtils.PI2;
            float speed = 90f + random.nextFloat() * (120f + level * 10f);
            particles.add(new Particle(
                    x,
                    y,
                    MathUtils.cos(angle) * speed,
                    MathUtils.sin(angle) * speed,
                    0.38f + random.nextFloat() * 0.32f,
                    color
            ));
        }
    }

    private Color fruitAccent(int level) {
        float hue = (level * 0.087f + 0.12f) % 1f;
        return new Color(1f, 1f, 1f, 1f)
                .fromHsv(hue * 360f, 0.58f, 1f);
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
        drawEffects();
        shapes.end();

        /*
         * 所有装饰形状都在水果和文字之前完成。旧版最后使用 ShapeType.Line
         * 重画圆角，既会在 arc 圆心生成“方括号”短线，也会直接穿过文字。
         */
        batch.setProjectionMatrix(camera.combined);
        batch.begin();
        drawFruitBodies();
        drawPreviewAndQueue();
        drawText();
        batch.end();

        if (!alive) {
            drawGameOverOverlay();
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

        // 用“外层填色 + 内层填色”形成可靠圆角边框，不再调用 Line arc。
        roundedRectTop(28f, 68f, 112f, 64f, 12f, CARD_SHADOW);
        roundedRectTop(28f, 64f, 112f, 64f, 12f, BOARD_FRAME_SOFT);
        roundedRectTop(30f, 66f, 108f, 60f, 10f, SCORE_CARD);

        roundedRectTop(150f, 68f, 112f, 64f, 12f, CARD_SHADOW);
        roundedRectTop(150f, 64f, 112f, 64f, 12f, BOARD_FRAME_SOFT);
        roundedRectTop(152f, 66f, 108f, 60f, 10f, SCORE_CARD);

        roundedRectTop(272f, 68f, 260f, 64f, 12f, CARD_SHADOW);
        roundedRectTop(272f, 64f, 260f, 64f, 12f, BOARD_FRAME_SOFT);
        roundedRectTop(274f, 66f, 256f, 60f, 10f, NEXT_CARD);

        // 标题旁的小叶片与队列槽给顶部区域增加果园识别度，但不替换水果本身。
        drawLeaf(263f, 35f, 28f, 11f, -24f, LEAF_ACCENT);
        drawLeaf(280f, 31f, 24f, 9f, 28f, LEAF_ACCENT);
        shapes.setColor(DANGER.r, DANGER.g, DANGER.b, 0.68f);
        shapes.circle(278f, toRenderY(40f), 3.8f, 16);
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
                aiEnabled ? ACCENT_SOFT : SWITCH_OFF
        );

        /*
         * 棋盘只保留阴影、外框和内底三层。左右/底部不再额外画 wall rect，
         * 避免旧版在 floor 以下多出 20px 线条。
         */
        roundedRectTop(18f, 254f, 524f, 854f, 18f, CARD_SHADOW);
        roundedRectTop(18f, 248f, 524f, 860f, 18f, BOARD_FRAME);
        roundedRectTop(20f, 250f, 520f, 856f, 16f, BOARD_FRAME_SOFT);
        roundedRectTop(22f, 252f, 516f, 848f, 14f, BOARD_COLOR);
        shapes.setColor(
                ORCHARD_GLOW.r,
                ORCHARD_GLOW.g,
                ORCHARD_GLOW.b,
                0.045f
        );
        shapes.circle(88f, toRenderY(1018f), 58f, 36);
        shapes.circle(474f, toRenderY(1028f), 50f, 36);

        float dangerAlpha = dangerSeconds <= 0f
                ? 0.48f
                : 0.55f + MathUtils.sin(elapsedSeconds * 12f) * 0.25f;
        shapes.setColor(DANGER.r, DANGER.g, DANGER.b, dangerAlpha);
        float y = toRenderY(FruitRules.SPAWN_Y + 4f);
        for (float x = 30f; x < 530f; x += 20f) {
            shapes.rect(x, y - 1f, 11f, 2f);
        }

        if (waiting) {
            float guideAlpha = aiEnabled ? 0.19f : 0.13f;
            shapes.setColor(ACCENT.r, ACCENT.g, ACCENT.b, guideAlpha);
            float guideTop = previewY(currentLevel) + FruitRules.displayRadius(currentLevel);
            float guideBottom = FruitRules.FLOOR_Y;
            for (float screenY = guideTop + 14f;
                    screenY < guideBottom;
                    screenY += 26f) {
                shapes.circle(previewX, toRenderY(screenY), 1.7f, 10);
            }

            // 实心半透明外圆在水果之前绘制，水果本身会盖住中心形成干净的光环。
            shapes.setColor(
                    ACCENT.r,
                    ACCENT.g,
                    ACCENT.b,
                    aiEnabled ? 0.22f : 0.14f
            );
            shapes.circle(
                    previewX,
                    toRenderY(previewY(currentLevel)),
                    FruitRules.displayRadius(currentLevel) + 5f,
                    40
            );
        }

        // AI 开关圆钮。
        float knobCenterX = aiEnabled
                ? AI_TOGGLE_LEFT + AI_TOGGLE_WIDTH - 21f
                : AI_TOGGLE_LEFT + 21f;
        float knobCenterY = AI_TOGGLE_TOP + AI_TOGGLE_HEIGHT / 2f;
        shapes.setColor(CARD_SHADOW);
        shapes.circle(knobCenterX, toRenderY(knobCenterY + 3f), 15f, 28);
        shapes.setColor(SCORE_CARD);
        shapes.circle(knobCenterX, toRenderY(knobCenterY), 15f, 28);
        Color faceColor = aiEnabled ? ACCENT_DARK : TEXT_MUTED;
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

    private void drawEffects() {
        Gdx.gl.glEnable(GL20.GL_BLEND);
        for (Particle particle : particles) {
            float alpha = MathUtils.clamp(particle.life / 0.7f, 0f, 1f);
            shapes.setColor(
                    particle.color.r,
                    particle.color.g,
                    particle.color.b,
                    alpha
            );
            shapes.circle(particle.x, toRenderY(particle.y), 3.2f, 10);
        }
        for (Ring ring : rings) {
            float alpha = MathUtils.clamp(ring.life / 0.55f, 0f, 1f) * 0.22f;
            shapes.setColor(ring.color.r, ring.color.g, ring.color.b, alpha);
            shapes.circle(ring.x, toRenderY(ring.y), ring.radius, 36);
        }
    }

    private void drawFruitBodies() {
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

    private void drawText() {
        titleFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                titleFont,
                "MERGE MELON",
                28f,
                16f,
                330f,
                42f,
                Align.left
        );

        smallFont.setColor(DANGER);
        drawTextInBox(smallFont, "SCORE", 30f, 68f, 108f, 20f, Align.center);
        drawTextInBox(smallFont, "BEST", 152f, 68f, 108f, 20f, Align.center);
        drawTextInBox(smallFont, "NEXT", 284f, 68f, 52f, 20f, Align.left);

        normalFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                normalFont,
                Integer.toString(score),
                30f,
                88f,
                108f,
                34f,
                Align.center
        );
        drawTextInBox(
                normalFont,
                Integer.toString(bestScore),
                152f,
                88f,
                108f,
                34f,
                Align.center
        );

        smallFont.setColor(aiEnabled ? ACCENT_DARK : TEXT_MUTED);
        String aiLabel = aiEnabled ? "AI " + aiState.label : "AI OFF";
        float aiTextLeft = aiEnabled
                ? AI_TOGGLE_LEFT + 12f
                : AI_TOGGLE_LEFT + 43f;
        float aiTextWidth = aiEnabled ? 98f : 96f;
        drawTextInBox(
                smallFont,
                fitText(smallFont, aiLabel, aiTextWidth),
                aiTextLeft,
                AI_TOGGLE_TOP,
                aiTextWidth,
                AI_TOGGLE_HEIGHT,
                Align.center
        );

        if (!aiEnabled && alive) {
            smallFont.setColor(MANUAL_HINT);
            drawTextInBox(
                    smallFont,
                    "DRAG  -  RELEASE TO DROP",
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
                "GAME OVER",
                105f,
                420f,
                350f,
                54f,
                Align.center
        );
        normalFont.setColor(TEXT_MUTED);
        drawTextInBox(
                normalFont,
                "SCORE  " + score,
                105f,
                485f,
                350f,
                46f,
                Align.center
        );
        normalFont.setColor(TEXT_PRIMARY);
        drawTextInBox(
                normalFont,
                "TAP TO RESTART",
                145f,
                585f,
                270f,
                58f,
                Align.center
        );
        batch.end();
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
            return "unknown";
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
        cancelPendingDecision(enabled ? "AI enabled" : "manual control");
        aiState = enabled ? AiState.OBSERVING : AiState.MANUAL;
        aiDetail = enabled ? "AI enabled" : "drag and release";
        activeDragPointer = -1;
        stableSeconds = 0f;
        if (enabled) {
            aiLoadingSeconds = 0f;
        }
        aiService.vibrate(12);
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
        if (!alive) {
            resetGame();
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
            setAiEnabled(!aiEnabled);
            return true;
        }
        if (!aiEnabled && waiting && touchPoint.y >= 120f) {
            activeDragPointer = pointer;
            previewX = FruitRules.clampDropX(touchPoint.x, currentLevel);
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
        if (!aiEnabled && alive && waiting) {
            previewX = FruitRules.clampDropX(touchPoint.x, currentLevel);
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
        if (!aiEnabled && alive && waiting && touchPoint.y >= 120f) {
            previewX = FruitRules.clampDropX(touchPoint.x, currentLevel);
            dropCurrent(previewX);
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
        if (keycode == Input.Keys.R) {
            resetGame();
            return true;
        }
        if (keycode == Input.Keys.SPACE || keycode == Input.Keys.ENTER) {
            if (!aiEnabled) {
                dropCurrent(previewX);
            }
            return true;
        }
        if (keycode == Input.Keys.A || keycode == Input.Keys.LEFT) {
            if (!aiEnabled && waiting) {
                previewX = FruitRules.clampDropX(previewX - 14f, currentLevel);
            }
            return true;
        }
        if (keycode == Input.Keys.D || keycode == Input.Keys.RIGHT) {
            if (!aiEnabled && waiting) {
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

    private enum AiState {
        MANUAL("MANUAL"),
        OBSERVING("OBSERVE"),
        THINKING("THINK"),
        TESTING("TEST"),
        COMMITTING("COMMIT"),
        GAME_OVER("FINISHED");

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
        private float jitterRefresh;

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
                                alternative + signedRandom(random, 4f),
                                level
                        ),
                        0.16f + uncertainty * 0.18f,
                        0.07f + uncertainty * 0.13f,
                        AiState.TESTING,
                        "checking another lane"
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
                                    * (3f + random.nextFloat() * 8f),
                            level
                    ),
                    0.14f + random.nextFloat() * 0.16f,
                    0.05f + random.nextFloat() * 0.08f,
                    AiState.TESTING,
                    "fine adjustment"
            ));
            segments.add(new MotionSegment(
                    selected,
                    0.10f + random.nextFloat() * 0.13f,
                    0.07f + uncertainty * 0.10f,
                    AiState.COMMITTING,
                    "decision made"
            ));
            return new MotionPlan(decision, segments, currentX, random);
        }

        private MotionSample update(float delta) {
            if (holdTime > 0f) {
                MotionSegment heldSegment = segments.get(
                        Math.max(0, segmentIndex - 1)
                );
                holdTime = Math.max(0f, holdTime - delta);
                refreshJitter(
                        delta,
                        heldSegment.state == AiState.COMMITTING ? 0.8f : 2.2f
                );
                return new MotionSample(
                        currentX + jitter,
                        heldSegment.state,
                        holdTime <= 0f && segmentIndex >= segments.size
                                ? "drop"
                                : heldSegment.detail,
                        holdTime <= 0f && segmentIndex >= segments.size
                );
            }

            if (segmentIndex >= segments.size) {
                return new MotionSample(
                        currentX,
                        AiState.COMMITTING,
                        "drop",
                        true
                );
            }

            MotionSegment segment = segments.get(segmentIndex);
            segmentTime += delta;
            float progress = MathUtils.clamp(segmentTime / segment.duration, 0f, 1f);
            // 分段线性移动叠加离散颤动，刻意避免“机器人式”的完美平滑曲线。
            float baseX = startX + (segment.targetX - startX) * progress;
            refreshJitter(delta, segment.state == AiState.COMMITTING ? 0.65f : 1.8f);
            currentX = baseX;
            if (progress < 1f) {
                return new MotionSample(
                        currentX + jitter,
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
                    currentX + jitter,
                    segment.state,
                    segment.detail,
                    segmentIndex >= segments.size && holdTime <= 0f
            );
        }

        private void refreshJitter(float delta, float amplitude) {
            jitterRefresh -= delta;
            if (jitterRefresh <= 0f) {
                jitter = signedRandom(random, amplitude);
                jitterRefresh = 0.026f + random.nextFloat() * 0.055f;
            }
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

    private static final class Particle {
        private float x;
        private float y;
        private final float vx;
        private float vy;
        private float life;
        private final Color color;

        private Particle(
                float x,
                float y,
                float vx,
                float vy,
                float life,
                Color color) {
            this.x = x;
            this.y = y;
            this.vx = vx;
            this.vy = vy;
            this.life = life;
            this.color = new Color(color);
        }
    }

    private static final class Ring {
        private final float x;
        private final float y;
        private float radius;
        private final float speed;
        private float life;
        private final Color color;

        private Ring(
                float x,
                float y,
                float radius,
                float speed,
                float life,
                Color color) {
            this.x = x;
            this.y = y;
            this.radius = radius;
            this.speed = speed;
            this.life = life;
            this.color = new Color(color);
        }
    }
}
