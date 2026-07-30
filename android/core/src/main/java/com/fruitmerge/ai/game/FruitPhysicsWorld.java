package com.fruitmerge.ai.game;

import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.physics.box2d.Body;
import com.badlogic.gdx.physics.box2d.BodyDef;
import com.badlogic.gdx.physics.box2d.Box2D;
import com.badlogic.gdx.physics.box2d.CircleShape;
import com.badlogic.gdx.physics.box2d.Contact;
import com.badlogic.gdx.physics.box2d.ContactImpulse;
import com.badlogic.gdx.physics.box2d.ContactListener;
import com.badlogic.gdx.physics.box2d.Fixture;
import com.badlogic.gdx.physics.box2d.FixtureDef;
import com.badlogic.gdx.physics.box2d.Manifold;
import com.badlogic.gdx.physics.box2d.PolygonShape;
import com.badlogic.gdx.physics.box2d.World;
import com.badlogic.gdx.utils.Array;

import java.util.HashSet;
import java.util.Set;

/**
 * 移动端圆形水果物理世界。
 *
 * <p>首个可安装版本使用 libGDX Box2D，以便在纯 Windows Gradle 工具链中完成编译
 * 和设备验证。参数逐项映射桌面 Pymunk：重力 1800 px/s²、果实摩擦 0.88、弹性
 * 0.18、20 px 边界以及同级 post-solve 合成。所有公开位置仍使用 560×1120 的像素
 * 坐标，Box2D 内部按 50 px/m 缩放，避免求解器处理过大的世界单位。</p>
 */
public final class FruitPhysicsWorld implements ContactListener {
    private static final float PIXELS_PER_METER = 50f;
    private static final float FIXED_STEP = 1f / 120f;
    private static final int VELOCITY_ITERATIONS = 16;
    private static final int POSITION_ITERATIONS = 8;
    private static final int MAX_SNAPSHOT_FRUITS = 4096;
    private static final float RADIUS_EPSILON = 0.001f;

    private final World world;
    private final Array<FruitBody> fruits = new Array<>();
    private final Array<MergePair> pendingMerges = new Array<>();
    private final Array<MergeEvent> mergeEvents = new Array<>();
    private float accumulator;
    private int nextFruitId = 1;

    public FruitPhysicsWorld() {
        // gdx-box2d 的 JNI 库不会由 AndroidApplication 自动初始化；World 构造前
        // 显式加载，避免首次进入游戏时出现 native method not found。
        Box2D.init();
        // 本项目采用屏幕坐标（y 向下），因此 Box2D 重力也设为正 y。
        world = new World(
                new Vector2(
                        0f,
                        FruitRules.GRAVITY_PIXELS_PER_SECOND_SQUARED
                                / PIXELS_PER_METER),
                true
        );
        world.setContactListener(this);
        createBoundaries();
    }

    public FruitBody addDroppedFruit(int level, float x, float y) {
        FruitBody fruit = createFruit(
                level,
                FruitRules.clampDropX(x, level),
                y,
                FruitRules.droppedPhysicsRadius(level)
        );
        fruit.body.setLinearVelocity(0f, 80f / PIXELS_PER_METER);
        return fruit;
    }

    public Array<FruitBody> fruits() {
        return fruits;
    }

    public Array<MergeEvent> drainMergeEvents() {
        Array<MergeEvent> drained = new Array<>(mergeEvents);
        mergeEvents.clear();
        return drained;
    }

    /**
     * 导出当前物理世界的可持久化快照。
     *
     * <p>水果按 {@link #fruits()} 的原始顺序写入，除了画面会读取的等级、位置和
     * 速度，也保留 ID、下一 ID、显示/碰撞半径、角度和 ageFrames。后两种半径不能
     * 从等级简单反推：直接投放和由合成产生的同级水果在部分等级使用不同碰撞半径，
     * 丢失这个区别会让读档后的堆叠形状发生跳变。</p>
     *
     * <p>pending merge 和尚未 drain 的 MergeEvent 都不属于物理状态快照。调用方
     * 应把已结算分数单独保存；恢复时不重放旧事件，避免一次合成重复计分。</p>
     */
    public Snapshot snapshot() {
        FruitState[] states = new FruitState[fruits.size];
        for (int index = 0; index < fruits.size; index++) {
            FruitBody fruit = fruits.get(index);
            states[index] = new FruitState(
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
                    fruit.ageFrames
            );
        }
        return new Snapshot(nextFruitId, accumulator, states);
    }

    /**
     * 用之前导出的状态替换当前物理世界。
     *
     * <p>校验发生在销毁当前刚体之前，因此空字段、重复 ID、非有限数值、越界等级
     * 或伪造半径等损坏输入只会抛出 {@link IllegalArgumentException}，不会让当前
     * 对局先被清空。校验通过后按快照顺序重建 Body，并清除 pending merge 与旧
     * MergeEvent；恢复后的第一次 step 只会产生当下真实接触引起的新事件。</p>
     */
    public void restore(Snapshot snapshot) {
        validateSnapshot(snapshot);

        clear();
        for (FruitState state : snapshot.fruits) {
            FruitBody fruit = createFruit(
                    state.id,
                    state.level,
                    state.displayRadius,
                    state.physicsRadius,
                    state.x,
                    state.y,
                    state.angle,
                    state.vx,
                    state.vy,
                    state.angularVelocity,
                    state.ageFrames
            );
            // createFruit 已按数组遍历顺序追加，因此 fruits() 的稳定顺序也随之恢复。
            fruit.pendingMerge = false;
        }
        nextFruitId = snapshot.nextFruitId;
        accumulator = snapshot.accumulatorSeconds;
        pendingMerges.clear();
        mergeEvents.clear();
    }

    public boolean isStable() {
        float maxVelocitySquared =
                FruitRules.STABLE_VELOCITY_PIXELS_PER_SECOND
                        * FruitRules.STABLE_VELOCITY_PIXELS_PER_SECOND;
        for (FruitBody fruit : fruits) {
            Vector2 velocity = fruit.body.getLinearVelocity();
            float vx = velocity.x * PIXELS_PER_METER;
            float vy = velocity.y * PIXELS_PER_METER;
            if (vx * vx + vy * vy > maxVelocitySquared) {
                return false;
            }
            if (Math.abs(fruit.body.getAngularVelocity())
                    > FruitRules.STABLE_ANGULAR_VELOCITY) {
                return false;
            }
        }
        return pendingMerges.size == 0;
    }

    public void step(float realDeltaSeconds) {
        // 限制一次恢复到前台后的超大 delta，避免物理世界“追帧”导致水果穿墙。
        accumulator += Math.min(realDeltaSeconds, 0.1f);
        while (accumulator >= FIXED_STEP) {
            world.step(FIXED_STEP, VELOCITY_ITERATIONS, POSITION_ITERATIONS);
            processPendingMerges();
            accumulator -= FIXED_STEP;
            for (FruitBody fruit : fruits) {
                fruit.ageFrames += 1;
            }
        }
    }

    public void clear() {
        pendingMerges.clear();
        mergeEvents.clear();
        for (FruitBody fruit : fruits) {
            if (fruit.active) {
                world.destroyBody(fruit.body);
                fruit.active = false;
            }
        }
        fruits.clear();
        accumulator = 0f;
        nextFruitId = 1;
    }

    public void dispose() {
        world.dispose();
    }

    @Override
    public void beginContact(Contact contact) {
        // 合成使用 postSolve；beginContact 不修改求解中的世界。
    }

    @Override
    public void endContact(Contact contact) {
        // 当前游戏不需要分离事件。
    }

    @Override
    public void preSolve(Contact contact, Manifold oldManifold) {
        // 使用 Box2D 默认接触解算。
    }

    @Override
    public void postSolve(Contact contact, ContactImpulse impulse) {
        Object firstData = contact.getFixtureA().getBody().getUserData();
        Object secondData = contact.getFixtureB().getBody().getUserData();
        if (!(firstData instanceof FruitBody) || !(secondData instanceof FruitBody)) {
            return;
        }
        FruitBody first = (FruitBody) firstData;
        FruitBody second = (FruitBody) secondData;
        if (!first.active || !second.active
                || first.pendingMerge || second.pendingMerge
                || first.level != second.level
                || first.level >= FruitRules.MAX_LEVEL) {
            return;
        }

        // 世界仍处于 locked 状态，只记录配对；真正删除和创建刚体在 step 返回后执行。
        first.pendingMerge = true;
        second.pendingMerge = true;
        pendingMerges.add(new MergePair(first, second));
    }

    private void createBoundaries() {
        createStaticBox(
                0f,
                FruitRules.BOARD_HEIGHT / 2f,
                FruitRules.WALL_WIDTH,
                FruitRules.BOARD_HEIGHT / 2f
        );
        createStaticBox(
                FruitRules.BOARD_WIDTH,
                FruitRules.BOARD_HEIGHT / 2f,
                FruitRules.WALL_WIDTH,
                FruitRules.BOARD_HEIGHT / 2f
        );
        createStaticBox(
                FruitRules.BOARD_WIDTH / 2f,
                FruitRules.BOARD_HEIGHT,
                FruitRules.BOARD_WIDTH / 2f,
                FruitRules.WALL_WIDTH
        );
    }

    private void createStaticBox(
            float centerX,
            float centerY,
            float halfWidth,
            float halfHeight) {
        BodyDef definition = new BodyDef();
        definition.type = BodyDef.BodyType.StaticBody;
        definition.position.set(
                centerX / PIXELS_PER_METER,
                centerY / PIXELS_PER_METER
        );
        Body body = world.createBody(definition);
        PolygonShape shape = new PolygonShape();
        shape.setAsBox(
                halfWidth / PIXELS_PER_METER,
                halfHeight / PIXELS_PER_METER
        );
        Fixture fixture = body.createFixture(shape, 0f);
        fixture.setFriction(0.6f);
        shape.dispose();
    }

    private FruitBody createFruit(
            int level,
            float x,
            float y,
            float physicsRadius) {
        return createFruit(
                nextFruitId++,
                level,
                FruitRules.displayRadius(level),
                physicsRadius,
                x,
                y,
                0f,
                0f,
                0f,
                0f,
                0
        );
    }

    private FruitBody createFruit(
            int id,
            int level,
            float displayRadius,
            float physicsRadius,
            float x,
            float y,
            float angle,
            float vx,
            float vy,
            float angularVelocity,
            int ageFrames) {
        BodyDef definition = new BodyDef();
        definition.type = BodyDef.BodyType.DynamicBody;
        definition.position.set(
                x / PIXELS_PER_METER,
                y / PIXELS_PER_METER
        );
        definition.angle = angle;
        definition.linearVelocity.set(
                vx / PIXELS_PER_METER,
                vy / PIXELS_PER_METER
        );
        definition.angularVelocity = angularVelocity;
        definition.linearDamping = 0.005f;
        definition.angularDamping = 0.005f;

        Body body = world.createBody(definition);
        CircleShape circle = new CircleShape();
        float radiusMeters = physicsRadius / PIXELS_PER_METER;
        circle.setRadius(radiusMeters);

        FixtureDef fixture = new FixtureDef();
        fixture.shape = circle;
        fixture.density = FruitRules.mass(level)
                / ((float) Math.PI * radiusMeters * radiusMeters);
        fixture.friction = 0.88f;
        fixture.restitution = 0.18f;
        body.createFixture(fixture);
        circle.dispose();

        FruitBody fruit = new FruitBody(
                id,
                level,
                displayRadius,
                physicsRadius,
                body
        );
        fruit.ageFrames = ageFrames;
        body.setUserData(fruit);
        fruits.add(fruit);
        return fruit;
    }

    private static void validateSnapshot(Snapshot snapshot) {
        if (snapshot == null) {
            throw new IllegalArgumentException("physics snapshot must not be null");
        }
        if (snapshot.fruits == null) {
            throw new IllegalArgumentException(
                    "physics snapshot fruits must not be null"
            );
        }
        if (snapshot.fruits.length > MAX_SNAPSHOT_FRUITS) {
            throw new IllegalArgumentException(
                    "physics snapshot contains too many fruits"
            );
        }
        if (snapshot.nextFruitId <= 0
                || snapshot.nextFruitId == Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "physics snapshot next fruit id is invalid"
            );
        }
        if (!Float.isFinite(snapshot.accumulatorSeconds)
                || snapshot.accumulatorSeconds < 0f
                || snapshot.accumulatorSeconds >= FIXED_STEP) {
            throw new IllegalArgumentException(
                    "physics snapshot accumulator is invalid"
            );
        }

        Set<Integer> ids = new HashSet<>();
        int greatestId = 0;
        for (FruitState state : snapshot.fruits) {
            validateFruitState(state);
            if (!ids.add(state.id)) {
                throw new IllegalArgumentException(
                        "physics snapshot contains duplicate fruit ids"
                );
            }
            greatestId = Math.max(greatestId, state.id);
        }
        if (snapshot.nextFruitId <= greatestId) {
            throw new IllegalArgumentException(
                    "physics snapshot next fruit id must exceed all existing ids"
            );
        }
    }

    private static void validateFruitState(FruitState state) {
        if (state == null) {
            throw new IllegalArgumentException(
                    "physics snapshot contains a null fruit"
            );
        }
        if (state.id <= 0) {
            throw new IllegalArgumentException(
                    "physics snapshot fruit id must be positive"
            );
        }
        if (state.level < FruitRules.MIN_LEVEL
                || state.level > FruitRules.MAX_LEVEL) {
            throw new IllegalArgumentException(
                    "physics snapshot fruit level is invalid"
            );
        }

        float expectedDisplayRadius = FruitRules.displayRadius(state.level);
        if (!approximatelyEqual(state.displayRadius, expectedDisplayRadius)) {
            throw new IllegalArgumentException(
                    "physics snapshot fruit display radius is invalid"
            );
        }
        float droppedRadius = FruitRules.droppedPhysicsRadius(state.level);
        float mergedRadius = FruitRules.mergedPhysicsRadius(state.level);
        boolean droppedRadiusMatches =
                approximatelyEqual(state.physicsRadius, droppedRadius);
        boolean mergedRadiusMatches = state.level > FruitRules.MIN_LEVEL
                && approximatelyEqual(state.physicsRadius, mergedRadius);
        if (!droppedRadiusMatches && !mergedRadiusMatches) {
            throw new IllegalArgumentException(
                    "physics snapshot fruit physics radius is invalid"
            );
        }

        if (!Float.isFinite(state.x)
                || !Float.isFinite(state.y)
                || !Float.isFinite(state.vx)
                || !Float.isFinite(state.vy)
                || !Float.isFinite(state.angle)
                || !Float.isFinite(state.angularVelocity)) {
            throw new IllegalArgumentException(
                    "physics snapshot fruit motion contains a non-finite value"
            );
        }
        if (state.ageFrames < 0) {
            throw new IllegalArgumentException(
                    "physics snapshot fruit age must not be negative"
            );
        }
    }

    private static boolean approximatelyEqual(float first, float second) {
        return Float.isFinite(first)
                && Math.abs(first - second) <= RADIUS_EPSILON;
    }

    private void processPendingMerges() {
        if (pendingMerges.size == 0) {
            return;
        }
        Array<MergePair> work = new Array<>(pendingMerges);
        pendingMerges.clear();
        for (MergePair pair : work) {
            FruitBody first = pair.first;
            FruitBody second = pair.second;
            if (!first.active || !second.active || first.level != second.level) {
                first.pendingMerge = false;
                second.pendingMerge = false;
                continue;
            }

            Vector2 firstPosition = first.body.getPosition();
            Vector2 secondPosition = second.body.getPosition();
            Vector2 lowerPosition = firstPosition.y > secondPosition.y
                    ? firstPosition : secondPosition;
            // Body#getPosition 返回的是刚体持有的内部向量。destroyBody 后不能再依赖
            // 该引用，所以先复制成普通数值，再销毁两颗源水果。
            float mergedX = lowerPosition.x * PIXELS_PER_METER;
            float mergedY = lowerPosition.y * PIXELS_PER_METER;
            int resultingLevel = first.level + 1;
            removeFruit(first);
            removeFruit(second);

            FruitBody result = createFruit(
                    resultingLevel,
                    mergedX,
                    mergedY,
                    FruitRules.mergedPhysicsRadius(resultingLevel)
            );
            mergeEvents.add(new MergeEvent(
                    result.id,
                    first.id,
                    second.id,
                    resultingLevel,
                    result.x(),
                    result.y(),
                    FruitRules.mergeScore(resultingLevel)
            ));
        }
    }

    private void removeFruit(FruitBody fruit) {
        if (!fruit.active) {
            return;
        }
        fruit.active = false;
        fruit.pendingMerge = false;
        fruits.removeValue(fruit, true);
        world.destroyBody(fruit.body);
    }

    /**
     * 一个物理世界在完整 fixed-step 之间的持久化数据。
     *
     * <p>构造器保持公开，存储层可以显式编码/解码而不依赖反射。数组在输入和输出
     * 两端都会复制，防止存档写入线程或调用方在 restore 校验后修改元素顺序。</p>
     */
    public static final class Snapshot {
        private final int nextFruitId;
        private final float accumulatorSeconds;
        private final FruitState[] fruits;

        public Snapshot(
                int nextFruitId,
                float accumulatorSeconds,
                FruitState[] fruits) {
            if (fruits == null) {
                throw new IllegalArgumentException(
                        "physics snapshot fruits must not be null"
                );
            }
            this.nextFruitId = nextFruitId;
            this.accumulatorSeconds = accumulatorSeconds;
            this.fruits = fruits.clone();
        }

        public int nextFruitId() {
            return nextFruitId;
        }

        public float accumulatorSeconds() {
            return accumulatorSeconds;
        }

        public int fruitCount() {
            return fruits.length;
        }

        public FruitState fruit(int index) {
            return fruits[index];
        }

        public FruitState[] fruits() {
            return fruits.clone();
        }
    }

    /**
     * 单颗水果的 Box2D 可恢复状态，线速度仍以像素/秒表示，角速度为弧度/秒。
     */
    public static final class FruitState {
        private final int id;
        private final int level;
        private final float displayRadius;
        private final float physicsRadius;
        private final float x;
        private final float y;
        private final float vx;
        private final float vy;
        private final float angle;
        private final float angularVelocity;
        private final int ageFrames;

        public FruitState(
                int id,
                int level,
                float displayRadius,
                float physicsRadius,
                float x,
                float y,
                float vx,
                float vy,
                float angle,
                float angularVelocity,
                int ageFrames) {
            this.id = id;
            this.level = level;
            this.displayRadius = displayRadius;
            this.physicsRadius = physicsRadius;
            this.x = x;
            this.y = y;
            this.vx = vx;
            this.vy = vy;
            this.angle = angle;
            this.angularVelocity = angularVelocity;
            this.ageFrames = ageFrames;
        }

        public int id() {
            return id;
        }

        public int level() {
            return level;
        }

        public float displayRadius() {
            return displayRadius;
        }

        public float physicsRadius() {
            return physicsRadius;
        }

        public float x() {
            return x;
        }

        public float y() {
            return y;
        }

        public float vx() {
            return vx;
        }

        public float vy() {
            return vy;
        }

        public float angle() {
            return angle;
        }

        public float angularVelocity() {
            return angularVelocity;
        }

        public int ageFrames() {
            return ageFrames;
        }
    }

    public static final class FruitBody {
        public final int id;
        public final int level;
        public final float displayRadius;
        public final float physicsRadius;
        private final Body body;
        private boolean active = true;
        private boolean pendingMerge;
        private int ageFrames;

        private FruitBody(
                int id,
                int level,
                float displayRadius,
                float physicsRadius,
                Body body) {
            this.id = id;
            this.level = level;
            this.displayRadius = displayRadius;
            this.physicsRadius = physicsRadius;
            this.body = body;
        }

        public float x() {
            return body.getPosition().x * PIXELS_PER_METER;
        }

        public float y() {
            return body.getPosition().y * PIXELS_PER_METER;
        }

        public float vx() {
            return body.getLinearVelocity().x * PIXELS_PER_METER;
        }

        public float vy() {
            return body.getLinearVelocity().y * PIXELS_PER_METER;
        }

        public float angle() {
            return body.getAngle();
        }

        public float angularVelocity() {
            return body.getAngularVelocity();
        }

        public int ageFrames() {
            return ageFrames;
        }

        public boolean isStable() {
            float speedSquared = vx() * vx() + vy() * vy();
            return speedSquared <= FruitRules.STABLE_VELOCITY_PIXELS_PER_SECOND
                    * FruitRules.STABLE_VELOCITY_PIXELS_PER_SECOND
                    && Math.abs(angularVelocity())
                    <= FruitRules.STABLE_ANGULAR_VELOCITY;
        }
    }

    public static final class MergeEvent {
        public final int fruitId;
        public final int sourceFruitIdA;
        public final int sourceFruitIdB;
        public final int level;
        public final float x;
        public final float y;
        public final int scoreDelta;

        private MergeEvent(
                int fruitId,
                int sourceFruitIdA,
                int sourceFruitIdB,
                int level,
                float x,
                float y,
                int scoreDelta) {
            this.fruitId = fruitId;
            this.sourceFruitIdA = sourceFruitIdA;
            this.sourceFruitIdB = sourceFruitIdB;
            this.level = level;
            this.x = x;
            this.y = y;
            this.scoreDelta = scoreDelta;
        }
    }

    private static final class MergePair {
        private final FruitBody first;
        private final FruitBody second;

        private MergePair(FruitBody first, FruitBody second) {
            this.first = first;
            this.second = second;
        }
    }
}
