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
        BodyDef definition = new BodyDef();
        definition.type = BodyDef.BodyType.DynamicBody;
        definition.position.set(
                x / PIXELS_PER_METER,
                y / PIXELS_PER_METER
        );
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
                nextFruitId++,
                level,
                FruitRules.displayRadius(level),
                physicsRadius,
                body
        );
        body.setUserData(fruit);
        fruits.add(fruit);
        return fruit;
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
