package com.fruitmerge.ai.game;

import com.badlogic.gdx.utils.Array;

import java.util.HashSet;
import java.util.Set;

/**
 * 训练 Tensor/CUDA 求解器的单环境 Java 实现。
 *
 * <p>保持固定 64 槽、120 FPS、4 次顺序冲量迭代、低速回正和确定性合成配对，
 * 让 Android 游戏的权威状态与 SAB-T120 的训练物理处于同一算法域。画面直接读取
 * 这些圆形状态，不再让 Box2D 的接触求解结果反向进入模型。</p>
 */
public final class FruitPhysicsWorld {
    private static final int MAX_FRUITS = 64;
    private static final float FIXED_STEP = 1f / FruitRules.PHYSICS_FPS;
    private static final float FRAME_DAMPING =
            (float) Math.pow(0.995, FIXED_STEP);
    private static final int SOLVER_ITERATIONS = 4;
    private static final float ELASTICITY = 0.18f;
    private static final float RESTITUTION_VELOCITY_THRESHOLD = 35f;
    private static final float FRUIT_FRICTION = 0.88f;
    private static final float WALL_FRICTION = 0.60f;
    private static final float CONTACT_SLOP = 0.05f;
    private static final float POSITION_CORRECTION = 0.75f;
    private static final float MERGE_TOLERANCE = 0.25f;
    private static final int KINEMATIC_REST_FRAMES = 4;
    private static final float KINEMATIC_REST_SPEED = 12f;
    private static final float RADIUS_EPSILON = 0.001f;

    private final FruitBody[] slots = new FruitBody[MAX_FRUITS];
    private final Array<FruitBody> fruits = new Array<>();
    private final Array<MergeEvent> mergeEvents = new Array<>();
    private float accumulator;
    private int nextFruitId = 1;
    private boolean mergeOccurredLastFrame;

    public FruitPhysicsWorld() {
    }

    public FruitBody addDroppedFruit(int level, float x, float y) {
        int slot = firstFreeSlot();
        if (slot < 0) {
            throw new IllegalStateException("fruit capacity 64 is exhausted");
        }
        // CUDA 在 perform_drop=true 时会把整场的 quiet-frame 缓存清零。
        // 新水果可能撞开旧支撑，旧水果必须随之唤醒。
        wakeAllBodies();
        FruitBody fruit = new FruitBody(
                slot,
                nextFruitId++,
                level,
                FruitRules.displayRadius(level),
                FruitRules.droppedPhysicsRadius(level),
                FruitRules.clampDropX(x, level),
                y,
                0f,
                FruitRules.DROP_INITIAL_VELOCITY_Y,
                0f,
                0f,
                0
        );
        slots[slot] = fruit;
        rebuildFruitView();
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

    public Snapshot snapshot() {
        FruitState[] states = new FruitState[fruits.size];
        for (int index = 0; index < fruits.size; index++) {
            FruitBody fruit = fruits.get(index);
            states[index] = new FruitState(
                    fruit.slot,
                    fruit.id,
                    fruit.level,
                    fruit.displayRadius,
                    fruit.physicsRadius,
                    fruit.x,
                    fruit.y,
                    fruit.vx,
                    fruit.vy,
                    fruit.angle,
                    fruit.angularVelocity,
                    fruit.ageFrames
            );
        }
        return new Snapshot(nextFruitId, accumulator, states);
    }

    public void restore(Snapshot snapshot) {
        validateSnapshot(snapshot);
        clear();
        for (int index = 0; index < snapshot.fruits.length; index++) {
            FruitState state = snapshot.fruits[index];
            int slot = state.slot >= 0 ? state.slot : index;
            slots[slot] = new FruitBody(
                    slot,
                    state.id,
                    state.level,
                    state.displayRadius,
                    state.physicsRadius,
                    state.x,
                    state.y,
                    state.vx,
                    state.vy,
                    state.angle,
                    state.angularVelocity,
                    state.ageFrames
            );
        }
        nextFruitId = snapshot.nextFruitId;
        accumulator = snapshot.accumulatorSeconds;
        mergeEvents.clear();
        mergeOccurredLastFrame = false;
        rebuildFruitView();
    }

    public boolean isStable() {
        if (mergeOccurredLastFrame) {
            return false;
        }
        float maximumSpeedSquared =
                FruitRules.STABLE_VELOCITY_PIXELS_PER_SECOND
                        * FruitRules.STABLE_VELOCITY_PIXELS_PER_SECOND;
        for (FruitBody fruit : fruits) {
            if (fruit.vx * fruit.vx + fruit.vy * fruit.vy
                    > maximumSpeedSquared
                    || Math.abs(fruit.angularVelocity)
                    > FruitRules.STABLE_ANGULAR_VELOCITY) {
                return false;
            }
        }
        return true;
    }

    /** 与训练失败计时一致：忽略最新水果，按圆心截断后的 y 判定。 */
    public boolean isOverDangerLine() {
        int newestId = 0;
        for (FruitBody fruit : fruits) {
            newestId = Math.max(newestId, fruit.id);
        }
        for (FruitBody fruit : fruits) {
            if (fruit.id != newestId && (int) fruit.y < FruitRules.SPAWN_Y) {
                return true;
            }
        }
        return false;
    }

    /** 模型观察字段使用所有水果顶部，不复用失败计时器的圆心规则。 */
    public boolean hasFruitTopAboveSpawnLine() {
        for (FruitBody fruit : fruits) {
            if (fruit.y - fruit.physicsRadius < FruitRules.SPAWN_Y) {
                return true;
            }
        }
        return false;
    }

    public void step(float realDeltaSeconds) {
        step(realDeltaSeconds, null);
    }

    /**
     * 推进固定 120 Hz 物理帧，并在每一帧结束后报告训练契约使用的稳定/危险状态。
     * 观察者返回 false 时立即停在该物理帧边界，同时丢弃墙钟累积量；这样异步推理
     * 和渲染卡顿都不会让权威状态越过模型应当观察的决策边界。
     */
    public void step(float realDeltaSeconds, FrameObserver observer) {
        // 上层已过滤后台恢复的异常 delta；这里允许低帧率画面补足实际经过的
        // 物理时间，不能在每帧静默丢弃 100 ms 以外的部分。
        accumulator += Math.min(realDeltaSeconds, 0.5f);
        while (accumulator >= FIXED_STEP) {
            advanceFrame();
            accumulator -= FIXED_STEP;
            if (observer != null
                    && !observer.afterFrame(isStable(), isOverDangerLine())) {
                accumulator = 0f;
                break;
            }
        }
    }

    @FunctionalInterface
    public interface FrameObserver {
        boolean afterFrame(boolean stable, boolean overDangerLine);
    }

    private void advanceFrame() {
        mergeOccurredLastFrame = false;
        float[] startX = new float[MAX_FRUITS];
        float[] startY = new float[MAX_FRUITS];
        for (int slot = 0; slot < MAX_FRUITS; slot++) {
            FruitBody fruit = slots[slot];
            if (fruit == null) {
                continue;
            }
            startX[slot] = fruit.x;
            startY[slot] = fruit.y;
            fruit.ageFrames += 1;
            fruit.vy += FruitRules.GRAVITY_PIXELS_PER_SECOND_SQUARED
                    * FIXED_STEP;
            fruit.vx *= FRAME_DAMPING;
            fruit.vy *= FRAME_DAMPING;
            fruit.angularVelocity *= FRAME_DAMPING;
            fruit.x += fruit.vx * FIXED_STEP;
            fruit.y += fruit.vy * FIXED_STEP;
            fruit.angle += fruit.angularVelocity * FIXED_STEP;
        }

        for (int iteration = 0; iteration < SOLVER_ITERATIONS; iteration++) {
            for (int slot = 0; slot < MAX_FRUITS; slot++) {
                resolveWalls(slots[slot]);
            }
            for (int first = 0; first < MAX_FRUITS; first++) {
                if (slots[first] == null) {
                    continue;
                }
                for (int second = first + 1; second < MAX_FRUITS; second++) {
                    resolvePair(slots[first], slots[second]);
                }
            }
        }

        for (int slot = 0; slot < MAX_FRUITS; slot++) {
            resolveWalls(slots[slot]);
        }
        resolveMerges();
        applyKinematicRest(startX, startY);
        rebuildFruitView();
    }

    private static void resolveWalls(FruitBody fruit) {
        if (fruit == null) {
            return;
        }
        applyWall(
                fruit,
                1f,
                0f,
                FruitRules.WALL_WIDTH + fruit.physicsRadius - fruit.x
        );
        applyWall(
                fruit,
                -1f,
                0f,
                fruit.x - (FruitRules.BOARD_WIDTH
                        - FruitRules.WALL_WIDTH - fruit.physicsRadius)
        );
        applyWall(
                fruit,
                0f,
                -1f,
                fruit.y - (FruitRules.BOARD_HEIGHT
                        - FruitRules.WALL_WIDTH - fruit.physicsRadius)
        );
    }

    private static void applyWall(
            FruitBody fruit,
            float normalX,
            float normalY,
            float penetration) {
        if (penetration <= 0f) {
            return;
        }
        fruit.x += normalX * penetration;
        fruit.y += normalY * penetration;
        float normalVelocity = dot(
                fruit.vx, fruit.vy, normalX, normalY);
        if (normalVelocity >= 0f) {
            return;
        }
        float restitution = -normalVelocity
                >= RESTITUTION_VELOCITY_THRESHOLD ? ELASTICITY : 0f;
        float normalImpulse = -(1f + restitution) * normalVelocity
                / fruit.inverseMass;
        float tangentX = -normalY;
        float tangentY = normalX;
        float radiusX = -normalX * fruit.physicsRadius;
        float radiusY = -normalY * fruit.physicsRadius;
        float contactVelocityX = fruit.vx
                - fruit.angularVelocity * radiusY;
        float contactVelocityY = fruit.vy
                + fruit.angularVelocity * radiusX;
        float tangentVelocity = dot(
                contactVelocityX, contactVelocityY, tangentX, tangentY);
        float crossRadiusTangent = cross(
                radiusX, radiusY, tangentX, tangentY);
        float tangentDenominator = fruit.inverseMass
                + crossRadiusTangent * crossRadiusTangent
                * fruit.inverseInertia;
        float tangentImpulse = -tangentVelocity
                / Math.max(tangentDenominator, 1e-12f);
        float frictionLimit = WALL_FRICTION * normalImpulse;
        tangentImpulse = clamp(
                tangentImpulse, -frictionLimit, frictionLimit);
        float impulseX = normalX * normalImpulse
                + tangentX * tangentImpulse;
        float impulseY = normalY * normalImpulse
                + tangentY * tangentImpulse;
        fruit.vx += impulseX * fruit.inverseMass;
        fruit.vy += impulseY * fruit.inverseMass;
        fruit.angularVelocity += cross(
                radiusX, radiusY, impulseX, impulseY
        ) * fruit.inverseInertia;
    }

    private static void resolvePair(FruitBody first, FruitBody second) {
        if (first == null || second == null) {
            return;
        }
        float deltaX = second.x - first.x;
        float deltaY = second.y - first.y;
        float radiusSum = first.physicsRadius + second.physicsRadius;
        if (Math.abs(deltaX) > radiusSum || Math.abs(deltaY) > radiusSum) {
            return;
        }
        float distanceSquared = dot(deltaX, deltaY, deltaX, deltaY);
        if (distanceSquared > radiusSum * radiusSum) {
            return;
        }
        float distance = (float) Math.sqrt(Math.max(distanceSquared, 1e-12f));
        float inverseDistance = 1f / distance;
        float normalX = distanceSquared < 1e-12f
                ? 1f : deltaX * inverseDistance;
        float normalY = distanceSquared < 1e-12f
                ? 0f : deltaY * inverseDistance;
        float inverseMassSum = Math.max(
                first.inverseMass + second.inverseMass, 1e-12f);
        float penetration = Math.max(
                radiusSum - distance - CONTACT_SLOP, 0f);
        float correction = POSITION_CORRECTION * penetration / inverseMassSum;
        float firstCorrection = correction * first.inverseMass;
        float secondCorrection = correction * second.inverseMass;
        first.x -= normalX * firstCorrection;
        first.y -= normalY * firstCorrection;
        second.x += normalX * secondCorrection;
        second.y += normalY * secondCorrection;

        float firstRadiusX = normalX * first.physicsRadius;
        float firstRadiusY = normalY * first.physicsRadius;
        float secondRadiusX = -normalX * second.physicsRadius;
        float secondRadiusY = -normalY * second.physicsRadius;
        float firstContactX = first.vx
                - first.angularVelocity * firstRadiusY;
        float firstContactY = first.vy
                + first.angularVelocity * firstRadiusX;
        float secondContactX = second.vx
                - second.angularVelocity * secondRadiusY;
        float secondContactY = second.vy
                + second.angularVelocity * secondRadiusX;
        float relativeX = secondContactX - firstContactX;
        float relativeY = secondContactY - firstContactY;
        float normalVelocity = dot(
                relativeX, relativeY, normalX, normalY);
        if (normalVelocity >= 0f) {
            return;
        }
        float restitution = -normalVelocity
                >= RESTITUTION_VELOCITY_THRESHOLD ? ELASTICITY : 0f;
        float normalImpulse = -(1f + restitution) * normalVelocity
                / inverseMassSum;
        float tangentX = -normalY;
        float tangentY = normalX;
        float tangentVelocity = dot(
                relativeX, relativeY, tangentX, tangentY);
        float firstCross = cross(
                firstRadiusX, firstRadiusY, tangentX, tangentY);
        float secondCross = cross(
                secondRadiusX, secondRadiusY, tangentX, tangentY);
        float tangentDenominator = inverseMassSum
                + firstCross * firstCross * first.inverseInertia
                + secondCross * secondCross * second.inverseInertia;
        float tangentImpulse = -tangentVelocity
                / Math.max(tangentDenominator, 1e-12f);
        float frictionLimit = FRUIT_FRICTION * normalImpulse;
        tangentImpulse = clamp(
                tangentImpulse, -frictionLimit, frictionLimit);
        float impulseX = normalX * normalImpulse
                + tangentX * tangentImpulse;
        float impulseY = normalY * normalImpulse
                + tangentY * tangentImpulse;
        first.vx -= impulseX * first.inverseMass;
        first.vy -= impulseY * first.inverseMass;
        second.vx += impulseX * second.inverseMass;
        second.vy += impulseY * second.inverseMass;
        first.angularVelocity -= cross(
                firstRadiusX, firstRadiusY, impulseX, impulseY
        ) * first.inverseInertia;
        second.angularVelocity += cross(
                secondRadiusX, secondRadiusY, impulseX, impulseY
        ) * second.inverseInertia;
    }

    private void resolveMerges() {
        boolean[] claimed = new boolean[MAX_FRUITS];
        for (int firstSlot = 0; firstSlot < MAX_FRUITS; firstSlot++) {
            FruitBody first = slots[firstSlot];
            if (first == null || claimed[firstSlot]) {
                continue;
            }
            for (int secondSlot = firstSlot + 1;
                 secondSlot < MAX_FRUITS;
                 secondSlot++) {
                FruitBody second = slots[secondSlot];
                if (second == null || claimed[secondSlot]
                        || !touchingForMerge(first, second)) {
                    continue;
                }
                claimed[firstSlot] = true;
                claimed[secondSlot] = true;
                int sourceLevel = first.level;
                int sourceIdA = first.id;
                int sourceIdB = second.id;
                float midpointX = (first.x + second.x) * 0.5f;
                float midpointY = (first.y + second.y) * 0.5f;
                int resultingLevel = sourceLevel < FruitRules.MAX_LEVEL
                        ? sourceLevel + 1 : 0;
                int resultId = resultingLevel > 0 ? nextFruitId++ : 0;
                slots[firstSlot] = resultingLevel > 0
                        ? new FruitBody(
                                firstSlot,
                                resultId,
                                resultingLevel,
                                FruitRules.displayRadius(resultingLevel),
                                FruitRules.mergedPhysicsRadius(resultingLevel),
                                midpointX,
                                midpointY,
                                0f,
                                0f,
                                0f,
                                0f,
                                0
                        )
                        : null;
                slots[secondSlot] = null;
                mergeEvents.add(new MergeEvent(
                        resultId,
                        sourceIdA,
                        sourceIdB,
                        sourceLevel,
                        resultingLevel,
                        midpointX,
                        midpointY,
                        FruitRules.mergeScoreForSource(sourceLevel)
                ));
                mergeOccurredLastFrame = true;
                break;
            }
        }
    }

    private static boolean touchingForMerge(
            FruitBody first,
            FruitBody second) {
        if (first.level != second.level) {
            return false;
        }
        float deltaX = second.x - first.x;
        float deltaY = second.y - first.y;
        float radiusSum = first.physicsRadius + second.physicsRadius
                + MERGE_TOLERANCE;
        return Math.abs(deltaX) <= radiusSum
                && Math.abs(deltaY) <= radiusSum
                && dot(deltaX, deltaY, deltaX, deltaY)
                <= radiusSum * radiusSum;
    }

    private void applyKinematicRest(float[] startX, float[] startY) {
        float restingDisplacement =
                FruitRules.STABLE_VELOCITY_PIXELS_PER_SECOND * FIXED_STEP;
        float initialDisplacement = KINEMATIC_REST_SPEED * FIXED_STEP;
        for (int slot = 0; slot < MAX_FRUITS; slot++) {
            FruitBody fruit = slots[slot];
            if (fruit == null || fruit.ageFrames == 0) {
                if (fruit != null) {
                    fruit.quietFrames = 0;
                }
                continue;
            }
            float deltaX = fruit.x - startX[slot];
            float deltaY = fruit.y - startY[slot];
            float epsilon = fruit.quietFrames >= KINEMATIC_REST_FRAMES
                    ? restingDisplacement : initialDisplacement;
            if (dot(deltaX, deltaY, deltaX, deltaY) <= epsilon * epsilon) {
                fruit.quietFrames = Math.min(255, fruit.quietFrames + 1);
            } else {
                fruit.quietFrames = 0;
            }
            if (fruit.quietFrames >= KINEMATIC_REST_FRAMES) {
                fruit.vx = 0f;
                fruit.vy = 0f;
                fruit.angularVelocity = 0f;
            }
        }
    }

    public void clear() {
        for (int slot = 0; slot < MAX_FRUITS; slot++) {
            slots[slot] = null;
        }
        fruits.clear();
        mergeEvents.clear();
        accumulator = 0f;
        nextFruitId = 1;
        mergeOccurredLastFrame = false;
    }

    public void dispose() {
        clear();
    }

    private void wakeAllBodies() {
        for (FruitBody fruit : slots) {
            if (fruit != null) {
                fruit.quietFrames = 0;
            }
        }
    }

    private int firstFreeSlot() {
        for (int slot = 0; slot < MAX_FRUITS; slot++) {
            if (slots[slot] == null) {
                return slot;
            }
        }
        return -1;
    }

    private void rebuildFruitView() {
        fruits.clear();
        for (FruitBody fruit : slots) {
            if (fruit != null) {
                fruits.add(fruit);
            }
        }
    }

    private static float cross(
            float firstX,
            float firstY,
            float secondX,
            float secondY) {
        return firstX * secondY - firstY * secondX;
    }

    private static float dot(
            float firstX,
            float firstY,
            float secondX,
            float secondY) {
        return firstX * secondX + firstY * secondY;
    }

    private static float clamp(float value, float minimum, float maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private static void validateSnapshot(Snapshot snapshot) {
        if (snapshot == null || snapshot.fruits == null) {
            throw new IllegalArgumentException("physics snapshot is invalid");
        }
        if (snapshot.fruits.length > MAX_FRUITS) {
            throw new IllegalArgumentException("physics snapshot contains too many fruits");
        }
        if (snapshot.nextFruitId <= 0
                || snapshot.nextFruitId == Integer.MAX_VALUE) {
            throw new IllegalArgumentException("physics snapshot next fruit id is invalid");
        }
        if (!Float.isFinite(snapshot.accumulatorSeconds)
                || snapshot.accumulatorSeconds < 0f
                || snapshot.accumulatorSeconds >= FIXED_STEP) {
            throw new IllegalArgumentException("physics snapshot accumulator is invalid");
        }
        Set<Integer> ids = new HashSet<>();
        Set<Integer> occupiedSlots = new HashSet<>();
        int maximumId = 0;
        for (int index = 0; index < snapshot.fruits.length; index++) {
            FruitState state = snapshot.fruits[index];
            int slot = state == null || state.slot < 0 ? index : state.slot;
            if (state == null
                    || slot < 0
                    || slot >= MAX_FRUITS
                    || !occupiedSlots.add(slot)
                    || state.id <= 0
                    || !ids.add(state.id)
                    || state.level < FruitRules.MIN_LEVEL
                    || state.level > FruitRules.MAX_LEVEL
                    || state.ageFrames < 0
                    || !allFinite(
                            state.displayRadius,
                            state.physicsRadius,
                            state.x,
                            state.y,
                            state.vx,
                            state.vy,
                            state.angle,
                            state.angularVelocity)
                    || Math.abs(state.displayRadius
                            - FruitRules.displayRadius(state.level))
                            > RADIUS_EPSILON
                    || !validPhysicsRadius(state)) {
                throw new IllegalArgumentException("physics snapshot fruit is invalid");
            }
            maximumId = Math.max(maximumId, state.id);
        }
        if (snapshot.nextFruitId <= maximumId) {
            throw new IllegalArgumentException("next fruit id must exceed all fruit ids");
        }
    }

    private static boolean validPhysicsRadius(FruitState state) {
        return Math.abs(state.physicsRadius
                - FruitRules.droppedPhysicsRadius(state.level)) <= RADIUS_EPSILON
                || Math.abs(state.physicsRadius
                - FruitRules.mergedPhysicsRadius(state.level)) <= RADIUS_EPSILON;
    }

    private static boolean allFinite(float... values) {
        for (float value : values) {
            if (!Float.isFinite(value)) {
                return false;
            }
        }
        return true;
    }

    public static final class Snapshot {
        private final int nextFruitId;
        private final float accumulatorSeconds;
        private final FruitState[] fruits;

        public Snapshot(
                int nextFruitId,
                float accumulatorSeconds,
                FruitState[] fruits) {
            if (fruits == null) {
                throw new IllegalArgumentException("fruits must not be null");
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

    public static final class FruitState {
        private final int slot;
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
            this(
                    -1,
                    id,
                    level,
                    displayRadius,
                    physicsRadius,
                    x,
                    y,
                    vx,
                    vy,
                    angle,
                    angularVelocity,
                    ageFrames
            );
        }

        public FruitState(
                int slot,
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
            this.slot = slot;
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

        public int slot() { return slot; }
        public int id() { return id; }
        public int level() { return level; }
        public float displayRadius() { return displayRadius; }
        public float physicsRadius() { return physicsRadius; }
        public float x() { return x; }
        public float y() { return y; }
        public float vx() { return vx; }
        public float vy() { return vy; }
        public float angle() { return angle; }
        public float angularVelocity() { return angularVelocity; }
        public int ageFrames() { return ageFrames; }
    }

    public static final class FruitBody {
        public final int slot;
        public final int id;
        public final int level;
        public final float displayRadius;
        public final float physicsRadius;
        private final float inverseMass;
        private final float inverseInertia;
        private float x;
        private float y;
        private float vx;
        private float vy;
        private float angle;
        private float angularVelocity;
        private int ageFrames;
        private int quietFrames;

        private FruitBody(
                int slot,
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
            this.slot = slot;
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
            float mass = FruitRules.mass(level);
            inverseMass = 1f / mass;
            inverseInertia = 1f
                    / (0.5f * mass * physicsRadius * physicsRadius);
        }

        public float x() { return x; }
        public float y() { return y; }
        public float vx() { return vx; }
        public float vy() { return vy; }
        public float angle() { return angle; }
        public float angularVelocity() { return angularVelocity; }
        public int ageFrames() { return ageFrames; }

        public boolean isStable() {
            return vx * vx + vy * vy
                    <= FruitRules.STABLE_VELOCITY_PIXELS_PER_SECOND
                    * FruitRules.STABLE_VELOCITY_PIXELS_PER_SECOND
                    && Math.abs(angularVelocity)
                    <= FruitRules.STABLE_ANGULAR_VELOCITY;
        }
    }

    public static final class MergeEvent {
        public final int fruitId;
        public final int sourceFruitIdA;
        public final int sourceFruitIdB;
        public final int sourceLevel;
        public final int level;
        public final float x;
        public final float y;
        public final int scoreDelta;

        private MergeEvent(
                int fruitId,
                int sourceFruitIdA,
                int sourceFruitIdB,
                int sourceLevel,
                int level,
                float x,
                float y,
                int scoreDelta) {
            this.fruitId = fruitId;
            this.sourceFruitIdA = sourceFruitIdA;
            this.sourceFruitIdB = sourceFruitIdB;
            this.sourceLevel = sourceLevel;
            this.level = level;
            this.x = x;
            this.y = y;
            this.scoreDelta = scoreDelta;
        }
    }
}
