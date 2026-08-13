package com.fruitmerge.ai.game;

import com.badlogic.gdx.utils.Array;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * FruitPhysicsWorld 存档契约的真实训练同算法物理测试。
 *
 * <p>覆盖的不是 DTO 自身复制，而是重建真实运行状态后的位置、速度、半径和
 * 后续确定性合成。</p>
 */
public final class FruitPhysicsWorldSnapshotTest {
    private static final float FLOAT_EPSILON = 0.0001f;

    @Test
    public void roundTripPreservesSparseModelSlots() {
        FruitPhysicsWorld.FruitState first = new FruitPhysicsWorld.FruitState(
                0,
                8,
                2,
                FruitRules.displayRadius(2),
                FruitRules.droppedPhysicsRadius(2),
                100f,
                800f,
                0f,
                0f,
                0f,
                0f,
                20
        );
        FruitPhysicsWorld.FruitState fifth = new FruitPhysicsWorld.FruitState(
                5,
                12,
                4,
                FruitRules.displayRadius(4),
                FruitRules.mergedPhysicsRadius(4),
                420f,
                900f,
                0f,
                0f,
                0f,
                0f,
                40
        );
        FruitPhysicsWorld world = new FruitPhysicsWorld();
        try {
            world.restore(new FruitPhysicsWorld.Snapshot(
                    13,
                    0f,
                    new FruitPhysicsWorld.FruitState[]{first, fifth}
            ));
            assertEquals(0, world.fruits().get(0).slot);
            assertEquals(5, world.fruits().get(1).slot);
            FruitPhysicsWorld.Snapshot roundTrip = world.snapshot();
            assertEquals(0, roundTrip.fruit(0).slot());
            assertEquals(5, roundTrip.fruit(1).slot());
        } finally {
            world.dispose();
        }
    }

    @Test
    public void roundTripPreservesOrderMotionIdsAndDistinctRadii() {
        FruitPhysicsWorld.FruitState dropped =
                new FruitPhysicsWorld.FruitState(
                        8,
                        3,
                        FruitRules.displayRadius(3),
                        FruitRules.droppedPhysicsRadius(3),
                        104.5f,
                        410.25f,
                        12.75f,
                        -34.5f,
                        0.72f,
                        1.35f,
                        17
                );
        FruitPhysicsWorld.FruitState merged =
                new FruitPhysicsWorld.FruitState(
                        12,
                        3,
                        FruitRules.displayRadius(3),
                        FruitRules.mergedPhysicsRadius(3),
                        438.75f,
                        690.5f,
                        -23.25f,
                        44.75f,
                        -0.48f,
                        -0.85f,
                        91
                );
        FruitPhysicsWorld.Snapshot seed =
                new FruitPhysicsWorld.Snapshot(
                        13,
                        0.004f,
                        new FruitPhysicsWorld.FruitState[]{dropped, merged}
                );

        FruitPhysicsWorld firstWorld = new FruitPhysicsWorld();
        FruitPhysicsWorld restoredWorld = new FruitPhysicsWorld();
        try {
            firstWorld.restore(seed);
            FruitPhysicsWorld.Snapshot liveSnapshot = firstWorld.snapshot();
            restoredWorld.restore(liveSnapshot);
            FruitPhysicsWorld.Snapshot roundTrip = restoredWorld.snapshot();

            assertEquals(13, roundTrip.nextFruitId());
            assertEquals(0.004f, roundTrip.accumulatorSeconds(), FLOAT_EPSILON);
            assertEquals(2, roundTrip.fruitCount());
            assertArrayEquals(
                    new int[]{8, 12},
                    ids(roundTrip)
            );
            assertStateEquals(dropped, roundTrip.fruit(0));
            assertStateEquals(merged, roundTrip.fruit(1));
            assertEquals(
                    FruitRules.droppedPhysicsRadius(3),
                    roundTrip.fruit(0).physicsRadius(),
                    FLOAT_EPSILON
            );
            assertEquals(
                    FruitRules.mergedPhysicsRadius(3),
                    roundTrip.fruit(1).physicsRadius(),
                    FLOAT_EPSILON
            );
            assertTrue(
                    roundTrip.fruit(0).physicsRadius()
                            != roundTrip.fruit(1).physicsRadius()
            );

            FruitPhysicsWorld.FruitBody next =
                    restoredWorld.addDroppedFruit(1, 280f, 252f);
            assertEquals(13, next.id);
        } finally {
            firstWorld.dispose();
            restoredWorld.dispose();
        }
    }

    @Test
    public void restoredBodiesContinueThroughRealPhysicsMerge() {
        FruitPhysicsWorld source = new FruitPhysicsWorld();
        FruitPhysicsWorld restored = new FruitPhysicsWorld();
        try {
            FruitPhysicsWorld.FruitBody first =
                    source.addDroppedFruit(2, 250f, 700f);
            FruitPhysicsWorld.FruitBody second =
                    source.addDroppedFruit(2, 305f, 700f);
            FruitPhysicsWorld.Snapshot beforeContact = source.snapshot();

            restored.restore(beforeContact);
            restored.step(0.02f);
            Array<FruitPhysicsWorld.MergeEvent> events =
                    restored.drainMergeEvents();

            assertEquals(1, events.size);
            FruitPhysicsWorld.MergeEvent event = events.first();
            assertEquals(3, event.fruitId);
            assertTrue(
                    event.sourceFruitIdA == first.id
                            && event.sourceFruitIdB == second.id
                            || event.sourceFruitIdA == second.id
                            && event.sourceFruitIdB == first.id
            );
            assertEquals(3, event.level);
            assertEquals(1, restored.fruits().size);
            assertEquals(3, restored.fruits().first().id);
            assertEquals(
                    FruitRules.mergedPhysicsRadius(3),
                    restored.fruits().first().physicsRadius,
                    FLOAT_EPSILON
            );
        } finally {
            source.dispose();
            restored.dispose();
        }
    }

    @Test
    public void residualSlowRotationIsNotKinematicallyCleared() {
        int level = 3;
        float radius = FruitRules.droppedPhysicsRadius(level);
        FruitPhysicsWorld world = new FruitPhysicsWorld();
        try {
            world.restore(new FruitPhysicsWorld.Snapshot(
                    2,
                    0f,
                    new FruitPhysicsWorld.FruitState[]{
                            new FruitPhysicsWorld.FruitState(
                                    1,
                                    level,
                                    FruitRules.displayRadius(level),
                                    radius,
                                    FruitRules.BOARD_WIDTH * 0.5f,
                                    FruitRules.FLOOR_Y - radius,
                                    0f,
                                    0f,
                                    0f,
                                    0.1f,
                                    30
                            )
                    }
            ));

            world.step(2f);
            FruitPhysicsWorld.FruitBody rested = world.fruits().first();
            float angleBeforeMoreFrames = rested.angle();
            assertTrue(Math.abs(rested.angularVelocity()) > FLOAT_EPSILON);

            world.step(1f);
            assertTrue(Math.abs(
                    world.fruits().first().angle() - angleBeforeMoreFrames
            ) > FLOAT_EPSILON);
        } finally {
            world.dispose();
        }
    }

    @Test
    public void restoreDropsOldEventsAndRejectsCorruptionBeforeClearing() {
        FruitPhysicsWorld world = new FruitPhysicsWorld();
        try {
            FruitPhysicsWorld.FruitBody original =
                    world.addDroppedFruit(1, 160f, 500f);
            FruitPhysicsWorld.FruitState valid =
                    state(5, 2, FruitRules.droppedPhysicsRadius(2));
            FruitPhysicsWorld.FruitState duplicate =
                    state(5, 3, FruitRules.droppedPhysicsRadius(3));
            FruitPhysicsWorld.Snapshot duplicateIds =
                    new FruitPhysicsWorld.Snapshot(
                            6,
                            0f,
                            new FruitPhysicsWorld.FruitState[]{valid, duplicate}
                    );

            assertThrows(
                    IllegalArgumentException.class,
                    () -> world.restore(duplicateIds)
            );
            assertEquals(1, world.fruits().size);
            assertEquals(original.id, world.fruits().first().id);

            assertRejected(
                    world,
                    new FruitPhysicsWorld.Snapshot(
                            5,
                            0f,
                            new FruitPhysicsWorld.FruitState[]{valid}
                    )
            );
            assertRejected(
                    world,
                    new FruitPhysicsWorld.Snapshot(
                            6,
                            1f / 120f,
                            new FruitPhysicsWorld.FruitState[]{valid}
                    )
            );
            assertRejected(
                    world,
                    new FruitPhysicsWorld.Snapshot(
                            6,
                            0f,
                            new FruitPhysicsWorld.FruitState[]{
                                    new FruitPhysicsWorld.FruitState(
                                            5,
                                            2,
                                            FruitRules.displayRadius(2),
                                            27f,
                                            200f,
                                            500f,
                                            0f,
                                            0f,
                                            0f,
                                            0f,
                                            0
                                    )
                            }
                    )
            );
            assertRejected(
                    world,
                    new FruitPhysicsWorld.Snapshot(
                            6,
                            0f,
                            new FruitPhysicsWorld.FruitState[]{
                                    new FruitPhysicsWorld.FruitState(
                                            5,
                                            2,
                                            FruitRules.displayRadius(2),
                                            FruitRules.droppedPhysicsRadius(2),
                                            Float.NaN,
                                            500f,
                                            0f,
                                            0f,
                                            0f,
                                            0f,
                                            0
                                    )
                            }
                    )
            );
            assertRejected(
                    world,
                    new FruitPhysicsWorld.Snapshot(
                            6,
                            0f,
                            new FruitPhysicsWorld.FruitState[]{
                                    new FruitPhysicsWorld.FruitState(
                                            5,
                                            2,
                                            FruitRules.displayRadius(2),
                                            FruitRules.droppedPhysicsRadius(2),
                                            200f,
                                            500f,
                                            0f,
                                            0f,
                                            0f,
                                            0f,
                                            -1
                                    )
                            }
                    )
            );

            // 先制造一个未 drain 的事件，再恢复干净快照，确认旧合成不会重放。
            world.clear();
            world.addDroppedFruit(1, 250f, 700f);
            world.addDroppedFruit(1, 285f, 700f);
            world.step(0.02f);
            assertTrue(world.snapshot().nextFruitId() > 1);
            world.restore(
                    new FruitPhysicsWorld.Snapshot(
                            1,
                            0f,
                            new FruitPhysicsWorld.FruitState[0]
                    )
            );
            assertEquals(0, world.drainMergeEvents().size);
            assertEquals(0, world.fruits().size);
        } finally {
            world.dispose();
        }
    }

    private static void assertRejected(
            FruitPhysicsWorld world,
            FruitPhysicsWorld.Snapshot snapshot) {
        int originalId = world.fruits().first().id;
        assertThrows(
                IllegalArgumentException.class,
                () -> world.restore(snapshot)
        );
        assertEquals(1, world.fruits().size);
        assertEquals(originalId, world.fruits().first().id);
    }

    private static FruitPhysicsWorld.FruitState state(
            int id,
            int level,
            float physicsRadius) {
        return new FruitPhysicsWorld.FruitState(
                id,
                level,
                FruitRules.displayRadius(level),
                physicsRadius,
                200f + id,
                500f,
                0f,
                0f,
                0f,
                0f,
                id
        );
    }

    private static int[] ids(FruitPhysicsWorld.Snapshot snapshot) {
        int[] ids = new int[snapshot.fruitCount()];
        for (int index = 0; index < ids.length; index++) {
            ids[index] = snapshot.fruit(index).id();
        }
        return ids;
    }

    private static void assertStateEquals(
            FruitPhysicsWorld.FruitState expected,
            FruitPhysicsWorld.FruitState actual) {
        assertEquals(expected.id(), actual.id());
        assertEquals(expected.level(), actual.level());
        assertEquals(
                expected.displayRadius(),
                actual.displayRadius(),
                FLOAT_EPSILON
        );
        assertEquals(
                expected.physicsRadius(),
                actual.physicsRadius(),
                FLOAT_EPSILON
        );
        assertEquals(expected.x(), actual.x(), FLOAT_EPSILON);
        assertEquals(expected.y(), actual.y(), FLOAT_EPSILON);
        assertEquals(expected.vx(), actual.vx(), FLOAT_EPSILON);
        assertEquals(expected.vy(), actual.vy(), FLOAT_EPSILON);
        assertEquals(expected.angle(), actual.angle(), FLOAT_EPSILON);
        assertEquals(
                expected.angularVelocity(),
                actual.angularVelocity(),
                FLOAT_EPSILON
        );
        assertEquals(expected.ageFrames(), actual.ageFrames());
    }
}
