package com.fruitmerge.ai.preview;

import com.badlogic.gdx.utils.Array;
import com.fruitmerge.ai.game.FruitPhysicsWorld;
import com.fruitmerge.ai.game.FruitRules;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;

/** 一次性验证 SAB-T120 在 Android 单环境训练同算法规则上的长局表现。 */
public final class SabT120RolloutEvaluator {
    private static final int MAX_FRUITS = 64;
    private static final int MAX_DROPS = 1000;
    private static final int MAX_PHYSICS_FRAMES =
            FruitRules.MAX_PHYSICS_FRAMES_PER_DROP;
    private static final int STABLE_FRAMES = FruitRules.STABLE_FRAMES;
    private static final int DANGER_FRAMES = FruitRules.DANGER_FRAMES;
    private static final float DT = 1f / FruitRules.PHYSICS_FPS;

    private SabT120RolloutEvaluator() {
    }

    public static void main(String[] args) throws Exception {
        Path modelPath = Path.of(requiredProperty("fruitMerge.model"));
        int games = Integer.getInteger("fruitMerge.games", 32);
        long seedBase = Long.getLong("fruitMerge.seedBase", 42_000_000L);
        long seedStride = Long.getLong("fruitMerge.seedStride", 1_000_003L);
        if (!Files.isRegularFile(modelPath)) {
            throw new IllegalArgumentException("model does not exist: " + modelPath);
        }
        if (games <= 0) {
            throw new IllegalArgumentException("games must be positive");
        }

        int[] scores = new int[games];
        int[] drops = new int[games];
        int failures = 0;
        try (InferenceClient inference = new InferenceClient(
                requiredProperty("fruitMerge.python"),
                requiredProperty("fruitMerge.inferenceServer"),
                modelPath.toString())) {
            for (int game = 0; game < games; game++) {
                long seed = seedBase + game * seedStride;
                GameResult result = play(inference, seed);
                scores[game] = result.score;
                drops[game] = result.drops;
                if (result.failed) {
                    failures += 1;
                }
                System.out.printf(
                        Locale.ROOT,
                        "game=%d seed=%d score=%d drops=%d failed=%s%n",
                        game,
                        seed,
                        result.score,
                        result.drops,
                        result.failed
                );
            }
        }

        int[] sortedScores = scores.clone();
        Arrays.sort(sortedScores);
        System.out.printf(
                Locale.ROOT,
                "RESULT games=%d mean_score=%.2f median_score=%.2f mean_drops=%.2f "
                        + "failures=%d min_score=%d max_score=%d%n",
                games,
                mean(scores),
                median(sortedScores),
                mean(drops),
                failures,
                sortedScores[0],
                sortedScores[sortedScores.length - 1]
        );
    }

    private static GameResult play(InferenceClient inference, long seed)
            throws IOException {
        FruitPhysicsWorld world = new FruitPhysicsWorld();
        TensorLcg random = new TensorLcg(seed);
        int[] queue = new int[FruitRules.QUEUE_LENGTH];
        for (int index = 0; index < queue.length; index++) {
            queue[index] = random.nextLevel();
        }
        int score = 0;
        int dangerFrames = 0;
        boolean failed = false;
        int completedDrops = 0;
        try {
            while (completedDrops < MAX_DROPS && !failed) {
                if (world.fruits().size >= MAX_FRUITS) {
                    failed = true;
                    break;
                }
                int action = inference.infer(
                        world,
                        queue,
                        Math.min(1f, dangerFrames / (float) DANGER_FRAMES),
                        world.hasFruitTopAboveSpawnLine()
                );
                int level = queue[0];
                world.addDroppedFruit(
                        level,
                        FruitRules.actionDropX(action, level),
                        FruitRules.SPAWN_Y
                );
                System.arraycopy(queue, 1, queue, 0, queue.length - 1);
                queue[queue.length - 1] = random.nextLevel();
                completedDrops += 1;

                int stableFrames = 0;
                for (int frame = 0; frame < MAX_PHYSICS_FRAMES; frame++) {
                    world.step(DT);
                    Array<FruitPhysicsWorld.MergeEvent> events =
                            world.drainMergeEvents();
                    for (FruitPhysicsWorld.MergeEvent event : events) {
                        score += event.scoreDelta;
                    }

                    boolean overLine = world.isOverDangerLine();
                    dangerFrames = overLine ? dangerFrames + 1 : 0;
                    if (dangerFrames > DANGER_FRAMES) {
                        failed = true;
                        break;
                    }
                    stableFrames = world.isStable() ? stableFrames + 1 : 0;
                    if (stableFrames >= STABLE_FRAMES) {
                        break;
                    }
                }
            }
            return new GameResult(score, completedDrops, failed);
        } finally {
            world.dispose();
        }
    }

    private static String requiredProperty(String name) {
        String value = System.getProperty(name, "").trim();
        if (value.isEmpty()) {
            throw new IllegalArgumentException("missing system property: " + name);
        }
        return value;
    }

    private static double mean(int[] values) {
        long sum = 0L;
        for (int value : values) {
            sum += value;
        }
        return sum / (double) values.length;
    }

    private static double median(int[] sorted) {
        int middle = sorted.length / 2;
        return sorted.length % 2 == 0
                ? (sorted[middle - 1] + sorted[middle]) / 2.0
                : sorted[middle];
    }

    private static final class TensorLcg {
        private long state;

        TensorLcg(long seed) {
            state = seed & 0x7fffffffL;
        }

        int nextLevel() {
            state = (state * 1_103_515_245L + 12_345L) & 0x7fffffffL;
            return (int) (state % 5L) + 1;
        }
    }

    private static final class InferenceClient implements AutoCloseable {
        private final Process process;
        private final BufferedWriter input;
        private final BufferedReader output;

        InferenceClient(String python, String server, String model)
                throws IOException {
            process = new ProcessBuilder(python, server, model)
                    .redirectError(ProcessBuilder.Redirect.INHERIT)
                    .start();
            input = new BufferedWriter(new OutputStreamWriter(
                    process.getOutputStream()));
            output = new BufferedReader(new InputStreamReader(
                    process.getInputStream()));
            String greeting = output.readLine();
            if (!"READY".equals(greeting)) {
                throw new IOException("inference server failed to start: " + greeting);
            }
        }

        int infer(
                FruitPhysicsWorld world,
                int[] queue,
                float dangerProgress,
                boolean overDangerLine) throws IOException {
            StringBuilder line = new StringBuilder(8192);
            FruitPhysicsWorld.FruitBody[] slots = new FruitPhysicsWorld.FruitBody[MAX_FRUITS];
            for (FruitPhysicsWorld.FruitBody fruit : world.fruits()) {
                slots[fruit.slot] = fruit;
            }
            for (int slot = 0; slot < MAX_FRUITS; slot++) {
                FruitPhysicsWorld.FruitBody fruit = slots[slot];
                append(line, fruit == null ? 0f : fruit.x());
                append(line, fruit == null ? 0f : fruit.y());
            }
            for (int slot = 0; slot < MAX_FRUITS; slot++) {
                FruitPhysicsWorld.FruitBody fruit = slots[slot];
                append(line, fruit == null ? 0f : fruit.vx());
                append(line, fruit == null ? 0f : fruit.vy());
            }
            for (int slot = 0; slot < MAX_FRUITS; slot++) {
                append(line, slots[slot] == null
                        ? 0f : slots[slot].angularVelocity());
            }
            for (int slot = 0; slot < MAX_FRUITS; slot++) {
                append(line, slots[slot] == null ? 0 : slots[slot].level);
            }
            for (int slot = 0; slot < MAX_FRUITS; slot++) {
                append(line, slots[slot] == null
                        ? 0f : slots[slot].physicsRadius);
            }
            for (int slot = 0; slot < MAX_FRUITS; slot++) {
                append(line, slots[slot] == null
                        ? 0 : slots[slot].ageFrames());
            }
            for (int slot = 0; slot < MAX_FRUITS; slot++) {
                append(line, slots[slot] == null ? 0 : 1);
            }
            for (int level : queue) {
                append(line, level);
            }
            append(line, dangerProgress);
            append(line, overDangerLine ? 1 : 0);
            input.write(line.toString());
            input.newLine();
            input.flush();
            String response = output.readLine();
            if (response == null) {
                throw new IOException("inference server closed unexpectedly");
            }
            int action;
            try {
                action = Integer.parseInt(response);
            } catch (NumberFormatException error) {
                throw new IOException("invalid inference response: " + response, error);
            }
            if (action < 0 || action >= FruitRules.ACTION_COUNT) {
                throw new IOException("inference action is out of range: " + action);
            }
            return action;
        }

        private static void append(StringBuilder target, float value) {
            if (target.length() > 0) {
                target.append(' ');
            }
            target.append(value);
        }

        private static void append(StringBuilder target, int value) {
            if (target.length() > 0) {
                target.append(' ');
            }
            target.append(value);
        }

        @Override
        public void close() throws IOException {
            try {
                input.write("STOP");
                input.newLine();
                input.flush();
            } catch (IOException ignored) {
                // 服务器已退出时只需回收进程。
            }
            try {
                input.close();
            } finally {
                output.close();
                process.destroy();
            }
        }
    }

    private static final class GameResult {
        final int score;
        final int drops;
        final boolean failed;

        GameResult(int score, int drops, boolean failed) {
            this.score = score;
            this.drops = drops;
            this.failed = failed;
        }
    }
}
