package com.fruitmerge.ai;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Build;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.os.VibratorManager;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import com.fruitmerge.ai.game.AiDecision;
import com.fruitmerge.ai.game.AiService;
import com.fruitmerge.ai.game.FruitRules;
import com.fruitmerge.ai.game.GameSnapshot;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;

/** 无旁路 SAB-FF120 定长状态张量的 Android ONNX Runtime 实现。 */
public final class AndroidAiRuntime implements AiService, AutoCloseable {
    private static final String MODEL_ASSET = "model/sab_ff120.onnx";
    private static final String MODEL_SHA256 =
            "b10b8fbf853e07397e9a1774b1b7fd394e42502d41ff7700cfb76cbf5a4a4475";
    private static final int MAX_FRUITS = 64;

    private static final Set<String> REQUIRED_INPUTS = Collections.unmodifiableSet(
            new HashSet<>(Arrays.asList(
                    "positions",
                    "velocities",
                    "angular_velocities",
                    "levels",
                    "physics_radii",
                    "age_frames",
                    "active",
                    "fruit_queue",
                    "danger_progress",
                    "over_danger_line"
            ))
    );

    private final Context applicationContext;
    private final ExecutorService worker = Executors.newSingleThreadExecutor(runnable -> {
        Thread thread = new Thread(runnable, "sab-ff120-onnx");
        thread.setDaemon(true);
        return thread;
    });

    private volatile OrtSession session;
    private volatile boolean ready;
    private volatile boolean closed;
    private volatile String status = "正在载入 SAB-FF120";

    public AndroidAiRuntime(Context context) {
        applicationContext = context.getApplicationContext();
        worker.execute(this::initialize);
    }

    private void initialize() {
        try {
            byte[] modelBytes = readAsset(applicationContext.getAssets(), MODEL_ASSET);
            String actualHash = sha256(modelBytes);
            if (!MODEL_SHA256.equals(actualHash)) {
                throw new IllegalStateException(
                        "model integrity mismatch: " + actualHash.substring(0, 12)
                );
            }

            OrtEnvironment environment = OrtEnvironment.getEnvironment();
            OrtSession loadedSession;
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.setIntraOpNumThreads(1);
                options.setInterOpNumThreads(1);
                loadedSession = environment.createSession(modelBytes, options);
            }
            if (!loadedSession.getInputNames().equals(REQUIRED_INPUTS)) {
                loadedSession.close();
                throw new IllegalStateException(
                        "unexpected ONNX inputs: " + loadedSession.getInputNames()
                );
            }
            if (!loadedSession.getOutputNames().contains("q_values")) {
                loadedSession.close();
                throw new IllegalStateException("ONNX output q_values is missing");
            }
            if (closed) {
                loadedSession.close();
                return;
            }
            session = loadedSession;
            ready = true;
            status = "SAB-FF120 已就绪";
        } catch (Exception error) {
            ready = false;
            status = "安全策略：" + conciseError(error);
        }
    }

    @Override
    public boolean isAiReady() {
        return ready && !closed;
    }

    @Override
    public String aiRuntimeStatus() {
        return status;
    }

    @Override
    public void requestDecision(GameSnapshot snapshot, DecisionCallback callback) {
        if (snapshot == null || callback == null) {
            throw new IllegalArgumentException("snapshot and callback are required");
        }
        if (!isAiReady()) {
            callback.onFailure(status);
            return;
        }
        try {
            worker.execute(() -> infer(snapshot, callback));
        } catch (RejectedExecutionException error) {
            callback.onFailure("AI runtime is closed");
        }
    }

    private void infer(GameSnapshot snapshot, DecisionCallback callback) {
        try {
            OrtSession currentSession = session;
            if (closed || currentSession == null) {
                callback.onFailure("AI runtime is unavailable");
                return;
            }
            DenseInputs inputs = DenseInputs.fromSnapshot(snapshot);
            float[] qValues = runModel(currentSession, inputs);
            if (!closed) {
                callback.onSuccess(toDecision(qValues));
                status = String.format(
                        Locale.ROOT,
                        "SAB-FF120 已就绪 · %d fruits",
                        snapshot.fruits.size()
                );
            }
        } catch (Exception error) {
            if (!closed) {
                status = "上次推理失败：" + conciseError(error);
                callback.onFailure(conciseError(error));
            }
        }
    }

    private static float[] runModel(OrtSession currentSession, DenseInputs dense)
            throws OrtException {
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        Map<String, OnnxTensor> tensors = new HashMap<>();
        try {
            tensors.put("positions", floatTensor(environment, dense.positions, 1, MAX_FRUITS, 2));
            tensors.put("velocities", floatTensor(environment, dense.velocities, 1, MAX_FRUITS, 2));
            tensors.put("angular_velocities", floatTensor(environment, dense.angularVelocities, 1, MAX_FRUITS));
            tensors.put("levels", longTensor(environment, dense.levels, 1, MAX_FRUITS));
            tensors.put("physics_radii", floatTensor(environment, dense.physicsRadii, 1, MAX_FRUITS));
            tensors.put("age_frames", longTensor(environment, dense.ageFrames, 1, MAX_FRUITS));
            tensors.put("active", longTensor(environment, dense.active, 1, MAX_FRUITS));
            tensors.put("fruit_queue", longTensor(environment, dense.fruitQueue, 1, FruitRules.QUEUE_LENGTH));
            tensors.put("danger_progress", floatTensor(environment, dense.dangerProgress, 1));
            tensors.put("over_danger_line", longTensor(environment, dense.overDangerLine, 1));

            try (OrtSession.Result result = currentSession.run(tensors)) {
                Optional<OnnxValue> named = result.get("q_values");
                OnnxValue output = named.orElseGet(() -> result.get(0));
                return qValues(output.getValue());
            }
        } finally {
            for (OnnxTensor tensor : tensors.values()) {
                tensor.close();
            }
        }
    }

    private static OnnxTensor floatTensor(
            OrtEnvironment environment,
            float[] values,
            long... shape) throws OrtException {
        return OnnxTensor.createTensor(environment, FloatBuffer.wrap(values), shape);
    }

    private static OnnxTensor longTensor(
            OrtEnvironment environment,
            long[] values,
            long... shape) throws OrtException {
        return OnnxTensor.createTensor(environment, LongBuffer.wrap(values), shape);
    }

    private static float[] qValues(Object value) {
        float[] values;
        if (value instanceof float[]) {
            values = ((float[]) value).clone();
        } else if (value instanceof float[][] && ((float[][]) value).length == 1) {
            values = ((float[][]) value)[0].clone();
        } else {
            throw new IllegalStateException(
                    "unexpected q_values type: "
                            + (value == null ? "null" : value.getClass().getName())
            );
        }
        if (values.length != FruitRules.ACTION_COUNT) {
            throw new IllegalStateException(
                    "expected " + FruitRules.ACTION_COUNT
                            + " q values, got " + values.length
            );
        }
        for (float item : values) {
            requireFinite(item, "q value");
        }
        return values;
    }

    private static AiDecision toDecision(float[] qValues) {
        int best = 0;
        int second = 1;
        if (qValues[second] > qValues[best]) {
            int swap = best;
            best = second;
            second = swap;
        }
        for (int index = 2; index < qValues.length; index++) {
            if (qValues[index] > qValues[best]) {
                second = best;
                best = index;
            } else if (qValues[index] > qValues[second]) {
                second = index;
            }
        }
        return new AiDecision(
                best,
                second,
                qValues[best],
                qValues[second],
                qValues,
                "SAB-FF120 ONNX"
        );
    }

    @Override
    public void vibrate(int milliseconds) {
        if (milliseconds <= 0 || closed) {
            return;
        }
        try {
            Vibrator vibrator;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                VibratorManager manager = (VibratorManager) applicationContext
                        .getSystemService(Context.VIBRATOR_MANAGER_SERVICE);
                vibrator = manager == null ? null : manager.getDefaultVibrator();
            } else {
                vibrator = (Vibrator) applicationContext
                        .getSystemService(Context.VIBRATOR_SERVICE);
            }
            if (vibrator == null || !vibrator.hasVibrator()) {
                return;
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createOneShot(
                        milliseconds,
                        VibrationEffect.DEFAULT_AMPLITUDE
                ));
            } else {
                vibrator.vibrate(milliseconds);
            }
        } catch (RuntimeException ignored) {
            // 触觉反馈不可用不影响游戏。
        }
    }

    @Override
    public void close() {
        closed = true;
        ready = false;
        status = "closed";
        try {
            worker.execute(this::releaseSession);
        } catch (RejectedExecutionException ignored) {
            releaseSession();
        }
        worker.shutdown();
    }

    private void releaseSession() {
        OrtSession currentSession = session;
        session = null;
        if (currentSession != null) {
            try {
                currentSession.close();
            } catch (OrtException ignored) {
                // Activity 已销毁，无需展示关闭异常。
            }
        }
    }

    private static byte[] readAsset(AssetManager assets, String name)
            throws IOException {
        try (InputStream input = assets.open(name);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[32 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    private static String sha256(byte[] bytes) throws NoSuchAlgorithmException {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder hex = new StringBuilder(digest.length * 2);
        for (byte value : digest) {
            hex.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return hex.toString();
    }

    private static void requireFinite(float value, String name) {
        if (!Float.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
    }

    private static String conciseError(Throwable error) {
        String message = error.getMessage();
        if (message == null || message.trim().isEmpty()) {
            message = error.getClass().getSimpleName();
        }
        message = message.replace('\n', ' ').replace('\r', ' ').trim();
        return message.length() <= 96 ? message : message.substring(0, 96);
    }

    private static final class DenseInputs {
        final float[] positions = new float[MAX_FRUITS * 2];
        final float[] velocities = new float[MAX_FRUITS * 2];
        final float[] angularVelocities = new float[MAX_FRUITS];
        final long[] levels = new long[MAX_FRUITS];
        final float[] physicsRadii = new float[MAX_FRUITS];
        final long[] ageFrames = new long[MAX_FRUITS];
        final long[] active = new long[MAX_FRUITS];
        final long[] fruitQueue = new long[FruitRules.QUEUE_LENGTH];
        final float[] dangerProgress = new float[1];
        final long[] overDangerLine = new long[1];

        static DenseInputs fromSnapshot(GameSnapshot snapshot) {
            if (snapshot.fruits.size() > MAX_FRUITS) {
                throw new IllegalArgumentException(
                        "scene has more than " + MAX_FRUITS + " fruits"
                );
            }
            if (snapshot.queue.length != FruitRules.QUEUE_LENGTH) {
                throw new IllegalArgumentException("fruit queue length must be 4");
            }

            DenseInputs dense = new DenseInputs();
            boolean[] occupiedSlots = new boolean[MAX_FRUITS];
            for (GameSnapshot.FruitSnapshot fruit : snapshot.fruits) {
                int index = fruit.slot;
                if (index < 0 || index >= MAX_FRUITS || occupiedSlots[index]) {
                    throw new IllegalArgumentException("fruit slot is invalid or duplicated");
                }
                occupiedSlots[index] = true;
                if (fruit.level < FruitRules.MIN_LEVEL
                        || fruit.level > FruitRules.MAX_LEVEL) {
                    throw new IllegalArgumentException("fruit level is out of range");
                }
                if (fruit.ageFrames < 0) {
                    throw new IllegalArgumentException("fruit age must be non-negative");
                }
                requireFinite(fruit.x, "x");
                requireFinite(fruit.y, "y");
                requireFinite(fruit.vx, "vx");
                requireFinite(fruit.vy, "vy");
                requireFinite(fruit.angularVelocity, "angular velocity");
                requireFinite(fruit.physicsRadius, "physics radius");
                dense.positions[index * 2] = fruit.x;
                dense.positions[index * 2 + 1] = fruit.y;
                dense.velocities[index * 2] = fruit.vx;
                dense.velocities[index * 2 + 1] = fruit.vy;
                dense.angularVelocities[index] = fruit.angularVelocity;
                dense.levels[index] = fruit.level;
                dense.physicsRadii[index] = fruit.physicsRadius;
                dense.ageFrames[index] = fruit.ageFrames;
                dense.active[index] = 1L;
            }
            for (int index = 0; index < snapshot.queue.length; index++) {
                int level = snapshot.queue[index];
                if (level < FruitRules.SPAWN_MIN_LEVEL
                        || level > FruitRules.SPAWN_MAX_LEVEL) {
                    throw new IllegalArgumentException("queue level is out of range");
                }
                dense.fruitQueue[index] = level;
            }
            requireFinite(snapshot.dangerProgress, "danger progress");
            dense.dangerProgress[0] = Math.max(0f, Math.min(1f, snapshot.dangerProgress));
            dense.overDangerLine[0] = snapshot.overDangerLine ? 1L : 0L;
            return dense;
        }
    }
}
