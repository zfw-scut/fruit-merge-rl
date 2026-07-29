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

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.fruitmerge.ai.game.AiDecision;
import com.fruitmerge.ai.game.AiService;
import com.fruitmerge.ai.game.FruitRules;
import com.fruitmerge.ai.game.GameSnapshot;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

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

/**
 * Chaquopy 构图与 ONNX Runtime 推理的 Android 实现。
 *
 * <p>初始化、完整状态分析和五输入 GNN 推理全部在单独后台线程串行执行，避免阻塞
 * libGDX 渲染线程。模型不可用或单次输入失败时通过 {@link AiService} 的失败回调
 * 通知游戏；游戏仍会使用自己的安全启发式动作，因此 AI 开关不会让游戏卡死。</p>
 */
public final class AndroidAiRuntime implements AiService, AutoCloseable {
    private static final String MODEL_ASSET = "model/fruit_merge_ai.onnx";
    private static final String MODEL_SHA256 =
            "7e1c95c958799714579f75d8dbd8dc7b1e3ad182ab23fa3dac51306d4865e5f4";
    private static final int NODE_FEATURE_DIM = 62;
    private static final int EDGE_FEATURE_DIM = 47;

    // 不使用 Java 9 Set.of：应用最低支持 Android 7（API 24），这里保持纯 Java 8
    // collection API，避免旧设备在 AI runtime 初始化时触发 NoSuchMethodError。
    private static final Set<String> REQUIRED_INPUTS = Collections.unmodifiableSet(
            new HashSet<>(Arrays.asList(
                    "node_features",
                    "edge_index",
                    "edge_features",
                    "action_node_indices",
                    "global_node_index"
            ))
    );

    private final Context applicationContext;
    private final ExecutorService worker = Executors.newSingleThreadExecutor(runnable -> {
        Thread thread = new Thread(runnable, "fruit-merge-ai");
        thread.setDaemon(true);
        return thread;
    });

    private volatile OrtSession session;
    private volatile PyObject pythonBridge;
    private volatile boolean ready;
    private volatile boolean closed;
    private volatile String status = "loading model";

    public AndroidAiRuntime(Context context) {
        applicationContext = context.getApplicationContext();
        worker.execute(this::initialize);
    }

    private void initialize() {
        try {
            if (!Python.isStarted()) {
                throw new IllegalStateException("Chaquopy has not started");
            }

            PyObject bridge = Python.getInstance().getModule("mobile_bridge");
            String health = bridge.callAttr("healthcheck").toString();
            if (!"ready".equals(health)) {
                throw new IllegalStateException("Python bridge healthcheck failed");
            }

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
                // 手机端局面只有几十个节点。单线程避免小张量在线程池调度上的额外开销，
                // 同时不给 libGDX 渲染线程制造 CPU 争抢。
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
            pythonBridge = bridge;
            session = loadedSession;
            status = "model ready";
            ready = true;
        } catch (Exception error) {
            ready = false;
            status = "fallback: " + conciseError(error);
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
            PyObject bridge = pythonBridge;
            if (closed || currentSession == null || bridge == null) {
                callback.onFailure("AI runtime is unavailable");
                return;
            }

            String sceneJson = snapshotToJson(snapshot).toString();
            String graphJson = bridge
                    .callAttr("build_mobile_graph_json", sceneJson)
                    .toString();
            GraphInputs graph = GraphInputs.fromJson(new JSONObject(graphJson));
            float[] qValues = runModel(currentSession, graph);
            if (closed) {
                return;
            }
            callback.onSuccess(toDecision(qValues));
            status = String.format(
                    Locale.ROOT,
                    "model ready · %d nodes",
                    graph.nodeCount
            );
        } catch (Exception error) {
            // 局面序列化或构图失败不销毁已验证的 session。下一稳定边界仍可继续尝试；
            // 当前一步由 core 的安全 fallback 接管。
            if (closed) {
                return;
            }
            status = "last inference failed: " + conciseError(error);
            callback.onFailure(conciseError(error));
        }
    }

    private static JSONObject snapshotToJson(GameSnapshot snapshot)
            throws JSONException {
        JSONObject scene = new JSONObject();
        JSONObject geometry = new JSONObject();
        geometry.put("width", FruitRules.BOARD_WIDTH);
        geometry.put("height", FruitRules.BOARD_HEIGHT);
        geometry.put("spawn_y", FruitRules.SPAWN_Y);
        geometry.put("wall_width", FruitRules.WALL_WIDTH);
        geometry.put("floor_y", FruitRules.FLOOR_Y);
        scene.put("geometry", geometry);

        scene.put("score", snapshot.score);
        scene.put("last_score", snapshot.lastScore);
        scene.put("step_count", snapshot.stepCount);
        scene.put("physics_frame", 0);
        scene.put("done", false);
        scene.put("stable_boundary", true);

        JSONArray queue = new JSONArray();
        for (int level : snapshot.queue) {
            queue.put(level);
        }
        scene.put("fruit_queue", queue);

        JSONArray fruits = new JSONArray();
        for (GameSnapshot.FruitSnapshot fruit : snapshot.fruits) {
            JSONObject item = new JSONObject();
            item.put("fruit_id", fruit.id);
            item.put("level", fruit.level);
            item.put("radius", finite(fruit.displayRadius, "display radius"));
            item.put("physics_radius", finite(fruit.physicsRadius, "physics radius"));
            item.put("x", finite(fruit.x, "x"));
            item.put("y", finite(fruit.y, "y"));
            item.put("vx", finite(fruit.vx, "vx"));
            item.put("vy", finite(fruit.vy, "vy"));
            item.put("angle", finite(fruit.angle, "angle"));
            item.put(
                    "angular_velocity",
                    finite(fruit.angularVelocity, "angular velocity")
            );
            item.put("age_frames", fruit.ageFrames);
            item.put("stable", fruit.stable);
            fruits.put(item);
        }
        scene.put("fruits", fruits);
        return scene;
    }

    private static double finite(float value, String name) {
        if (!Float.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
        return value;
    }

    private static float[] runModel(OrtSession currentSession, GraphInputs graph)
            throws OrtException {
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        Map<String, OnnxTensor> inputs = new HashMap<>();
        try {
            inputs.put(
                    "node_features",
                    OnnxTensor.createTensor(
                            environment,
                            FloatBuffer.wrap(graph.nodeFeatures),
                            new long[]{graph.nodeCount, NODE_FEATURE_DIM}
                    )
            );
            inputs.put(
                    "edge_index",
                    OnnxTensor.createTensor(
                            environment,
                            LongBuffer.wrap(graph.edgeIndex),
                            new long[]{2, graph.edgeCount}
                    )
            );
            inputs.put(
                    "edge_features",
                    OnnxTensor.createTensor(
                            environment,
                            FloatBuffer.wrap(graph.edgeFeatures),
                            new long[]{graph.edgeCount, EDGE_FEATURE_DIM}
                    )
            );
            inputs.put(
                    "action_node_indices",
                    OnnxTensor.createTensor(
                            environment,
                            LongBuffer.wrap(graph.actionNodeIndices),
                            new long[]{FruitRules.ACTION_COUNT}
                    )
            );
            inputs.put(
                    "global_node_index",
                    OnnxTensor.createTensor(
                            environment,
                            LongBuffer.wrap(graph.globalNodeIndex),
                            new long[]{1}
                    )
            );

            try (OrtSession.Result result = currentSession.run(inputs)) {
                Optional<OnnxValue> named = result.get("q_values");
                OnnxValue output = named.orElseGet(() -> result.get(0));
                return qValues(output.getValue());
            }
        } finally {
            for (OnnxTensor tensor : inputs.values()) {
                tensor.close();
            }
        }
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
        for (float valueItem : values) {
            if (!Float.isFinite(valueItem)) {
                throw new IllegalStateException("model returned non-finite q value");
            }
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
                "onnx"
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
                // minSdk=24，需要保留 Android 7 的兼容分支。
                vibrator.vibrate(milliseconds);
            }
        } catch (RuntimeException ignored) {
            // 触觉反馈是装饰功能；系统策略拒绝震动时不能中断游戏循环。
        }
    }

    @Override
    public void close() {
        closed = true;
        ready = false;
        status = "closed";
        // OrtSession.close 不能和正在执行的 native run 并发。把释放动作排到同一个
        // 单线程队列尾部，使 Activity 销毁和后台推理之间不存在 use-after-close。
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
        pythonBridge = null;
        if (currentSession != null) {
            try {
                currentSession.close();
            } catch (OrtException ignored) {
                // Activity 已经销毁，无法再向用户展示关闭错误。
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

    private static String sha256(byte[] bytes)
            throws NoSuchAlgorithmException {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder hex = new StringBuilder(digest.length * 2);
        for (byte value : digest) {
            hex.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return hex.toString();
    }

    private static String conciseError(Throwable error) {
        String message = error.getMessage();
        if (message == null || message.trim().isEmpty()) {
            message = error.getClass().getSimpleName();
        }
        message = message.replace('\n', ' ').replace('\r', ' ').trim();
        return message.length() <= 96 ? message : message.substring(0, 96);
    }

    /** 已校验 shape 的五个 ONNX 输入。 */
    private static final class GraphInputs {
        final int nodeCount;
        final int edgeCount;
        final float[] nodeFeatures;
        final long[] edgeIndex;
        final float[] edgeFeatures;
        final long[] actionNodeIndices;
        final long[] globalNodeIndex;

        private GraphInputs(
                int nodeCount,
                int edgeCount,
                float[] nodeFeatures,
                long[] edgeIndex,
                float[] edgeFeatures,
                long[] actionNodeIndices,
                long[] globalNodeIndex) {
            this.nodeCount = nodeCount;
            this.edgeCount = edgeCount;
            this.nodeFeatures = nodeFeatures;
            this.edgeIndex = edgeIndex;
            this.edgeFeatures = edgeFeatures;
            this.actionNodeIndices = actionNodeIndices;
            this.globalNodeIndex = globalNodeIndex;
        }

        static GraphInputs fromJson(JSONObject root) throws JSONException {
            if (root.getInt("schema_version") != 1) {
                throw new IllegalArgumentException("unsupported mobile graph schema");
            }
            int[] nodeShape = intArray(root.getJSONArray("node_features_shape"));
            int[] edgeIndexShape = intArray(root.getJSONArray("edge_index_shape"));
            int[] edgeShape = intArray(root.getJSONArray("edge_features_shape"));
            if (nodeShape.length != 2
                    || nodeShape[0] <= 0
                    || nodeShape[1] != NODE_FEATURE_DIM) {
                throw new IllegalArgumentException("invalid node feature shape");
            }
            if (edgeIndexShape.length != 2
                    || edgeIndexShape[0] != 2
                    || edgeIndexShape[1] < 0) {
                throw new IllegalArgumentException("invalid edge index shape");
            }
            if (edgeShape.length != 2
                    || edgeShape[0] != edgeIndexShape[1]
                    || edgeShape[1] != EDGE_FEATURE_DIM) {
                throw new IllegalArgumentException("invalid edge feature shape");
            }

            float[] nodes = floatArray(root.getJSONArray("node_features"));
            long[] edges = longArray(root.getJSONArray("edge_index"));
            float[] edgeFeatures = floatArray(root.getJSONArray("edge_features"));
            long[] actionNodes = longArray(
                    root.getJSONArray("action_node_indices")
            );
            long[] globalNode = longArray(root.getJSONArray("global_node_index"));
            if (nodes.length != nodeShape[0] * NODE_FEATURE_DIM
                    || edges.length != 2 * edgeIndexShape[1]
                    || edgeFeatures.length != edgeShape[0] * EDGE_FEATURE_DIM
                    || actionNodes.length != FruitRules.ACTION_COUNT
                    || globalNode.length != 1) {
                throw new IllegalArgumentException("graph tensor length mismatch");
            }
            for (long index : actionNodes) {
                if (index < 0 || index >= nodeShape[0]) {
                    throw new IllegalArgumentException("action node index out of range");
                }
            }
            if (globalNode[0] < 0 || globalNode[0] >= nodeShape[0]) {
                throw new IllegalArgumentException("global node index out of range");
            }
            return new GraphInputs(
                    nodeShape[0],
                    edgeShape[0],
                    nodes,
                    edges,
                    edgeFeatures,
                    actionNodes,
                    globalNode
            );
        }

        private static int[] intArray(JSONArray array) throws JSONException {
            int[] values = new int[array.length()];
            for (int index = 0; index < values.length; index++) {
                values[index] = array.getInt(index);
            }
            return values;
        }

        private static long[] longArray(JSONArray array) throws JSONException {
            long[] values = new long[array.length()];
            for (int index = 0; index < values.length; index++) {
                values[index] = array.getLong(index);
            }
            return values;
        }

        private static float[] floatArray(JSONArray array) throws JSONException {
            float[] values = new float[array.length()];
            for (int index = 0; index < values.length; index++) {
                double value = array.getDouble(index);
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                            "graph contains non-finite feature"
                    );
                }
                values[index] = (float) value;
            }
            return values;
        }
    }
}
