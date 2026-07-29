package com.fruitmerge.ai.game;

/**
 * core 游戏循环访问 Android Python/ONNX 服务的窄接口。
 *
 * <p>回调允许在任意后台线程触发；游戏实现必须通过
 * {@code Gdx.app.postRunnable} 把结果送回渲染线程。</p>
 */
public interface AiService {
    boolean isAiReady();

    String aiRuntimeStatus();

    void requestDecision(GameSnapshot snapshot, DecisionCallback callback);

    void vibrate(int milliseconds);

    interface DecisionCallback {
        void onSuccess(AiDecision decision);

        void onFailure(String message);
    }
}
