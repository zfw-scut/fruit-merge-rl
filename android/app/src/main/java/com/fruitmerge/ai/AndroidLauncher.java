package com.fruitmerge.ai;

import android.os.Bundle;

import com.badlogic.gdx.backends.android.AndroidApplication;
import com.badlogic.gdx.backends.android.AndroidApplicationConfiguration;
import com.fruitmerge.ai.game.FruitMergeApplication;

/** Android 原生入口：启动后台 ONNX runtime 与完整 libGDX 游戏。 */
public final class AndroidLauncher extends AndroidApplication {
    private AndroidAiRuntime aiRuntime;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        AndroidApplicationConfiguration configuration = new AndroidApplicationConfiguration();
        configuration.useImmersiveMode = true;
        configuration.useAccelerometer = false;
        configuration.useCompass = false;
        configuration.useWakelock = true;

        aiRuntime = new AndroidAiRuntime(this);
        initialize(new FruitMergeApplication(aiRuntime), configuration);
    }

    @Override
    protected void onDestroy() {
        if (aiRuntime != null) {
            aiRuntime.close();
            aiRuntime = null;
        }
        super.onDestroy();
    }
}
