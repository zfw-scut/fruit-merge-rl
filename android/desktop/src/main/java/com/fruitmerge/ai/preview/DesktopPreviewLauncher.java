package com.fruitmerge.ai.preview;

import com.badlogic.gdx.ApplicationListener;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.Pixmap;
import com.badlogic.gdx.graphics.PixmapIO;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;
import com.badlogic.gdx.graphics.glutils.HdpiMode;
import com.badlogic.gdx.utils.BufferUtils;
import com.badlogic.gdx.utils.ScreenUtils;
import com.fruitmerge.ai.game.AiService;
import com.fruitmerge.ai.game.FruitMergeApplication;
import com.fruitmerge.ai.game.GameSnapshot;

import java.nio.ByteBuffer;

/**
 * Windows 本地 UI 预览入口。
 *
 * <p>它直接运行 Android 使用的 {@link FruitMergeApplication}，因此字体、卡片、
 * 水果、逻辑坐标和绘制顺序都来自同一份源码。唯一被替换的是 Android 的 ONNX 服务：
 * 桌面预览使用不可用状态的窄 stub，让 core 自动走已有安全启发式，不加载模型。</p>
 */
public final class DesktopPreviewLauncher {
    private static final String OUTPUT_PROPERTY = "fruitMerge.previewOutput";
    private static final String WIDTH_PROPERTY = "fruitMerge.previewWidth";
    private static final String HEIGHT_PROPERTY = "fruitMerge.previewHeight";

    private DesktopPreviewLauncher() {
        // 入口类不需要实例。
    }

    public static void main(String[] args) {
        String outputPath = System.getProperty(OUTPUT_PROPERTY, "").trim();
        int previewWidth = Integer.getInteger(WIDTH_PROPERTY, 560);
        int previewHeight = Integer.getInteger(HEIGHT_PROPERTY, 1120);
        Lwjgl3ApplicationConfiguration configuration =
                new Lwjgl3ApplicationConfiguration();
        configuration.setTitle("Fruit Merge AI - Local UI Preview");
        configuration.setWindowedMode(previewWidth, previewHeight);
        configuration.setResizable(false);
        configuration.setHdpiMode(HdpiMode.Pixels);
        configuration.setInitialVisible(outputPath.isEmpty());
        configuration.useVsync(true);
        configuration.setForegroundFPS(60);

        FruitMergeApplication game =
                new FruitMergeApplication(new PreviewAiService());
        new Lwjgl3Application(
                new PreviewApplication(game, outputPath),
                configuration
        );
    }

    /**
     * 把真实游戏监听器包一层，仅在指定系统属性时截取最终 framebuffer。
     * 普通 {@code :desktop:run} 不设置属性，因此会保留可交互窗口。
     */
    private static final class PreviewApplication implements ApplicationListener {
        private static final int CAPTURE_AFTER_FRAMES = 12;

        private final FruitMergeApplication delegate;
        private final String outputPath;
        private int renderedFrames;
        private boolean captured;

        private PreviewApplication(
                FruitMergeApplication delegate,
                String outputPath) {
            this.delegate = delegate;
            this.outputPath = outputPath;
        }

        @Override
        public void create() {
            delegate.create();
        }

        @Override
        public void resize(int width, int height) {
            delegate.resize(width, height);
        }

        @Override
        public void render() {
            delegate.render();
            renderedFrames += 1;
            if (!captured
                    && !outputPath.isEmpty()
                    && renderedFrames >= CAPTURE_AFTER_FRAMES) {
                captured = true;
                writeScreenshot(outputPath);
                Gdx.app.exit();
            }
        }

        private void writeScreenshot(String absolutePath) {
            int width = Gdx.graphics.getBackBufferWidth();
            int height = Gdx.graphics.getBackBufferHeight();
            byte[] pixels = ScreenUtils.getFrameBufferPixels(
                    0,
                    0,
                    width,
                    height,
                    true
            );
            Pixmap pixmap = new Pixmap(
                    width,
                    height,
                    Pixmap.Format.RGBA8888
            );
            ByteBuffer target = pixmap.getPixels();
            BufferUtils.copy(pixels, 0, target, pixels.length);
            Gdx.files.absolute(absolutePath).parent().mkdirs();
            PixmapIO.writePNG(Gdx.files.absolute(absolutePath), pixmap);
            pixmap.dispose();
            System.out.println("UI_PREVIEW=" + absolutePath);
        }

        @Override
        public void pause() {
            delegate.pause();
        }

        @Override
        public void resume() {
            delegate.resume();
        }

        @Override
        public void dispose() {
            delegate.dispose();
        }
    }

    /** 不加载移动模型；core 会自动使用项目现有的本地安全策略。 */
    private static final class PreviewAiService implements AiService {
        @Override
        public boolean isAiReady() {
            return false;
        }

        @Override
        public String aiRuntimeStatus() {
            return "desktop preview";
        }

        @Override
        public void requestDecision(
                GameSnapshot snapshot,
                DecisionCallback callback) {
            callback.onFailure("desktop preview does not load ONNX");
        }

        @Override
        public void vibrate(int milliseconds) {
            // Windows 预览不模拟手机触觉反馈。
        }
    }
}
