package com.fruitmerge.ai.preview;

import com.badlogic.gdx.ApplicationListener;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.Pixmap;
import com.badlogic.gdx.graphics.PixmapIO;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;
import com.badlogic.gdx.graphics.glutils.FrameBuffer;
import com.badlogic.gdx.graphics.glutils.HdpiMode;
import com.badlogic.gdx.files.FileHandle;
import com.badlogic.gdx.utils.BufferUtils;
import com.badlogic.gdx.utils.GdxRuntimeException;
import com.badlogic.gdx.utils.ScreenUtils;
import com.fruitmerge.ai.game.AiService;
import com.fruitmerge.ai.game.FruitMergeApplication;
import com.fruitmerge.ai.game.GameSnapshot;

import java.io.DataInputStream;
import java.io.IOException;
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
    private static final String SHOWCASE_PROPERTY =
            "fruitMerge.previewShowcase";
    private static final String CAPTURE_FRAMES_PROPERTY =
            "fruitMerge.previewCaptureFrames";

    private DesktopPreviewLauncher() {
        // 入口类不需要实例。
    }

    public static void main(String[] args) {
        String outputPath = System.getProperty(OUTPUT_PROPERTY, "").trim();
        int previewWidth = Integer.getInteger(WIDTH_PROPERTY, 560);
        int previewHeight = Integer.getInteger(HEIGHT_PROPERTY, 1120);
        boolean showcase = Boolean.getBoolean(SHOWCASE_PROPERTY);
        int captureFrames = Math.max(
                1,
                Integer.getInteger(CAPTURE_FRAMES_PROPERTY, 12)
        );
        boolean capture = !outputPath.isEmpty();
        Lwjgl3ApplicationConfiguration configuration =
                new Lwjgl3ApplicationConfiguration();
        configuration.setTitle("Fruit Merge AI - Local UI Preview");
        /*
         * Windows 会把高于桌面工作区的窗口静默缩小。自动截图不再依赖宿主窗口
         * 的 backbuffer，只创建一个适中的隐藏窗口提供 OpenGL 上下文；真正的
         * previewWidth × previewHeight 由 PreviewApplication 的 FBO 保证。
         */
        configuration.setWindowedMode(
                capture ? Math.min(previewWidth, 560) : previewWidth,
                capture ? Math.min(previewHeight, 960) : previewHeight
        );
        configuration.setResizable(false);
        configuration.setHdpiMode(HdpiMode.Pixels);
        configuration.setInitialVisible(!capture);
        configuration.useVsync(true);
        configuration.setForegroundFPS(60);

        FruitMergeApplication game =
                new FruitMergeApplication(new PreviewAiService());
        new Lwjgl3Application(
                new PreviewApplication(
                        game,
                        outputPath,
                        previewWidth,
                        previewHeight,
                        showcase,
                        captureFrames
                ),
                configuration
        );
    }

    /**
     * 把真实游戏监听器包一层，仅在指定系统属性时截取最终 framebuffer。
     * 普通 {@code :desktop:run} 不设置属性，因此会保留可交互窗口。
     */
    private static final class PreviewApplication implements ApplicationListener {
        private final FruitMergeApplication delegate;
        private final String outputPath;
        private final int captureWidth;
        private final int captureHeight;
        private final boolean showcase;
        private final int captureAfterFrames;
        private FrameBuffer captureBuffer;
        private int renderedFrames;
        private boolean captured;

        private PreviewApplication(
                FruitMergeApplication delegate,
                String outputPath,
                int captureWidth,
                int captureHeight,
                boolean showcase,
                int captureAfterFrames) {
            this.delegate = delegate;
            this.outputPath = outputPath;
            this.captureWidth = captureWidth;
            this.captureHeight = captureHeight;
            this.showcase = showcase;
            this.captureAfterFrames = captureAfterFrames;
        }

        @Override
        public void create() {
            delegate.create();
            if (isCaptureMode()) {
                captureBuffer = new FrameBuffer(
                        Pixmap.Format.RGBA8888,
                        captureWidth,
                        captureHeight,
                        false
                );
                delegate.resize(captureWidth, captureHeight);
            }
            if (showcase) {
                delegate.startPresentationShowcase();
            }
        }

        @Override
        public void resize(int width, int height) {
            if (isCaptureMode()) {
                // 宿主隐藏窗口可能被操作系统缩小，不能覆盖 FBO 的目标逻辑尺寸。
                delegate.resize(captureWidth, captureHeight);
            } else {
                delegate.resize(width, height);
            }
        }

        @Override
        public void render() {
            if (captureBuffer == null) {
                delegate.render();
                return;
            }

            boolean exitAfterFrame = false;
            captureBuffer.begin();
            try {
                /*
                 * FrameBuffer.begin 会把 GL viewport 切到离屏纹理；随后让同一个
                 * FruitMergeApplication 按目标像素尺寸更新 FitViewport 并绘制。
                 */
                delegate.resize(captureWidth, captureHeight);
                delegate.render();
                renderedFrames += 1;
                if (!captured && renderedFrames >= captureAfterFrames) {
                    captured = true;
                    writeScreenshot(outputPath, captureWidth, captureHeight);
                    exitAfterFrame = true;
                }
            } finally {
                captureBuffer.end();
            }
            if (exitAfterFrame) {
                Gdx.app.exit();
            }
        }

        private boolean isCaptureMode() {
            return !outputPath.isEmpty();
        }

        private void writeScreenshot(
                String absolutePath,
                int expectedWidth,
                int expectedHeight) {
            int framebufferWidth =
                    captureBuffer.getColorBufferTexture().getWidth();
            int framebufferHeight =
                    captureBuffer.getColorBufferTexture().getHeight();
            if (framebufferWidth != expectedWidth
                    || framebufferHeight != expectedHeight) {
                throw new GdxRuntimeException(
                        "Preview FBO size mismatch: expected "
                                + expectedWidth + "x" + expectedHeight
                                + ", got "
                                + framebufferWidth + "x" + framebufferHeight
                );
            }

            byte[] pixels = ScreenUtils.getFrameBufferPixels(
                    0,
                    0,
                    framebufferWidth,
                    framebufferHeight,
                    true
            );
            int expectedBytes = Math.multiplyExact(
                    Math.multiplyExact(framebufferWidth, framebufferHeight),
                    4
            );
            if (pixels.length != expectedBytes) {
                throw new GdxRuntimeException(
                        "Preview pixel count mismatch: expected "
                                + expectedBytes + " bytes, got " + pixels.length
                );
            }
            Pixmap pixmap = new Pixmap(
                    framebufferWidth,
                    framebufferHeight,
                    Pixmap.Format.RGBA8888
            );
            FileHandle output = Gdx.files.absolute(absolutePath);
            try {
                ByteBuffer target = pixmap.getPixels();
                BufferUtils.copy(pixels, 0, target, pixels.length);
                output.parent().mkdirs();
                PixmapIO.writePNG(output, pixmap);
            } finally {
                pixmap.dispose();
            }
            verifyPngDimensions(output, expectedWidth, expectedHeight);
            System.out.println(
                    "UI_PREVIEW=" + absolutePath
                            + " SIZE=" + expectedWidth + "x" + expectedHeight
            );
        }

        /**
         * 直接校验 PNG 文件头中的 IHDR，而不是相信请求尺寸或 FBO 元数据。
         */
        private void verifyPngDimensions(
                FileHandle output,
                int expectedWidth,
                int expectedHeight) {
            try (DataInputStream input =
                         new DataInputStream(output.read())) {
                int signatureFirst = input.readInt();
                int signatureSecond = input.readInt();
                int ihdrLength = input.readInt();
                int ihdrType = input.readInt();
                int actualWidth = input.readInt();
                int actualHeight = input.readInt();
                if (signatureFirst != 0x89504E47
                        || signatureSecond != 0x0D0A1A0A
                        || ihdrLength != 13
                        || ihdrType != 0x49484452) {
                    throw new GdxRuntimeException(
                            "Preview output is not a valid PNG with IHDR: "
                                    + output.path()
                    );
                }
                if (actualWidth != expectedWidth
                        || actualHeight != expectedHeight) {
                    throw new GdxRuntimeException(
                            "Preview PNG size mismatch: expected "
                                    + expectedWidth + "x" + expectedHeight
                                    + ", got "
                                    + actualWidth + "x" + actualHeight
                    );
                }
            } catch (IOException error) {
                throw new GdxRuntimeException(
                        "Unable to verify preview PNG: " + output.path(),
                        error
                );
            }
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
            if (captureBuffer != null) {
                captureBuffer.dispose();
                captureBuffer = null;
            }
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
