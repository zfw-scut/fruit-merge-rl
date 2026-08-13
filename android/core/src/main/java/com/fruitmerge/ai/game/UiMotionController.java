package com.fruitmerge.ai.game;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Small renderer-agnostic touch motion state machine for the hand-drawn UI.
 *
 * <p>It deliberately owns no libGDX drawing objects. Screens register a logical
 * control rectangle on touch-down, feed pointer movement, and query a visual
 * transform while drawing. A control therefore responds immediately to press,
 * follows the finger by a bounded amount while dragged, and springs back after
 * release without changing its actual hit rectangle.</p>
 */
public final class UiMotionController {
    private static final float MAX_FOLLOW_OFFSET = 8f;
    private static final float COMMIT_SLOP = 14f;

    private final Map<String, Motion> motions = new HashMap<>();

    private String activeId;
    private Bounds activeBounds;
    private int activePointer = -1;
    private float downX;
    private float downY;

    /** Starts one control gesture. A second pointer cannot steal it. */
    public boolean begin(
            String id,
            Bounds bounds,
            int pointer,
            float x,
            float y) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(bounds, "bounds");
        if (activePointer >= 0 || !bounds.contains(x, y, 0f)) {
            return false;
        }
        activeId = id;
        activeBounds = bounds;
        activePointer = pointer;
        downX = x;
        downY = y;
        Motion motion = motion(id);
        motion.pressed = true;
        motion.inside = true;
        motion.targetOffsetX = 0f;
        motion.targetOffsetY = 1.8f;
        return true;
    }

    /** Updates the bounded pseudo-physical follow offset for the active control. */
    public boolean drag(int pointer, float x, float y) {
        if (pointer != activePointer || activeId == null) {
            return false;
        }
        Motion motion = motion(activeId);
        motion.targetOffsetX = clamp(
                (x - downX) * 0.10f,
                -MAX_FOLLOW_OFFSET,
                MAX_FOLLOW_OFFSET
        );
        motion.targetOffsetY = clamp(
                1.8f + (y - downY) * 0.10f,
                -MAX_FOLLOW_OFFSET,
                MAX_FOLLOW_OFFSET
        );
        motion.inside = activeBounds.contains(x, y, COMMIT_SLOP);
        return true;
    }

    /**
     * Releases the gesture and returns the action id only when it still commits.
     */
    public String release(int pointer, float x, float y) {
        if (pointer != activePointer || activeId == null) {
            return null;
        }
        String releasedId = activeId;
        Motion motion = motion(releasedId);
        boolean commits = activeBounds.contains(x, y, COMMIT_SLOP);
        motion.pressed = false;
        motion.inside = commits;
        motion.targetOffsetX = 0f;
        motion.targetOffsetY = 0f;
        motion.releasePulse = commits ? 1f : 0.35f;
        clearActive();
        return commits ? releasedId : null;
    }

    /** Cancels the active pointer without triggering its action. */
    public boolean cancel(int pointer) {
        if (pointer != activePointer || activeId == null) {
            return false;
        }
        Motion motion = motion(activeId);
        motion.pressed = false;
        motion.inside = false;
        motion.targetOffsetX = 0f;
        motion.targetOffsetY = 0f;
        motion.releasePulse = 0.25f;
        clearActive();
        return true;
    }

    /** Cancels any active control, for example when changing screen. */
    public void cancelAll() {
        if (activeId != null) {
            cancel(activePointer);
        }
    }

    /** Advances press, follow, and release-spring animation using real time. */
    public void update(float deltaSeconds) {
        float delta = clamp(deltaSeconds, 0f, 0.1f);
        for (Motion motion : motions.values()) {
            float pressureTarget =
                    motion.pressed && motion.inside ? 1f : 0f;
            motion.pressure = approach(
                    motion.pressure,
                    pressureTarget,
                    22f,
                    delta
            );
            motion.offsetX = approach(
                    motion.offsetX,
                    motion.targetOffsetX,
                    motion.pressed ? 28f : 18f,
                    delta
            );
            motion.offsetY = approach(
                    motion.offsetY,
                    motion.targetOffsetY,
                    motion.pressed ? 28f : 18f,
                    delta
            );
            motion.releasePulse = Math.max(
                    0f,
                    motion.releasePulse - delta * 5.5f
            );
        }
    }

    /** Current visual transform for a control; querying creates its idle state. */
    public Visual visual(String id) {
        Motion motion = motion(id);
        float spring = motion.releasePulse <= 0f
                ? 0f
                : (float) Math.sin(
                        motion.releasePulse * Math.PI
                ) * motion.releasePulse;
        return new Visual(
                1f - motion.pressure * 0.035f + spring * 0.018f,
                motion.offsetX,
                motion.offsetY - spring * 1.5f,
                motion.pressure,
                motion.releasePulse
        );
    }

    public boolean hasActiveControl() {
        return activeId != null;
    }

    public int activePointer() {
        return activePointer;
    }

    public String activeId() {
        return activeId;
    }

    private Motion motion(String id) {
        return motions.computeIfAbsent(
                Objects.requireNonNull(id, "id"),
                ignored -> new Motion()
        );
    }

    private void clearActive() {
        activeId = null;
        activeBounds = null;
        activePointer = -1;
    }

    private static float approach(
            float value,
            float target,
            float response,
            float delta) {
        float blend = 1f - (float) Math.exp(-response * delta);
        return value + (target - value) * blend;
    }

    private static float clamp(float value, float minimum, float maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    public static final class Bounds {
        public final float left;
        public final float top;
        public final float width;
        public final float height;

        public Bounds(float left, float top, float width, float height) {
            if (!Float.isFinite(left)
                    || !Float.isFinite(top)
                    || !Float.isFinite(width)
                    || !Float.isFinite(height)
                    || width <= 0f
                    || height <= 0f) {
                throw new IllegalArgumentException(
                        "bounds must be finite and have positive size"
                );
            }
            this.left = left;
            this.top = top;
            this.width = width;
            this.height = height;
        }

        public boolean contains(float x, float y, float slop) {
            return x >= left - slop
                    && x <= left + width + slop
                    && y >= top - slop
                    && y <= top + height + slop;
        }
    }

    public static final class Visual {
        public final float scale;
        public final float offsetX;
        public final float offsetY;
        public final float pressure;
        public final float releasePulse;

        private Visual(
                float scale,
                float offsetX,
                float offsetY,
                float pressure,
                float releasePulse) {
            this.scale = scale;
            this.offsetX = offsetX;
            this.offsetY = offsetY;
            this.pressure = pressure;
            this.releasePulse = releasePulse;
        }
    }

    private static final class Motion {
        private float pressure;
        private float offsetX;
        private float offsetY;
        private float targetOffsetX;
        private float targetOffsetY;
        private float releasePulse;
        private boolean pressed;
        private boolean inside = true;
    }
}
