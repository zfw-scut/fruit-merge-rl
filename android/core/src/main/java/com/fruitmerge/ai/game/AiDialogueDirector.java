package com.fruitmerge.ai.game;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;

/**
 * AI 完整语料的随机抽取与发言节奏控制器。
 *
 * <p>语料文件中的每一行都是已经完整写好的独立句子。运行时只从完整句子中抽取，
 * 不拼接前缀、主体或结尾，也不会把任何文字随机与水果队列、AI 动作随机混用。</p>
 */
public final class AiDialogueDirector {
    public static final int MIN_LINES_PER_MOOD = 1_000;
    public static final int MAX_LINE_CODE_POINTS = 32;
    public static final float HARD_SPEECH_GAP_SECONDS = 2.2f;

    private static final float MIN_SILENCE_SECONDS = 3.5f;
    private static final float MAX_SILENCE_SECONDS = 7f;
    private static final float MIN_URGENT_GAP_SECONDS = 3.2f;
    private static final float MAX_URGENT_GAP_SECONDS = 5f;

    private final EnumMap<Mood, String[]> catalog =
            new EnumMap<>(Mood.class);
    private final EnumMap<Mood, ShuffleBag> bags =
            new EnumMap<>(Mood.class);
    private final EnumMap<Mood, Float> nextMoodSpeechAt =
            new EnumMap<>(Mood.class);
    private final Random random;

    private float elapsedSeconds;
    private float lastSpeechAt = -1_000f;
    private float nextSpeechAt;
    private float nextUrgentSpeechAt;

    /**
     * Creates a strict production catalog.
     *
     * @param texts one UTF-8 text blob per mood, one complete sentence per line
     * @throws IllegalArgumentException if a mood has fewer than 1000 valid unique lines
     */
    public AiDialogueDirector(
            Map<Mood, String> texts,
            Random random) {
        this(texts, random, true);
    }

    private AiDialogueDirector(
            Map<Mood, String> texts,
            Random random,
            boolean strict) {
        this.random = Objects.requireNonNull(random, "random");
        Objects.requireNonNull(texts, "texts");
        for (Mood mood : Mood.values()) {
            String[] lines = parseLines(texts.get(mood), mood, strict);
            catalog.put(mood, lines);
            bags.put(mood, new ShuffleBag(lines.length, random));
            nextMoodSpeechAt.put(mood, 0f);
        }
    }

    /**
     * Runtime safety fallback used only if packaged dialogue assets are damaged.
     */
    public static AiDialogueDirector fallback(Random random) {
        EnumMap<Mood, String> texts = new EnumMap<>(Mood.class);
        texts.put(Mood.THINKING, "让我认真看看这一步。\n我先观察一下局面。");
        texts.put(Mood.HESITATING, "两个位置都不错呢。\n这一步让我再想想。");
        texts.put(Mood.WELCOME, "接下来一起看看局面吧。\n我会认真陪你玩下去。");
        texts.put(Mood.READY, "准备好了，随时出发！\n我已经看准位置啦！");
        texts.put(Mood.HAPPY, "好耶，又顺利合成了！\n这一分拿得真漂亮！");
        texts.put(Mood.SURPRISED, "哇，这个连锁太精彩了！\n你的铺垫让我吃了一惊！");
        texts.put(Mood.WORRIED, "水果堆有点危险了。\n得赶紧给上面腾空间！");
        return new AiDialogueDirector(texts, random, false);
    }

    public void update(float deltaSeconds) {
        if (Float.isFinite(deltaSeconds) && deltaSeconds > 0f) {
            elapsedSeconds += deltaSeconds;
        }
    }

    /**
     * Clears only pacing gates when a new mode starts. Catalog shuffle state remains intact so
     * repeatedly entering a mode does not replay the same first sentence.
     */
    public void resetPacing() {
        nextSpeechAt = elapsedSeconds;
        nextUrgentSpeechAt = elapsedSeconds;
        for (Mood mood : Mood.values()) {
            nextMoodSpeechAt.put(mood, elapsedSeconds);
        }
    }

    /**
     * Attempts to show a new complete line.
     *
     * <p>Low-priority ambient events are sampled and discarded while cooling down; they are not
     * queued because a delayed "I am thinking" message would already be stale. Important events
     * can bypass the soft silence window but never the global hard gap.</p>
     */
    public Line offer(
            Mood mood,
            float requestedDuration,
            int priority,
            boolean force,
            int activePriority,
            float activeAgeSeconds) {
        Objects.requireNonNull(mood, "mood");
        int safePriority = Math.max(0, priority);
        boolean active = activeAgeSeconds >= 0f;

        if (elapsedSeconds - lastSpeechAt < HARD_SPEECH_GAP_SECONDS) {
            return null;
        }
        if (!force) {
            if (active) {
                if (activeAgeSeconds < 0.85f) {
                    return null;
                }
                if (safePriority <= activePriority + 1) {
                    return null;
                }
            }

            boolean urgent = safePriority >= 6;
            if (urgent) {
                if (elapsedSeconds < nextUrgentSpeechAt) {
                    return null;
                }
            } else {
                if (elapsedSeconds < nextSpeechAt
                        || elapsedSeconds < nextMoodSpeechAt.get(mood)) {
                    return null;
                }
                if (random.nextFloat() > ordinarySpeakProbability(safePriority)) {
                    return null;
                }
            }
        }

        float duration = clamp(
                requestedDuration
                        + randomRange(-0.12f, 0.24f),
                1.8f,
                2.8f
        );
        String[] lines = catalog.get(mood);
        String text = lines[bags.get(mood).next()];
        String[] emoticons = mood.emoticons();
        String emoticon = emoticons[random.nextInt(emoticons.length)];

        lastSpeechAt = elapsedSeconds;
        nextSpeechAt = elapsedSeconds
                + duration
                + randomRange(MIN_SILENCE_SECONDS, MAX_SILENCE_SECONDS);
        nextUrgentSpeechAt = elapsedSeconds
                + randomRange(MIN_URGENT_GAP_SECONDS, MAX_URGENT_GAP_SECONDS);
        nextMoodSpeechAt.put(
                mood,
                elapsedSeconds + moodCooldown(mood)
        );
        return new Line(text, emoticon, mood, duration, safePriority);
    }

    public int lineCount(Mood mood) {
        return catalog.get(Objects.requireNonNull(mood, "mood")).length;
    }

    public String lineAt(Mood mood, int index) {
        String[] lines = catalog.get(Objects.requireNonNull(mood, "mood"));
        if (index < 0 || index >= lines.length) {
            throw new IndexOutOfBoundsException("dialogue line index out of range");
        }
        return lines[index];
    }

    /**
     * Samples only a visual kaomoji. This does not consume a dialogue line or change any speech
     * gate, allowing the face to react immediately while text remains deliberately infrequent.
     */
    public String randomEmoticon(Mood mood) {
        String[] emoticons =
                Objects.requireNonNull(mood, "mood").emoticons();
        return emoticons[random.nextInt(emoticons.length)];
    }

    private float moodCooldown(Mood mood) {
        switch (mood) {
            case THINKING:
                return randomRange(12f, 18f);
            case HESITATING:
                return randomRange(10f, 15f);
            case WELCOME:
                return randomRange(10f, 14f);
            case READY:
                return randomRange(8f, 12f);
            case HAPPY:
                return randomRange(5f, 9f);
            case SURPRISED:
                return randomRange(4.5f, 7f);
            case WORRIED:
                return randomRange(8f, 14f);
            default:
                throw new IllegalStateException("unknown mood " + mood);
        }
    }

    private static float ordinarySpeakProbability(int priority) {
        if (priority <= 1) {
            return 0.15f;
        }
        if (priority == 2) {
            return 0.22f;
        }
        if (priority == 3) {
            return 0.38f;
        }
        if (priority == 4) {
            return 0.62f;
        }
        return 0.84f;
    }

    private float randomRange(float minimum, float maximum) {
        return minimum + (maximum - minimum) * random.nextFloat();
    }

    private static String[] parseLines(
            String text,
            Mood mood,
            boolean strict) {
        if (text == null) {
            throw new IllegalArgumentException(
                    "missing dialogue catalog for " + mood);
        }
        String normalized = text
                .replace("\r\n", "\n")
                .replace('\r', '\n');
        if (!normalized.isEmpty() && normalized.charAt(0) == '\uFEFF') {
            normalized = normalized.substring(1);
        }
        String[] rawLines = normalized.split("\n", -1);
        int end = rawLines.length;
        if (end > 0 && rawLines[end - 1].isEmpty()) {
            end -= 1;
        }

        List<String> parsed = new ArrayList<>(end);
        Set<String> unique = new HashSet<>(end * 2);
        for (int index = 0; index < end; index++) {
            String line = rawLines[index].trim();
            if (line.isEmpty()) {
                throw new IllegalArgumentException(
                        mood + " dialogue contains a blank line at "
                                + (index + 1));
            }
            int codePoints = line.codePointCount(0, line.length());
            if (codePoints < 4 || codePoints > MAX_LINE_CODE_POINTS) {
                throw new IllegalArgumentException(
                        mood + " dialogue line length is invalid at "
                                + (index + 1));
            }
            for (int offset = 0; offset < line.length(); offset++) {
                if (Character.isISOControl(line.charAt(offset))) {
                    throw new IllegalArgumentException(
                            mood + " dialogue contains a control character");
                }
            }
            if (!unique.add(line)) {
                throw new IllegalArgumentException(
                        mood + " dialogue contains a duplicate line: " + line);
            }
            parsed.add(line);
        }
        if (strict && parsed.size() < MIN_LINES_PER_MOOD) {
            throw new IllegalArgumentException(
                    mood + " dialogue requires at least "
                            + MIN_LINES_PER_MOOD + " complete lines");
        }
        if (parsed.isEmpty()) {
            throw new IllegalArgumentException(
                    mood + " dialogue must not be empty");
        }
        return parsed.toArray(new String[0]);
    }

    private static float clamp(float value, float minimum, float maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    public enum Mood {
        THINKING("思考中", new String[]{"(._.)?", "(-_-?)", "(o_o?)"}),
        HESITATING("有点纠结", new String[]{"(>_<)", "(._.;)", "(..;)"}),
        WELCOME("陪你开玩", new String[]{"(^_^)/", "(o_o)/", "(^_^)"}),
        READY("准备好了", new String[]{"(^_^)b", "(^o^)/", "(o_o)b"}),
        HAPPY("开心得分", new String[]{"(^o^)", "(^_^)", "(*^_^*)"}),
        SURPRISED("大吃一惊", new String[]{"(O_O)!", "(@_@)!", "(o_O)!"}),
        WORRIED("紧张起来", new String[]{"(;_;)", "(T_T)", "(>_<;)"}),
        ;

        private final String label;
        private final String[] emoticons;

        Mood(String label, String[] emoticons) {
            this.label = label;
            this.emoticons = emoticons;
        }

        public String label() {
            return label;
        }

        public String assetPath() {
            return "dialogue/"
                    + name().toLowerCase(java.util.Locale.ROOT)
                    + ".txt";
        }

        private String[] emoticons() {
            return emoticons;
        }
    }

    public static final class Line {
        private final String text;
        private final String emoticon;
        private final Mood mood;
        private final float duration;
        private final int priority;

        private Line(
                String text,
                String emoticon,
                Mood mood,
                float duration,
                int priority) {
            this.text = text;
            this.emoticon = emoticon;
            this.mood = mood;
            this.duration = duration;
            this.priority = priority;
        }

        public String text() {
            return text;
        }

        public String emoticon() {
            return emoticon;
        }

        public Mood mood() {
            return mood;
        }

        public float duration() {
            return duration;
        }

        public int priority() {
            return priority;
        }
    }

    private static final class ShuffleBag {
        private final int[] values;
        private final Random random;
        private int cursor;
        private int previous = -1;

        private ShuffleBag(int size, Random random) {
            values = new int[size];
            this.random = random;
            for (int index = 0; index < size; index++) {
                values[index] = index;
            }
            cursor = size;
        }

        private int next() {
            if (cursor >= values.length) {
                shuffle();
                cursor = 0;
            }
            int value = values[cursor++];
            previous = value;
            return value;
        }

        private void shuffle() {
            for (int index = values.length - 1; index > 0; index--) {
                int swap = random.nextInt(index + 1);
                int held = values[index];
                values[index] = values[swap];
                values[swap] = held;
            }
            if (values.length > 1 && values[0] == previous) {
                int swap = 1 + random.nextInt(values.length - 1);
                int held = values[0];
                values[0] = values[swap];
                values[swap] = held;
            }
        }
    }
}
