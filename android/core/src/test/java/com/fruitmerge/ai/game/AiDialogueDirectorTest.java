package com.fruitmerge.ai.game;

import org.junit.Test;

import java.util.EnumMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public final class AiDialogueDirectorTest {
    @Test
    public void strictCatalogRequiresOneThousandCompleteUniqueLinesPerMood() {
        EnumMap<AiDialogueDirector.Mood, String> texts =
                completeCatalog(1_000);
        AiDialogueDirector director =
                new AiDialogueDirector(texts, new Random(7L));

        for (AiDialogueDirector.Mood mood
                : AiDialogueDirector.Mood.values()) {
            assertEquals(1_000, director.lineCount(mood));
            Set<String> unique = new HashSet<>();
            for (int index = 0; index < director.lineCount(mood); index++) {
                String line = director.lineAt(mood, index);
                assertTrue(line.codePointCount(0, line.length()) >= 4);
                unique.add(line);
            }
            assertEquals(1_000, unique.size());
            if (mood.ordinal() > 0) {
                director.update(
                        AiDialogueDirector.HARD_SPEECH_GAP_SECONDS
                                + 0.01f
                );
            }
            assertNotNull(director.offer(
                    mood,
                    2f,
                    4,
                    true,
                    -1,
                    -1f
            ));
        }

        assertThrows(
                IllegalArgumentException.class,
                () -> new AiDialogueDirector(
                        completeCatalog(999),
                        new Random(7L)
                )
        );
    }

    @Test
    public void shuffleBagDoesNotRepeatBeforeEveryLineWasUsed() {
        AiDialogueDirector director = new AiDialogueDirector(
                completeCatalog(1_000),
                new Random(19L)
        );
        Set<String> firstCycle = new HashSet<>();

        for (int index = 0; index < 1_000; index++) {
            if (index > 0) {
                director.update(
                        AiDialogueDirector.HARD_SPEECH_GAP_SECONDS
                                + 0.01f
                );
            }
            AiDialogueDirector.Line line = director.offer(
                    AiDialogueDirector.Mood.HAPPY,
                    2f,
                    4,
                    true,
                    -1,
                    -1f
            );
            assertNotNull(line);
            assertTrue(firstCycle.add(line.text()));
        }
        assertEquals(1_000, firstCycle.size());
    }

    @Test
    public void urgentEventsCannotBreakTheHardSpeechGap() {
        AiDialogueDirector director =
                AiDialogueDirector.fallback(new Random(3L));
        AiDialogueDirector.Line first = director.offer(
                AiDialogueDirector.Mood.READY,
                2f,
                4,
                true,
                -1,
                -1f
        );
        assertNotNull(first);

        director.update(
                AiDialogueDirector.HARD_SPEECH_GAP_SECONDS - 0.01f
        );
        assertNull(director.offer(
                AiDialogueDirector.Mood.WORRIED,
                2f,
                7,
                false,
                -1,
                -1f
        ));
        assertNull(director.offer(
                AiDialogueDirector.Mood.WORRIED,
                2f,
                7,
                true,
                -1,
                -1f
        ));
        director.resetPacing();
        assertNull(director.offer(
                AiDialogueDirector.Mood.WELCOME,
                2f,
                4,
                true,
                -1,
                -1f
        ));

        director.update(2.9f);
        assertNotNull(director.offer(
                AiDialogueDirector.Mood.WORRIED,
                2f,
                7,
                false,
                -1,
                -1f
        ));
    }

    @Test
    public void activeMessageCannotBeReplacedBeforeMinimumHold() {
        AiDialogueDirector director =
                AiDialogueDirector.fallback(new Random(5L));
        assertNotNull(director.offer(
                AiDialogueDirector.Mood.THINKING,
                2f,
                1,
                true,
                -1,
                -1f
        ));
        director.update(5.1f);

        assertNull(director.offer(
                AiDialogueDirector.Mood.SURPRISED,
                2f,
                7,
                false,
                1,
                0.5f
        ));
        assertNotNull(director.offer(
                AiDialogueDirector.Mood.SURPRISED,
                2f,
                7,
                false,
                1,
                0.9f
        ));
    }

    @Test
    public void sameSeedProducesTheSameCompleteLineSequence() {
        EnumMap<AiDialogueDirector.Mood, String> texts =
                completeCatalog(1_000);
        AiDialogueDirector first =
                new AiDialogueDirector(texts, new Random(71L));
        AiDialogueDirector second =
                new AiDialogueDirector(texts, new Random(71L));

        for (int index = 0; index < 40; index++) {
            if (index > 0) {
                first.update(
                        AiDialogueDirector.HARD_SPEECH_GAP_SECONDS
                                + 0.01f
                );
                second.update(
                        AiDialogueDirector.HARD_SPEECH_GAP_SECONDS
                                + 0.01f
                );
            }
            AiDialogueDirector.Line firstLine = first.offer(
                    AiDialogueDirector.Mood.THINKING,
                    2f,
                    4,
                    true,
                    -1,
                    -1f
            );
            AiDialogueDirector.Line secondLine = second.offer(
                    AiDialogueDirector.Mood.THINKING,
                    2f,
                    4,
                    true,
                    -1,
                    -1f
            );
            assertEquals(firstLine.text(), secondLine.text());
            assertEquals(firstLine.emoticon(), secondLine.emoticon());
        }
    }

    @Test
    public void duplicateOrBlankLinesAreRejected() {
        EnumMap<AiDialogueDirector.Mood, String> duplicateCatalog =
                completeCatalog(1_000);
        duplicateCatalog.put(
                AiDialogueDirector.Mood.READY,
                "准备出发啦。\n准备出发啦。\n"
                        + catalogText(
                                AiDialogueDirector.Mood.READY,
                                998,
                                10_000
                        )
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new AiDialogueDirector(
                        duplicateCatalog,
                        new Random(2L)
                )
        );

        EnumMap<AiDialogueDirector.Mood, String> blankCatalog =
                completeCatalog(1_000);
        blankCatalog.put(
                AiDialogueDirector.Mood.HAPPY,
                "今天状态很好。\n\n" + catalogText(
                        AiDialogueDirector.Mood.HAPPY,
                        999,
                        20_000
                )
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new AiDialogueDirector(
                        blankCatalog,
                        new Random(2L)
                )
        );
    }

    private static EnumMap<AiDialogueDirector.Mood, String>
            completeCatalog(int count) {
        EnumMap<AiDialogueDirector.Mood, String> texts =
                new EnumMap<>(AiDialogueDirector.Mood.class);
        for (AiDialogueDirector.Mood mood
                : AiDialogueDirector.Mood.values()) {
            texts.put(mood, catalogText(mood, count, 0));
        }
        return texts;
    }

    private static String catalogText(
            AiDialogueDirector.Mood mood,
            int count,
            int offset) {
        StringBuilder text = new StringBuilder(count * 20);
        for (int index = 0; index < count; index++) {
            text.append("这是")
                    .append(mood.label())
                    .append("的完整语句")
                    .append(index + offset)
                    .append("。")
                    .append('\n');
        }
        return text.toString();
    }
}
