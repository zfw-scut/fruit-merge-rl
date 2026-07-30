"""Generate the shared high-resolution Chinese libGDX UI bitmap font.

The Android and LWJGL3 preview renderers deliberately load the same `.fnt` and
PNG files.  This keeps font metrics deterministic across both platforms and
avoids enlarging libGDX's tiny built-in bitmap font on high-density phones.
Only the glyphs used by the mobile UI are packed, so a playful CJK font does
not turn into a multi-megabyte runtime atlas.
"""

from __future__ import annotations

import argparse
import io
import math
from dataclasses import dataclass
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "assets"
    / "fonts"
    / "source"
    / "ZCOOLKuaiLe-Regular.ttf"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "fonts"
JAVA_SOURCE_ROOTS = (
    PROJECT_ROOT / "android" / "core" / "src" / "main" / "java",
    PROJECT_ROOT / "android" / "app" / "src" / "main" / "java",
)
DIALOGUE_SOURCE_ROOT = PROJECT_ROOT / "assets" / "dialogue"
ASCII_CHARACTERS = "".join(chr(codepoint) for codepoint in range(32, 127))
CHINESE_UI_CHARACTERS = (
    "合成大西瓜分数最高下一颗陪玩开启关闭启动中模型就绪拖动"
    "水果松手投放点击重新开始等待稳定正在加载观察局面安全策略"
    "暂不可用考虑位置尝试其他微调决定完成手动思考试探已结束"
    "危险线游戏本局得分未知节点变化无式，·："
)


def java_string_literal_characters(source_roots: tuple[Path, ...]) -> str:
    """Collect non-ASCII glyphs from Java string literals, excluding comments.

    The mobile UI changes much faster than this small hand-curated seed list.
    Discovering literals from both Android modules makes font generation fail
    early when a newly added Chinese label is not covered by the source font,
    instead of shipping a square replacement glyph in the APK.
    """

    characters: dict[str, None] = {}
    for source_root in source_roots:
        if not source_root.is_dir():
            continue
        for source_path in sorted(source_root.rglob("*.java")):
            text = source_path.read_text(encoding="utf-8")
            index = 0
            state = "code"
            while index < len(text):
                character = text[index]
                following = text[index + 1] if index + 1 < len(text) else ""

                if state == "code":
                    if character == "/" and following == "/":
                        state = "line_comment"
                        index += 2
                        continue
                    if character == "/" and following == "*":
                        state = "block_comment"
                        index += 2
                        continue
                    if character == '"':
                        state = "string"
                        index += 1
                        continue
                    if character == "'":
                        state = "character"
                        index += 1
                        continue
                elif state == "line_comment":
                    if character in "\r\n":
                        state = "code"
                elif state == "block_comment":
                    if character == "*" and following == "/":
                        state = "code"
                        index += 2
                        continue
                elif state in {"string", "character"}:
                    terminator = '"' if state == "string" else "'"
                    if character == "\\":
                        if following == "u":
                            end = index + 2
                            while end < len(text) and text[end] == "u":
                                end += 1
                            digits = text[end : end + 4]
                            if len(digits) == 4:
                                try:
                                    decoded = chr(int(digits, 16))
                                except ValueError:
                                    decoded = ""
                                if state == "string" and ord(decoded or "\0") > 127:
                                    characters.setdefault(decoded, None)
                                index = end + 4
                                continue
                        index += 2
                        continue
                    if character == terminator:
                        state = "code"
                    elif state == "string" and ord(character) > 127:
                        characters.setdefault(character, None)

                index += 1
    return "".join(characters)


JAVA_UI_CHARACTERS = java_string_literal_characters(JAVA_SOURCE_ROOTS)


def dialogue_characters(source_root: Path) -> str:
    """Collect glyphs from complete authored AI dialogue assets."""

    characters: dict[str, None] = {}
    if not source_root.is_dir():
        return ""
    for source_path in sorted(source_root.glob("*.txt")):
        for character in source_path.read_text(encoding="utf-8-sig"):
            if character not in "\r\n":
                characters.setdefault(character, None)
    return "".join(characters)


DIALOGUE_UI_CHARACTERS = dialogue_characters(DIALOGUE_SOURCE_ROOT)
DEFAULT_CHARACTERS = "".join(
    dict.fromkeys(
        ASCII_CHARACTERS
        + CHINESE_UI_CHARACTERS
        + JAVA_UI_CHARACTERS
        + DIALOGUE_UI_CHARACTERS
    )
)


@dataclass(frozen=True)
class Glyph:
    """One AngelCode BMFont character record plus its atlas placement."""

    character: str
    x: int
    y: int
    width: int
    height: int
    x_offset: int
    y_offset: int
    x_advance: int
    left: int
    top: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font-size", type=int, default=64)
    parser.add_argument("--weight", type=float, default=750.0)
    # The complete dialogue corpus needs 1580+ glyphs. A 1024-wide atlas grows to 8192px high,
    # which exceeds GL_MAX_TEXTURE_SIZE on some supported Android devices; 2048x4096 stays safe.
    parser.add_argument("--atlas-width", type=int, default=2048)
    parser.add_argument("--output-name", default="ui-cute")
    parser.add_argument("--face-name", default="ZCOOL KuaiLe UI")
    return parser.parse_args()


def load_font(source: Path, font_size: int, weight: float) -> tuple[ImageFont.FreeTypeFont, io.BytesIO]:
    """Load a static font or instantiate a requested variable-font weight."""

    source_font = TTFont(source)
    if "fvar" in source_font:
        rendered_font = instantiateVariableFont(
            source_font,
            {"wght": weight},
            inplace=False,
        )
    else:
        rendered_font = source_font
    buffer = io.BytesIO()
    rendered_font.save(buffer)
    buffer.seek(0)
    return ImageFont.truetype(buffer, font_size), buffer


def validate_character_coverage(source: Path, characters: str) -> None:
    """Fail before atlas generation if a requested UI glyph is absent."""

    cmap = TTFont(source).getBestCmap() or {}
    missing = [character for character in characters if ord(character) not in cmap]
    if missing:
        escaped = " ".join(
            f"{character!r}(U+{ord(character):04X})"
            for character in missing
        )
        raise ValueError(f"font is missing requested UI glyphs: {escaped}")


def next_power_of_two(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def pack_glyphs(
    font: ImageFont.FreeTypeFont,
    characters: str,
    atlas_width: int,
    ascent: int,
) -> tuple[list[Glyph], int]:
    """Pack glyph rectangles row by row with enough filtering padding."""

    padding = 4
    cursor_x = padding
    cursor_y = padding
    row_height = 0
    glyphs: list[Glyph] = []

    for character in characters:
        left, top, right, bottom = font.getbbox(character, anchor="ls")
        width = max(0, math.ceil(right - left))
        height = max(0, math.ceil(bottom - top))
        if width > 0 and cursor_x + width + padding > atlas_width:
            cursor_x = padding
            cursor_y += row_height + padding
            row_height = 0

        glyph_x = cursor_x if width > 0 else 0
        glyph_y = cursor_y if height > 0 else 0
        glyphs.append(
            Glyph(
                character=character,
                x=glyph_x,
                y=glyph_y,
                width=width,
                height=height,
                x_offset=math.floor(left),
                y_offset=math.floor(ascent + top),
                x_advance=round(font.getlength(character)),
                left=math.floor(left),
                top=math.floor(top),
            )
        )
        if width > 0:
            cursor_x += width + padding
            row_height = max(row_height, height)

    return glyphs, next_power_of_two(cursor_y + row_height + padding)


def render_atlas(
    font: ImageFont.FreeTypeFont,
    glyphs: list[Glyph],
    atlas_width: int,
    atlas_height: int,
) -> Image.Image:
    atlas = Image.new("RGBA", (atlas_width, atlas_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(atlas)
    for glyph in glyphs:
        if glyph.width == 0 or glyph.height == 0:
            continue
        # `ls` anchors at the left-side baseline. Offsetting by the recorded
        # bounding box makes the visible pixels start exactly at x/y.
        draw.text(
            (glyph.x - glyph.left, glyph.y - glyph.top),
            glyph.character,
            font=font,
            fill=(255, 255, 255, 255),
            anchor="ls",
        )
    return atlas


def kerning_pairs(
    font: ImageFont.FreeTypeFont,
    characters: str,
) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    advances = {character: font.getlength(character) for character in characters}
    for first in characters:
        for second in characters:
            amount = round(
                font.getlength(first + second)
                - advances[first]
                - advances[second]
            )
            if amount != 0:
                pairs.append((ord(first), ord(second), amount))
    return pairs


def write_fnt(
    output: Path,
    page_filename: str,
    face_name: str,
    glyphs: list[Glyph],
    kernings: list[tuple[int, int, int]],
    font_size: int,
    ascent: int,
    descent: int,
    atlas_width: int,
    atlas_height: int,
) -> None:
    lines = [
        (
            'info face="{face}" size={size} bold=0 italic=0 charset="" '
            "unicode=1 stretchH=100 smooth=1 aa=1 padding=0,0,0,0 "
            "spacing=1,1"
        ).format(face=face_name, size=font_size),
        (
            "common lineHeight={line_height} base={base} "
            "scaleW={width} scaleH={height} pages=1 packed=0"
        ).format(
            line_height=ascent + descent,
            base=ascent,
            width=atlas_width,
            height=atlas_height,
        ),
        f'page id=0 file="{page_filename}"',
        f"chars count={len(glyphs)}",
    ]
    for glyph in glyphs:
        lines.append(
            "char id={id} x={x} y={y} width={width} height={height} "
            "xoffset={x_offset} yoffset={y_offset} xadvance={x_advance} "
            "page=0 chnl=15".format(
                id=ord(glyph.character),
                x=glyph.x,
                y=glyph.y,
                width=glyph.width,
                height=glyph.height,
                x_offset=glyph.x_offset,
                y_offset=glyph.y_offset,
                x_advance=glyph.x_advance,
            )
        )
    lines.append(f"kernings count={len(kernings)}")
    for first, second, amount in kernings:
        lines.append(
            f"kerning first={first} second={second} amount={amount}"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"font source not found: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_character_coverage(source, DEFAULT_CHARACTERS)

    font, font_buffer = load_font(source, args.font_size, args.weight)
    # Keep the BytesIO alive while Pillow reads glyph tables lazily.
    _ = font_buffer
    ascent, descent = font.getmetrics()
    glyphs, atlas_height = pack_glyphs(
        font,
        DEFAULT_CHARACTERS,
        args.atlas_width,
        ascent,
    )
    atlas = render_atlas(font, glyphs, args.atlas_width, atlas_height)
    atlas_path = output_dir / f"{args.output_name}.png"
    fnt_path = output_dir / f"{args.output_name}.fnt"
    atlas.save(atlas_path, optimize=True)
    write_fnt(
        fnt_path,
        atlas_path.name,
        args.face_name,
        glyphs,
        kerning_pairs(font, DEFAULT_CHARACTERS),
        args.font_size,
        ascent,
        descent,
        args.atlas_width,
        atlas_height,
    )
    print(f"FONT_ATLAS={atlas_path}")
    print(f"FONT_DESCRIPTOR={fnt_path}")
    print(f"ATLAS_SIZE={args.atlas_width}x{atlas_height}")
    print(f"GLYPH_COUNT={len(glyphs)}")


if __name__ == "__main__":
    main()
