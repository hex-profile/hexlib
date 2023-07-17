import numpy as np
from freetype import *
import cv2

#================================================================
#
# rasterize_font
#
#================================================================

def rasterize_font(face, num_chars):

    rasterized_chars = []

    for char_code in range(num_chars):

        face.load_char(char_code, FT_LOAD_RENDER | FT_LOAD_TARGET_NORMAL)

        if not (face.glyph.advance.x % 64 == 0 and face.glyph.advance.y == 0):
            raise ValueError(f"Advance value for char code {char_code} is not an integer or has vertical offset")

        bitmap = face.glyph.bitmap

        buffer = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.width)

        rasterized_chars.append({
            "buffer": buffer,
            "start_x": face.glyph.bitmap_left,
            "start_y": -face.glyph.bitmap_top,
            "advance_x": face.glyph.advance.x // 64,
            "advance_y": 0
        })

    return rasterized_chars

#================================================================
#
# draw_text
#
# draw_text takes an image, text, x and y coordinates, font properties, and
# rasterized characters as input. It draws the text on the image at the
# specified coordinates and returns the modified image.
# The function iterates through the characters in the text, and for each
# character, it finds the corresponding rasterized glyph, its bitmap, starting
# x and y coordinates, and advance values.
# It calculates the position of the character on the image and clips it if
# necessary. It then blends the character bitmap with the image using the
# np.maximum function.
#
#================================================================

def draw_text(image, text, x, y, font_properties, rasterized_chars):
    current_x, current_y = x, y
    for char in text:
        char_index = ord(char)
        if char_index >= font_properties['num_chars']:
            continue

        glyph = rasterized_chars[char_index]
        bitmap = glyph['buffer']
        start_x = glyph['start_x']
        start_y = glyph['start_y']
        advance_x = glyph['advance_x']
        advance_y = glyph['advance_y']

        if bitmap.size > 0:
            x_pos = current_x + start_x
            y_pos = current_y + start_y
            x_max, y_max = x_pos + bitmap.shape[1], y_pos + bitmap.shape[0]

            x_min = max(0, x_pos)
            y_min = max(0, y_pos)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)

            src_x_min = x_min - x_pos
            src_y_min = y_min - y_pos
            src_x_max = src_x_min + (x_max - x_min)
            src_y_max = src_y_min + (y_max - y_min)

            if x_max > x_min and y_max > y_min:
                image[y_min:y_max, x_min:x_max] = np.maximum(image[y_min:y_max, x_min:x_max], bitmap[src_y_min:src_y_max, src_x_min:src_x_max])

        current_x += advance_x
        current_y += advance_y

#================================================================
#
# calculate_font_properties
#
#================================================================

def calculate_font_properties(rasterized_chars):
    max_bitmap_size_x = 0
    max_bitmap_size_y = 0

    max_bitmap_left = 0
    max_bitmap_right = 0
    max_bitmap_top = 0
    max_bitmap_bottom = 0

    for char in rasterized_chars:
        size_x = char['buffer'].shape[1]
        size_y = char['buffer'].shape[0]

        max_bitmap_size_x = max(max_bitmap_size_x, size_x)
        max_bitmap_size_y = max(max_bitmap_size_y, size_y)

        start_x = char['start_x']
        start_y = char['start_y']

        max_bitmap_left = max(max_bitmap_left, -start_x)
        max_bitmap_right = max(max_bitmap_right, start_x + size_x)

        max_bitmap_top = max(max_bitmap_top, -start_y)
        max_bitmap_bottom = max(max_bitmap_bottom, start_y + size_y)

    font_properties = {
        "num_chars": len(rasterized_chars),
        "max_bitmap_left": max_bitmap_left,
        "max_bitmap_right": max_bitmap_right,
        "max_bitmap_top": max_bitmap_top,
        "max_bitmap_bottom": max_bitmap_bottom,
        "max_bitmap_size_x": max_bitmap_size_x,
        "max_bitmap_size_y": max_bitmap_size_y
    }

    return font_properties

#================================================================
#
# rasterize_font_outline
#
#================================================================

def rasterize_font_outline(face, num_chars, thickness):

    stroker = Stroker( )
    stroker.set(round(thickness * 64), FT_STROKER_LINECAP_ROUND, FT_STROKER_LINEJOIN_ROUND, 0 )

    ###

    rasterized_chars = []

    for char_code in range(num_chars):

        face.load_char(char_code, FT_LOAD_NO_BITMAP)

        if not (face.glyph.advance.x % 64 == 0 and face.glyph.advance.y == 0):
            raise ValueError(f"Advance value for char code {char_code} is not an integer or has vertical offset")

        ###

        slot = face.glyph
        glyph = slot.get_glyph()

        glyph.stroke(stroker, True)
        #error = FT_Glyph_StrokeBorder(byref(glyph._FT_Glyph), stroker._FT_Stroker, False, True)
        #if error: raise FT_Exception(error)

        blyph = glyph.to_bitmap(FT_RENDER_MODE_NORMAL, None, True)
        bitmap = blyph.bitmap

        buffer = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.width)

        rasterized_chars.append({
            "buffer": buffer,
            "start_x": face.glyph.bitmap_left,
            "start_y": -face.glyph.bitmap_top
        })

    return rasterized_chars

#================================================================
#
# main
#
#================================================================

if __name__ == "__main__":

    font_file = "fonts/OpenSans-Regular.ttf"
    font_size = 256
    num_chars = 128
    outline_thikness = 6.0
    text = "The quick brown fox jumps over the lazy dog"

    ###

    face = Face(font_file)
    face.set_char_size(round(font_size * 64))

    ###

    rasterized_chars = rasterize_font(face, num_chars)
    font_properties = calculate_font_properties(rasterized_chars)

    rasterized_outline = rasterize_font_outline(face, num_chars, outline_thikness)

    for i in range(num_chars):
        rasterized_outline[i]['advance_x'] = rasterized_chars[i]['advance_x']
        rasterized_outline[i]['advance_y'] = rasterized_chars[i]['advance_y']

    ###

    #print("Font properties:")
    #for key, value in font_properties.items():
    #    print(f"{key}: {value}")

    ###

    width, height = 1920, 1080

    ###

    x, y = font_properties['max_bitmap_left'], font_properties['max_bitmap_top']

    back_image = np.full((height, width), 0, dtype=np.uint8)
    draw_text(back_image, text, x, y, font_properties, rasterized_outline)

    fore_image = np.full((height, width), 0, dtype=np.uint8)
    draw_text(fore_image, text, x, y, font_properties, rasterized_chars)

    ###

    # y += (font_properties['max_bitmap_top'] + font_properties['max_bitmap_bottom'])
    # draw_text(image, text, x, y, font_properties, rasterized_chars)

    cv2.imwrite('fore_image.png', 255 - fore_image)
    cv2.imwrite('back_image.png', 255 - back_image)
