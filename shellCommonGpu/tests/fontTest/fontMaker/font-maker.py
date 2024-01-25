from freetype import *
import numpy as np
import cv2

#================================================================
#
# ensure
#
#================================================================

def ensure(condition, exception = AssertionError):
    if not np.all(condition):
        raise exception

#================================================================
#
# impose
#
#================================================================

def impose(image, pattern, pos):
    y, x = pos
    size_y, size_x = pattern.shape
    image_size_y, image_size_x = image.shape

    ensure(x >= 0 and x + size_x <= image_size_x)
    ensure(y >= 0 and y + size_y <= image_size_y)

    image[y:y+size_y, x:x+size_x] = np.maximum(image[y:y+size_y, x:x+size_x], pattern)

#================================================================
#
# get_slice
# set_slice
#
#================================================================

def get_slice(image, org, end):
    return image[org[0]:end[0], org[1]:end[1]]

def set_slice(image, org, end, value):
    image[org[0]:end[0], org[1]:end[1]] = value

#================================================================
#
# impose_clip
#
#================================================================

def impose_clip(image, pattern, pos):

    image_size = image.shape
    zero = np.zeros_like(image_size)

    org = pos
    end = pos + pattern.shape

    ###

    org = np.clip(org, zero, image_size)
    end = np.clip(end, zero, image_size)

    if not np.all(org < end):
        return

    ###

    pattern_org = org - pos
    pattern_end = end - pos

    pattern_part = pattern[pattern_org[0]:pattern_end[0], pattern_org[1]:pattern_end[1]]

    ###

    image[org[0]:end[0], org[1]:end[1]] = np.maximum(image[org[0]:end[0], org[1]:end[1]], pattern_part)

#================================================================
#
# convert_point_to_freetype_size
#
#================================================================

def convert_point_to_freetype_size(point_size, dpi, scaling_factor):
    inches = point_size / 72

    # Account for system scaling
    scaled_inches = inches * scaling_factor

    # Convert inches to pixels
    pixels = scaled_inches * dpi

    # Convert pixels to FreeType units
    freetype_size = round(pixels * 64)

    return freetype_size

#================================================================
#
# shift_image
#
#================================================================

def shift_image(image, dx, dy):
    size_y, size_x = image.shape

    shifted_image = np.zeros_like(image)

    shifted_image[max(dy, 0):min(size_y + dy, size_y),
                  max(dx, 0):min(size_x + dx, size_x)] = \
        image[max(-dy, 0):min(size_y - dy, size_y),
              max(-dx, 0):min(size_x - dx, size_x)]

    return shifted_image

#================================================================
#
# convert_to_u8
#
#================================================================

def convert_to_u8(image):
    return (np.clip(image, 0, 1) * 0xFF + 0.5).astype('uint8')

#================================================================
#
# upscale_image
#
#================================================================

def upscale_image(image, upscale):
    result = np.repeat(image, upscale, axis=0)
    result = np.repeat(result, upscale, axis=1)
    return result

#================================================================
#
# generate_c_font
#
#================================================================

def generate_c_font(font_name, font_header, font_letters, indent_size=4, bytes_per_line=16):

    def format_array(arr, line_prefix):
        return f"{line_prefix}{{{arr[1]}, {arr[0]}}}"

    struct_name = font_name[0].upper() + font_name[1:]
    indent = ' ' * indent_size

    # Calculate offsets and create the combined buffer
    offset = 0
    combined_buffer = np.array([], dtype=np.uint8)
    for letter in font_letters:
        letter['offset'] = offset
        offset += len(letter['buffer'])
        combined_buffer = np.append(combined_buffer, letter['buffer'])

    # Generating constants for the lengths of arrays
    letters_length = len(font_letters)
    buffer_length = len(combined_buffer)

    s = f"#include \"{font_name}.h\"\n\nusing namespace fontTypes;\n\n"
    s += "//================================================================\n"
    s += f"//\n"
    s += f"// {struct_name}\n"
    s += f"//\n"
    s += "//================================================================\n\n"

    # Defining constants
    s += f"static constexpr int32 {font_name}Letters = {letters_length};\n"
    s += f"static constexpr int32 {font_name}BufferSize = {buffer_length};\n\n"

    s += '////\n\n'

    # Generate the font structure
    s += f"struct {struct_name}\n{{\n"
    s += f"{indent}FontHeader header;\n"
    s += f"{indent}FontLetter letters[{font_name}Letters];\n"
    s += f"{indent}uint8 buffer[{font_name}BufferSize];\n}};\n\n"
    s += f"COMPILE_ASSERT(alignof({struct_name}) == fontTypesAligment);\n\n////\n\n"

    # Add font data
    s += f"static const {struct_name} {font_name}Data =\n{{\n"
    s += f"{indent}// header\n{indent}{{\n"
    s += f"{indent * 2}{font_header['rangeOrg']}, // rangeOrg\n"
    s += f"{indent * 2}{font_header['rangeEnd']}, // rangeEnd\n"
    s += f"{indent * 2}{font_header['unsupportedCode']} // unsupportedCode\n"

    s += f"{indent}}},\n{indent}// letters\n{indent}{{\n"

    for letter in font_letters:
        s += f"{indent * 2}{{\n"
        s += format_array(letter['advance'], indent * 3) + ", // advance\n"
        s += format_array(letter['org'], indent * 3) + ", // org\n"
        s += format_array(letter['size'], indent * 3) + ", // size\n"
        s += f"{indent * 3}{letter['offset']} // offset\n"
        s += f"{indent * 2}}},\n"
    s = s.rstrip(',\n') + "\n"

    s += f"{indent}}},\n{indent}// buffer\n{indent}{{\n"

    # Format the combined buffer with specified bytes per line
    for i in range(0, len(combined_buffer), bytes_per_line):
        line = ', '.join(f"0x{byte:02X}" for byte in combined_buffer[i:i+bytes_per_line])
        s += f"{indent * 2}{line},\n"
    s = s.rstrip(',\n') + "\n"

    s += f"{indent}}}\n}};\n\n////\n\n"
    s += f"const FontHeader& {font_name}()\n{{\n"
    s += f"{indent}return {font_name}Data.header;\n}}\n"

    # Writing to a file
    file_name = f"{font_name}.c"
    with open(file_name, "w") as file:
        file.write(s)

#================================================================
#
# main
#
#================================================================

if __name__ == "__main__":

    try:
        os.remove('main_image.png')
        os.remove('outline_image.png')
        os.remove('result_image.png')
    except FileNotFoundError:
        pass

    ###

    font_file = 'fonts/Arial Unicode MS.TTF'
    font_file = "fonts/DroidSans.ttf"
    font_file = "fonts/DroidSansMono.ttf"
    font_file = 'fonts/DancingScript-Regular.ttf'
    font_file = "fonts/OpenSans-Regular.ttf"

    font_pts = 16

    font_size = convert_point_to_freetype_size(font_pts, 96, 1.25)
    outline_thickness = convert_point_to_freetype_size(0.75, 96, 1.25)

    ###

    letter_region_defined_only_by_main_part = False

    unified_buffer_on = True

    ###

    test_image_file = 'test.bmp'
    test_image_color = None
    # test_image_color = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)

    ###

    upscale = 2

    #----------------------------------------------------------------
    #
    # Data
    #
    #----------------------------------------------------------------

    class Data:
        outline_buffer = None
        outline_anchor = None
        advance = None
        main_buffer = None
        main_anchor = None
        org = None
        end = None
        unified_buffer = None

    #----------------------------------------------------------------
    #
    # Build all the letters of the font.
    #
    #----------------------------------------------------------------

    max_int = np.iinfo(np.int32).max
    min_int = np.iinfo(np.int32).min

    ###

    face = Face(font_file)
    face.set_char_size(font_size)

    stroker = Stroker()
    stroker.set(outline_thickness, FT_STROKER_LINECAP_ROUND, FT_STROKER_LINEJOIN_ROUND, 0)

    ###

    font_data = {}

    font_range_org = 32
    font_range_end = 128

    font_range = [chr(i) for i in range(font_range_org, font_range_end)]

    ###

    for char in font_range:

        data = Data()

        #----------------------------------------------------------------
        #
        # Outline
        #
        #----------------------------------------------------------------

        face.load_char(char, FT_LOAD_DEFAULT | FT_LOAD_NO_BITMAP)
        glyph = face.glyph.get_glyph()
        glyph.stroke(stroker, True)
        blyph = glyph.to_bitmap(FT_RENDER_MODE_NORMAL, Vector(0, 0), True)
        bitmap = blyph.bitmap

        ###

        outline_pitch = bitmap.pitch
        outline_size = np.array([bitmap.rows, bitmap.width])
        ensure(outline_pitch == outline_size[1])
        ensure(outline_size[0] * outline_size[1] == len(bitmap.buffer))
        ensure(face.glyph.advance.x % 64 == 0 and face.glyph.advance.y == 0)
        outline_anchor = np.array([-blyph.top, blyph.left])

        ###

        outline_buffer = np.array(bitmap.buffer, dtype=np.uint8).reshape(outline_size)
        outline_buffer = outline_buffer.astype(np.float32) / 0xFF

        ###

        data.outline_buffer = outline_buffer
        data.outline_anchor = outline_anchor + 0

        #----------------------------------------------------------------
        #
        # Main.
        #
        #----------------------------------------------------------------

        face.load_char(char, FT_LOAD_RENDER | FT_LOAD_TARGET_NORMAL)
        glyph = face.glyph
        bitmap = glyph.bitmap

        ###

        main_pitch = bitmap.pitch
        main_size = np.array([bitmap.rows, bitmap.width])
        ensure(main_pitch == main_size[1])
        ensure(main_size[0] * main_size[1] == len(bitmap.buffer))
        main_anchor = np.array([-glyph.bitmap_top, glyph.bitmap_left])

        ###

        ensure(glyph.advance.x % 64 == 0 and glyph.advance.y == 0)
        advance = np.array([face.glyph.advance.y // 64, face.glyph.advance.x // 64])
        data.advance = advance + 0

        ###

        main_buffer = np.array(bitmap.buffer, dtype=np.uint8).reshape(main_size)
        main_buffer = main_buffer.astype(np.float32) / 0xFF

        ###

        data.main_buffer = main_buffer
        data.main_anchor = main_anchor + 0

        ###

        if letter_region_defined_only_by_main_part:
            data.org = main_anchor + 0
            data.end = main_anchor + main_size
        else:
            data.org = np.minimum(outline_anchor, main_anchor)
            data.end = np.maximum(outline_anchor + outline_size, main_anchor + main_size)

        ###

        if unified_buffer_on:
            common_org = np.minimum(outline_anchor, main_anchor)
            common_end = np.maximum(outline_anchor + outline_size, main_anchor + main_size)
            common_size = common_end - common_org

            main_anchor -= common_org
            outline_anchor -= common_org

            main = np.zeros(common_size, dtype=np.float32)
            set_slice(main, main_anchor, main_anchor + main_size, main_buffer)

            outline = np.zeros(common_size, dtype=np.float32)
            set_slice(outline, outline_anchor, outline_anchor + outline_size, outline_buffer)

            ###

            opacity = main + (1 - main) * outline

            unified_bit = (opacity >= 0.9)
            unified_value = np.where(unified_bit, main, outline)

            #
            # Passing through the bottleneck of font storage.
            #

            unified_value = np.clip(unified_value, 0, 1)

            if True: # Test fixed point storage

                fixed_value = np.floor(unified_value * 0x7F + 0.5).astype(np.uint8)
                fixed_value |= (unified_bit.astype(np.uint8) << 7)

                ###

                unified_bit = (fixed_value & 0x80) != 0
                unified_value = (fixed_value & 0x7F).astype(np.float32) / 0x7F

            ###

            main = np.where(unified_bit, unified_value, 0)
            outline = np.where(unified_bit, 1, unified_value)

            ###

            data.org = common_org + 0
            data.end = common_end + 0

            data.main_anchor = common_org + 0
            data.main_buffer = main

            data.outline_anchor = common_org + 0
            data.outline_buffer = outline

            data.unified_buffer = fixed_value

        ###

        font_data[char] = data

    #----------------------------------------------------------------
    #
    # Font row height.
    #
    #----------------------------------------------------------------

    min_font_org = np.array([max_int, max_int])
    max_font_end = np.array([min_int, min_int])

    for char in font_range:
        data = font_data[char]
        min_font_org = np.minimum(min_font_org, data.org)
        max_font_end = np.maximum(max_font_end, data.end)

    ensure(min_font_org <= max_font_end)

    ###

    font_row_height = max_font_end[0] - min_font_org[0]

    #----------------------------------------------------------------
    #
    # Making `org` non-negative everywhere.
    #
    #----------------------------------------------------------------

    correction = -min_font_org

    for char in font_range:

        data = font_data[char]

        data.org += correction
        data.end += correction

        data.main_anchor += correction
        data.outline_anchor += correction

        ensure(data.org >= 0)

    ###

    min_font_org += correction
    max_font_end += correction

    ###

    ensure(min_font_org == 0)

    del min_font_org
    del max_font_end

    #----------------------------------------------------------------
    #
    # Text.
    #
    #----------------------------------------------------------------

    text = [
        'A quick brown fox jumps over the lazy dog',
        'Breathtaking views at every turn',
        'Cascades of light illuminate the night',
        'Dreams unfold in the whisper of the wind',
        'Eloquence is the art of speaking well',
        "3 times the charm, in life's funny dance.",
        "!Exciting adventures, just around the corner.",
        "mYsteries unfold in the pages of history.",
        "#Trending topics sweep the digital world.",
        "Quiet quills pen tales of yore.",
        "wHispers of wisdom, hidden in silence.",
        "7 wonders of the world, awaiting discovery.",
        "eNigmatic riddles, cloaked in shadow.",
        "@Signs of the times, in every tweet.",
        "Daring dreams, boldly chased.",
        "xMarks the spot of hidden treasures.",
        "%Percentages and probabilities, in life's equation.",
        "Jubilant journeys, filled with laughter.",
        "2 paths diverge in a yellow wood."
    ]

    ###

    k = 0
    count = len(font_range)

    while k < count:
        avail = count - k
        packet = min(avail, 64)
        s = ''.join(font_range[k: k + packet])
        text.append(s)
        k += packet

    #----------------------------------------------------------------
    #
    # Compute text dimensions.
    #
    # Aligning the left edge not according to the text, but according to the entire font.
    #
    #----------------------------------------------------------------

    row_count = len(text)

    max_text_end = np.array([min_int, min_int])

    for s in text:

        pos = np.array([0, 0])

        for char in s:
            data = font_data[char]
            ensure(data.org >= 0)
            max_text_end = np.maximum(max_text_end, pos + data.end)
            pos += data.advance

    max_text_size = max_text_end

    #----------------------------------------------------------------
    #
    # Draw.
    #
    #----------------------------------------------------------------

    image_size = np.array([row_count * font_row_height, max_text_size[1]])

    outline_image = np.full(image_size, 0, dtype=np.float32)
    main_image = np.full(image_size, 0, dtype=np.float32)

    ###

    for row in range(row_count):

        s = text[row]

        pos = np.array([row * font_row_height, 0])

        for char in s:

            data = font_data[char]

            impose_clip(outline_image, data.outline_buffer, pos + data.outline_anchor)
            impose_clip(main_image, data.main_buffer, pos + data.main_anchor)

            pos += data.advance

    #----------------------------------------------------------------
    #
    # Experimental outline
    #
    #----------------------------------------------------------------

    experiment = False

    if experiment:

        filter_data = [0.01519963, 0.21875173, 0.53209728, 0.21875173, 0.01519963]
        filter_ofs = [-2, -1, 0, +1, +2]

        ###

        blur = np.zeros_like(main_image)

        for (dx, fx) in zip(filter_ofs, filter_data):
            for (dy, fy) in zip(filter_ofs, filter_data):
                blur += fx * fy * shift_image(main_image, dx, dy)

        ###

        diff = main_image - blur

        outline_image = np.where(diff < 0, np.clip(-10 * diff, 0, 1), np.zeros_like(main_image))

        ###

        outline_blur = np.zeros_like(main_image)

        for (dx, fx) in zip(filter_ofs, filter_data):
            for (dy, fy) in zip(filter_ofs, filter_data):
                outline_blur += fx * fy * shift_image(outline_image, dx, dy)

    #----------------------------------------------------------------
    #
    # Save.
    #
    #----------------------------------------------------------------

    outline_image_up = np.repeat(outline_image, upscale, axis=0)
    outline_image_up = np.repeat(outline_image_up, upscale, axis=1)
    cv2.imwrite('outline_image.png', convert_to_u8(1 - outline_image_up))

    main_image_up = np.repeat(main_image, upscale, axis=0)
    main_image_up = np.repeat(main_image_up, upscale, axis=1)
    cv2.imwrite('main_image.png', convert_to_u8(1 - main_image_up))

    #----------------------------------------------------------------
    #
    # Overlay on real image.
    #
    #----------------------------------------------------------------

    test_image = cv2.imread(test_image_file).astype(np.float32) / 0xFF

    if test_image_color is not None:
        test_image = test_image * 0 + test_image_color

    ###

    test_size = test_image.shape
    common_size = np.minimum(image_size, test_size[:2])

    ###

    test_area = test_image[0:common_size[0], 0:common_size[1]]
    main_area = main_image[0:common_size[0], 0:common_size[1]]
    outline_area = outline_image[0:common_size[0], 0:common_size[1]]

    ###

    shadow_area = np.zeros_like(main_area)

    for (dx, dy) in [(1, 1), (2, 2)]:
        shadow_area = np.maximum(shadow_area, shift_image(main_area, dx, dy))

    ###

    shadow_alpha = np.expand_dims(shadow_area, axis=-1)
    outline_alpha = np.expand_dims(outline_area, axis=-1)
    main_alpha = np.expand_dims(main_area, axis=-1)

    ###

    main_color = np.array([[[0, 1, 1]]], dtype=np.float32)
    outline_color = np.array([[[0, 0, 1]]], dtype=np.float32)
    shadow_color = np.array([[[0, 0, 0]]], dtype=np.float32)

    shadow_alpha *= 0.9

    result_area = test_area
    # result_area = shadow_alpha * shadow_color + (1 - shadow_alpha) * result_area
    result_area = outline_alpha * outline_color + (1 - outline_alpha) * result_area
    result_area = main_alpha * main_color + (1 - main_alpha) * result_area

    ###

    result_area = np.repeat(result_area, upscale, axis=0)
    result_area = np.repeat(result_area, upscale, axis=1)
    cv2.imwrite('result_image.png', convert_to_u8(result_area))

    #----------------------------------------------------------------
    #
    # Generate file.
    #
    #----------------------------------------------------------------

    header = {
        'rangeOrg': font_range_org,
        'rangeEnd': font_range_end,
        'unsupportedCode': 127
    }

    letters = []

    for c in font_range:
        data = font_data[c]
        t = {}
        t['advance'] = data.advance + 0
        t['org'] = data.org + 0
        t['size'] = np.array(data.unified_buffer.shape) + 0
        t['buffer'] = data.unified_buffer.flatten()

        letters.append(t)

    generate_c_font('theFonty', header, letters)
