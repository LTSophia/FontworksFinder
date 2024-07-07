import unicodedata
import os
import cv2
from fontTools import ttLib
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import unicodedata

fw_fonts = []


FONT_DIR = "D:\\Downloads\\Fontworks\\Fonts"

def sort_contoursize(cnt):
    _, _, w, h = cv2.boundingRect(cnt)
    return w * h

def drawchar(character, font):
    left, top, right, bottom = font.getbbox(character, anchor='la')
    anch_x = 0
    anch_y = 0
    width  = abs(right) + abs(left) + 10
    height = abs(bottom) + abs(top) + 10
    image = Image.new('L', (width, height), color=0)
    drawing = ImageDraw.Draw(image)
    drawing.text(
                (anch_x, anch_y),
                character,
                fill=(255),
                anchor='la',
                font=font
    )
    image_np = np.array(image)
    _, thresh = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY)
    if thresh is None or not thresh.any():
        return ([], 0, 0)
    x   = None
    f_w = None
    y   = None
    f_h = None
    for h in range(height):
        if y is None and thresh[h,:].any():
            y = max(0, h - 1)
        if f_h is None and thresh[(h+1)*-1,:].any():
            f_h = height - max(0, h - 1)
        if y is not None and f_h is not None:
            break
    for w in range(width):
        if x is None and thresh[:,w].any():
            x = max(0, w - 1)
        if f_w is None and thresh[:,(w+1)*-1].any():
            f_w = width - max(0, w - 1)
        if x is not None and f_w is not None:
            break
    thresh = thresh[y:f_h, x:f_w]
    h, w = thresh.shape
        #cv2.imwrite(f'{character} - {font.font.family}.png', thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    return (contours, w, h)
    
unihexlist = list(range(0x21, 0xAA)) + [0xAE, 0xB0, 0xB8, 0xBA, 0xF7, 0x2BB, 0x2BC, 0x2C6, 0x2DA, 0x2DC, 0x20AC, 0x2122] + list(range(0x2010, 0x2020)) + list(range(0x2032, 0x2038))

font_list = list(filter(lambda x: x.lower().endswith(".otf"), os.listdir(FONT_DIR)))
total = len(font_list)
tot_len = len(f'{total}')
i = 1
for file in font_list:
    fontpath = os.path.join(FONT_DIR,file)
    i_pad = '0'*max(0, tot_len - len(f'{i}'))
    charconts = {}
    pil_font = ImageFont.truetype(fontpath, 250)
    with ttLib.TTFont(fontpath) as font:
        cmap = font['cmap'].buildReversed()
        for key in cmap.keys():
            charlist = sorted(list(cmap[key]))
            if len(charlist) > 0:
                unichar = None #chr(charlist[0])
                for charhex in charlist:
                    if charhex in unihexlist:
                        unichar = chr(charhex)
                        break
                if unichar is not None:
                    unichar = unicodedata.normalize('NFKC', unichar)
                    if unichar not in charconts.keys():
                        conts, width, height = drawchar(unichar, pil_font)
                        if len(conts) > 0:
                            charconts[unichar] = {
                                'cont': conts,
                                'width': width,
                                'height': height
                            }
        fontname = font['name'].getBestFullName()
        spacing = ' '*max(1, 45-len(fontname))
        print(f'[{fontname}]{spacing}({i_pad}{i}/{total})')
        fontdict = {
            'name': fontname,
            'filename': file,
            'char_contours': charconts
            }
        fw_fonts.append(fontdict)
    i += 1
fw_fonts_np = np.array(fw_fonts)
np.save('FontData.npy', fw_fonts_np)




# def char_in_font(unicode_char, font):
#     for cmap in font['cmap'].tables:
#         if cmap.isUnicode():
#             if ord(unicode_char) in cmap.cmap:
#                 return True
#     return False

# def test(char):
#     for fontpath in fonts:
#         font = TTFont(fontpath)   # specify the path to the font in question
#         if char_in_font(char, font):
#             print(char + " "+ unicodedata.name(char) + " in " + fontpath) 

