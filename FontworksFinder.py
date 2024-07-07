import cv2
import os
import numpy as np
import PySimpleGUI as psg
import re
import unicodedata

#pytesseract.pytesseract.tesseract_cmd =  os.path.abspath(r'Tesseract-OCR\tesseract.exe')
supported = [
    ("All Image files", "*.bmp *.dib *.jpeg *.jpg *.jpe *.jp2 *.png *.webp *.pbm *.pgm *.ppm *.pxm *.pnm *.tiff *.tif *.hdr *.pic"),
    ("Windows bitmaps", "*.bmp *.dib"),
    ("JPEG files", "*.jpeg *.jpg *.jpe"),
    ("JPEG 2000 files", "*.jp2"),
    ("Portable Network Graphics files", "*.png"),
    ("WebP files", "*.webp"),
    ("Portable image format files", "*.pbm *.pgm *.ppm *.pxm *.pnm"),
    ("TIFF files", "*.tiff *.tif"),
    ("Radiance HDR files", "*.hdr *.pic")
]

def sort_contoursize(cnt):
    _, _, w, h = cv2.boundingRect(cnt)
    return w * h

def sort_contourx(cnts):
    s_cnts = sorted(cnts, key=sort_contoursize, reverse=True)
    x, _, w, _ = cv2.boundingRect(s_cnts[0])
    return x + (w/2)

image = None
while image is None:
    file = psg.popup_get_file('Select Input Image',  title="Image Selector", file_types=supported)
    if not os.path.exists(file):
        print("File does not exist or empty string.")
        continue
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if image is None: 
        print("Image could not be read, please choose a valid image file.")

MIN_HEIGHT_OR_WIDTH = 350

og_height, og_width = image.shape

if min(og_height, og_width) < MIN_HEIGHT_OR_WIDTH:
    if og_height < og_width:
        new_height = MIN_HEIGHT_OR_WIDTH
        new_width = round((og_width / og_height) * MIN_HEIGHT_OR_WIDTH)
    else:
        new_width = MIN_HEIGHT_OR_WIDTH
        new_height = round((og_height / og_width) * MIN_HEIGHT_OR_WIDTH)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
height, width = image.shape

min_dim = min(height, width)

blurh = blurw = int(np.ceil(max(5, min_dim*0.01)) // 2 * 2 + 1)

_, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

blacknum = whitenum = 0
for y in range(5):
    for x in range(width):
        pix = thresh[y, x]
        if pix == 0:
            blacknum += 1
        else:
            whitenum += 1
if whitenum > blacknum:
    thresh = cv2.bitwise_not(thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(blurw,blurh))

erosion = cv2.erode(thresh, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel, iterations=1)

# Get the text area
big_kxy = min_dim // 4
big_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (big_kxy, big_kxy))

# Applying dilation on the threshold image
areadilate = cv2.dilate(dilation, big_kernel, iterations=1)

# Finding contours
big_contours, _ = cv2.findContours(areadilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fw_fonts_np = np.load('FontData.npy', allow_pickle=True)
fw_fonts = fw_fonts_np.tolist()

text_areas = []

big_contours = sorted(list(big_contours), key=sort_contourx)

for big_cnt in big_contours:
    x, y, w, h = cv2.boundingRect(big_cnt)
    
    cropped = dilation[y:y + h, x:x + w]
    cropped_og = image[y:y + h, x:x + w]
    
    display_cropped = cropped_og.copy()

    display_cropped = cv2.bitwise_and(display_cropped, cv2.bitwise_not(cropped))

    disp_w = 450
    disp_h = round((h / w) * disp_w)

    if disp_w < w:
        display_cropped = cv2.resize(display_cropped, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        display_cropped = cv2.resize(display_cropped, (disp_w, disp_h), interpolation=cv2.INTER_CUBIC)

    contours, _ = cv2.findContours(cropped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    
    #foundtext = pytesseract.image_to_string(cropped_og).strip()
    #if foundtext is not None and foundtext != '':
    values = {}
    event = ''
        
    imgdata = cv2.imencode('.png', display_cropped)[1].tobytes()
    layout = [
        [psg.Image(key='-IMAGE-', data=imgdata)],
        [psg.Text('Enter text in Image '), psg.Input(expand_x=True, key='-TEXT-', focus=True)],
        [psg.OK(), psg.Cancel()]
    ]
    window = psg.Window('Image Viewer', layout, resizable=True, finalize=True)
    while True:
        event, values = window.read()
        if event in (None, psg.WIN_CLOSED, 'Cancel', 'OK'):
            break
    window.close()
    if event not in (None, psg.WIN_CLOSED, 'Cancel'):
        endtext = unicodedata.normalize('NFKC', values['-TEXT-'].strip())
        text_areas.append({
            'text': endtext,
            'croppedimg' : cropped_og.copy(),
            'contours': contours
        })
    
def full_contour_area(conts):
    s_conts = sorted(conts, key=sort_contoursize, reverse=True)
    x, y, w, h = cv2.boundingRect(s_conts[0])
    f_w = x + w
    f_h = y + h
    for cnt in s_conts:
        tx, ty, tw, th = cv2.boundingRect(cnt)
        if tx < x:
            x = tx
        if ty < y:
            y = ty
        if tx + tw > f_w:
            f_w = tx + tw
        if ty + th > f_h:
            f_h = ty + th
    w = f_w - x
    h = f_h - y
    return x, y, w, h

def draw_full_contour_mat(conts, img):
    img2 = img.copy()
    for cnt in conts:
        img_cnt = np.zeros_like(img2)
        img_cnt = cv2.drawContours(img_cnt, [cnt], 0, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
        img2 = cv2.bitwise_xor(img2, img_cnt)
    return img2

for area in text_areas:
    chars = list(re.sub(r'\s', '', area['text']))
    basestr = area['text']
    conts = list(area['contours'])
    baseimg = area['croppedimg']
    
    
    conts_org = []
    while len(conts) > 0:
        cnt = conts.pop(0)
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        cnt_sublist = [cnt]
        
        i = 0
        while i < len(conts):
            p_cnt = conts[i]
            px, py, pw, ph = cv2.boundingRect(p_cnt)
            if ((px > cx and px < cx + cw) and (py > cy and py < cy + ch) and pw < cw and ph < ch):
                cnt_sublist.append(p_cnt)
                conts.pop(i)
            elif ((cx > px and cx < px + pw) and (cy > py and cy < py + ph) and cw < pw and ch < ph):
                conts.extend(cnt_sublist)
                cnt_sublist.clear()
                break
            else:
                i += 1
        if len(cnt_sublist) > 0:
            conts_org.append(cnt_sublist)
    
    conts_org.sort(key=sort_contourx)

    greybase = baseimg.copy()
    base_height, base_width = baseimg.shape
    baseimg[:,:] //= 3
    baseimg[:,:] += 85
    greybase[:,:] //= 4
    greybase[:,:] += 148
    cropped_images = []
    greyd_images = []
    for cnts in conts_org:
        x, y, w, h = full_contour_area(cnts)
        
        nwh = min(base_height, base_width)

        cx = x + (w/2)
        cy = y + (h/2)
        
        nx = max(0, round(cx - (nwh/2)))
        ny = max(0, round(cy - (nwh/2)))
        
        if nx + nwh > base_width:
            nx = base_width - nwh
        if ny + nwh > base_height:
            ny = base_height - nwh
            
        baseimg2 = baseimg.copy()
        greybase2 = greybase.copy()
        blackmat = np.zeros((base_height, base_width, 1), np.uint8)
        grey_conts = draw_full_contour_mat(cnts, blackmat)
        base_conts = cv2.bitwise_not(grey_conts.copy())
        grey_conts[:,:] //= 2

        baseimg2 = cv2.bitwise_and(baseimg2, base_conts)
        greybase2 = cv2.bitwise_or(cv2.bitwise_and(greybase2, base_conts), grey_conts)
        
        img = cv2.resize(baseimg2[ny:ny + nwh, nx:nx + nwh], (50,50))
        grimg = cv2.resize(greybase2[ny:ny + nwh, nx:nx + nwh], (50,50))
        cropped_images.append(cv2.imencode('.png', img)[1].tobytes())
        greyd_images.append(cv2.imencode('.png', grimg)[1].tobytes())
    column = []
    SPLIT_ON = 7
    i=0
    pos_char = 0
    samelen_bool = len(chars) == len(conts_org)
    for char in chars:
        while char != basestr[pos_char]:
            pos_char += 1
        startpad = ' '*(len(basestr) - len(basestr[:pos_char]))
        endpad = ' '*((len(basestr) - len(basestr[pos_char+1:])) - 1)
        column.extend([
            [psg.HorizontalSeparator()],
            [psg.Text('Select all parts of the following character:')],
            [psg.Text(f"{startpad}\'{basestr[:pos_char]}"), 
                psg.Text(char, font='Arial 15 bold'), 
                psg.Text(f"{basestr[pos_char+1:]}\'{endpad}")]
        ])
        cnt_layout = []
        j = 0
        for cnt in conts_org:
            if samelen_bool:
                if j != i:
                    image_data = greyd_images[j]
                    on_off = False
                    disabled = True
                else:
                    image_data = cropped_images[j]
                    on_off = True
                    disabled = False
            else:
                image_data = cropped_images[j]
                on_off = False
                disabled = False
            cnt_layout.extend([psg.Checkbox('',key=(i, j), enable_events=True, default=on_off, disabled=disabled), psg.Image(key=(i, j, 'image'), data=image_data)])
            j += 1
        split_cnt_layout = [cnt_layout[i:i + SPLIT_ON*2] for i in range(0, len(cnt_layout), SPLIT_ON*2)]
        column.extend(split_cnt_layout)
        i += 1
    layout = [
        [psg.Text('Click the checkbox on any shape that makes up the displayed symbol', justification='center', expand_x=True)],
        [psg.Column(column, scrollable=True, vertical_scroll_only=True, element_justification='center', expand_x=True, expand_y=True, 
                    size_subsample_height=max(1, round((len(chars) * np.ceil(len(conts_org)/SPLIT_ON))*.25)), size_subsample_width=0.975)],
       
        [psg.Column([[psg.Text('Click OK once finished')],[psg.OK()]], expand_x=True, size_subsample_height=1, size_subsample_width=0.975, element_justification='center')]
    ]
    window = psg.Window('Assemble Letters', layout, resizable=True, finalize=True)
    values = {}
    while True:
        event, values = window.read()
        if event in (psg.WIN_CLOSED, 'OK'):
            break
        else:
            row = event[0]
            column = event[1]
            for i in range(len(chars)):
                if i != row:
                    if values[event] == True:
                        window[(i, column)].update(disabled=True)
                        window[(i, column, 'image')].update(data=greyd_images[column])
                    elif values[event] == False:
                        window[(i, column)].update(disabled=False)
                        window[(i, column, 'image')].update(data=cropped_images[column])
    window.close()
    charconts = {}
    for char in chars:
        charconts[char] = []
    for key in values.keys():
        if values[key] == True:
            char_i = key[0]
            cnts_i = key[1]
            char = chars[char_i]
            cnts = conts_org[cnts_i]
            if chars.index(char) == char_i:
                charconts[char].extend(cnts)
    font_scores = []
    for font in fw_fonts:
        fw_charconts = font['char_contours']
        result_dict = {}
        result_score = 0
        if set(charconts.keys()).issubset(fw_charconts.keys()):
            for char in charconts.keys():
                conts = charconts[char].copy()
                fw_conts = list(fw_charconts[char]['cont'])
                fw_w = fw_charconts[char]['width']
                fw_h = fw_charconts[char]['height']
                x, y, w, h = full_contour_area(conts)
                #fw_x, fw_y, fw_w, fw_h = full_contour_area(fw_conts)
                img = np.zeros((y + h, x + w, 1), np.uint8)
                fw_img = np.zeros((fw_h, fw_w, 1), np.uint8)
                    # fw_cnt[:,:,0] = np.floor(fw_cnt[:,:,0] * fw_coef_x)
                img = draw_full_contour_mat(conts, img)
                img = img[y:y + h, x:x + w]
                h, w = img.shape[:2]
                fw_img = draw_full_contour_mat(fw_conts, fw_img)
                if fw_h > h :
                    fw_img = cv2.resize(fw_img, (w, h), interpolation=cv2.INTER_AREA)
                else:
                    fw_img = cv2.resize(fw_img, (w, h), interpolation=cv2.INTER_CUBIC)
                #cropped = img[y:y + h, x:x + w]
                #fw_cropped = fw_img[fw_y:fw_y + fw_h, fw_x:fw_x + fw_w]
                # cv2.imwrite(f'{char}.png', img)
                # cv2.imwrite(f'{char} - {font['filename']}.png', fw_img)
                # if fw_w != w or fw_h != h:
                #     fw_cropped = cv2.resize(fw_cropped, (w, h), interpolation=cv2.INTER_AREA)
                fw_img_result = fw_img.copy()
                result_img = cv2.matchTemplate(img, fw_img, cv2.TM_CCOEFF_NORMED)
                _, result, _, _ = cv2.minMaxLoc(result_img)
                result_min = (1 - result) * 0.5
                result_dict[char] = {
                    'score': result_min,
                    'img': fw_img_result
                }
                result_score += result_min
                if result_score / len(result_dict.keys()) > 0.3:
                    break
            if result_score / len(result_dict.keys()) > 0.3:
                continue
            result_score /= len(result_dict.keys())
            font_report = {
                'name': font['name'],
                'filename': font['filename'],
                'basic_score': result_score,
                'score_dict': result_dict
            }
            font_scores.append(font_report)
    font_scores.sort(key=lambda x: x['basic_score'])
    # scores_to_show = font_scores[:5]
    scores_to_show = []
    for report in font_scores:
        if report['basic_score'] < 0.05:
            scores_to_show.append(report)
            font_scores.remove(report)
    scores_to_show.extend(font_scores[:max(0, 10 - len(scores_to_show))])
    
    column = []
    text_extra_y = 20
    base_h, base_w = baseimg.shape[:2]
    text_y = base_h + (text_extra_y // 2)
    base_h = base_h + text_extra_y
    for report in scores_to_show:
        font_img_bse  = np.zeros((base_h, base_w), np.uint8)
        font_img_fnt  = np.zeros((base_h, base_w), np.uint8)
        font_img_info = np.zeros((base_h, base_w), np.uint8)
        for cnts in conts_org:
            font_img_bse = draw_full_contour_mat(cnts, font_img_bse)
        #print(f'{report['name']} [{report['filename']}]: {(1 - report['basic_score']) * 100}%')
        char_scores = report['score_dict']
        for i in range(len(chars)):
            char = chars[i]
            cnts = charconts[char].copy()

            char_dict = char_scores[char]
            score = char_dict['score']
            crop_img = char_dict['img'].copy()
            #print(f"\t\'{char}\':\t{(1 - char_dict['score']) * 100}%")
            x, y, _, _ = full_contour_area(cnts)
            # crop_img = cv2.resize(crop_img, (w, h), interpolation=cv2.INTER_AREA)
            w = crop_img.shape[1]
            h = crop_img.shape[0]
            
            text = f'{round((1 - score) * 100)}%'
            
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)
            
            text_w, _ = text_size
            
            text_x = x + (w // 2) - (text_w // 2)

            cv2.putText(font_img_info, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255))

            font_img_fnt[y:y + h, x:x + w] = crop_img[:,:]
            
        green = cv2.bitwise_and(font_img_bse, font_img_fnt)
        
        REDUCE_AMOUNT = 0.85
        font_img_bse[:,:] = np.floor(font_img_bse[:,:] * REDUCE_AMOUNT)
        font_img_fnt[:,:] = np.floor(font_img_fnt[:,:] * REDUCE_AMOUNT)
        
        font_img_bse = cv2.add(font_img_bse, green)
        font_img_fnt = cv2.add(font_img_fnt, green)
        
        green = cv2.add(font_img_info, green)
        merge_img = cv2.merge([font_img_bse, green, font_img_fnt])
        
        h, w = merge_img.shape[:2]
        
        disp_w = 850
        disp_h = round((h / w) * disp_w)
        
        if disp_w < w:
            resized = cv2.resize(merge_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(merge_img, (disp_w, disp_h), interpolation=cv2.INTER_CUBIC)
        
        imgdata = cv2.imencode('.png', resized)[1].tobytes()
        
        column.extend([
            [psg.Text(f'{report['name']} - score: {round((1 - report['basic_score']) * 100, 5)}%', font='Arial 15 bold')],
            [psg.Text(f'filename: {report['filename']}', font='Arial 14 normal')],
            [psg.Image(data=imgdata)],
            [psg.HorizontalSeparator()]
        ])
    layout = [
        [psg.Column(column, scrollable=True, vertical_scroll_only=True, element_justification='left', expand_x=True, expand_y=True, 
            size_subsample_height=max(1, round(len(scores_to_show)*0.65)), size_subsample_width=0.975)],
        [psg.OK()]
    ]
    window = psg.Window('Image Viewer', layout, resizable=True, finalize=True)
    while True:
        event, values = window.read()
        if event in (psg.WIN_CLOSED, 'OK'):
            break
    window.close()
        #cv2.imwrite(f'{report['name']}.png', merge_img)
    
    
#                if len(conts) != len(fw_conts):
#                else:
#                     total_results = 0
#                     for cnt in conts:
#                         cx, cy, cw, ch = cv2.boundingRect(cnt)
#                         per_x = (cx - x) / w
#                         per_y = (cy - y) / h
#                         per_w = cw / w
#                         per_h = ch / h
#                         best_fit = 1
#                         best_fit_cnt = None
#                         fw_conts_len = len(fw_conts)
#                         for i in range(fw_conts_len):
#                             fw_cnt = fw_conts.pop(0)
#                             fw_cx, fw_cy, fw_cw, fw_ch = cv2.boundingRect(fw_cnt)
#                             fw_per_x = (fw_cx - fw_x) / fw_w
#                             fw_per_y = (fw_cy - fw_y) / fw_h
#                             fw_per_w = fw_cw / fw_w
#                             fw_per_h = fw_ch / fw_h
#                             fit_score = abs(per_x - fw_per_x) + abs(per_y - fw_per_y) + abs(per_w - fw_per_w) + abs(per_h - fw_per_h)
#                             if fit_score < best_fit:
#                                 best_fit = fit_score
#                                 fw_conts.append(best_fit_cnt)
#                                 best_fit_cnt = fw_cnt
#                             else:
#                                 fw_conts.append(fw_cnt)
#                         total_results += cv2.matchShapes(cnt, best_fit_cnt, 1, 0.0)
#                     result = total_results / len(conts)
#                     result_dict[char] = result
#                     result_score += result