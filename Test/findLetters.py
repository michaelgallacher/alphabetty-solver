# from __future__ import print_function
import collections as co
import cv2 as cv
import glob
import numpy as np
import argparse
import random as rng
import sys
import time
import subprocess
import traceback

np.set_printoptions(threshold=sys.maxsize)

rng.seed(12345)

Thresh = co.namedtuple('Thresh', "channel value")


def my_round(int_input):
    # return int(np.round(int_input))
    ceiling = np.ceil(int_input)
    diff = ceiling - int_input
    return int(ceiling if diff >= 0.5 else np.floor(int_input))


g_fs_gutter = 2

find_letter_morph = 3
match_morph = 3
fbm_min_crop = 5
fbm_max_crop = 15
fbm_crop_step = 10
g_gaussian_blur_x = 3
g_gaussian_blur_y = 3

NORM_TILE_THRESH = Thresh(2, -168)
NORM_TILE_THRESH2 = Thresh(2, -136)
R_ON_Y_TILE_THRESH = Thresh(1, -168)
Y_ON_B_TILE_THRESH = Thresh(0, -140)
Y_ON_B_TILE_THRESH2 = Thresh(2, 188)
Y_ON_PU_TILE_THRESH = Thresh(1, 200)
VINES = Thresh(1, -114)
BUBBLES = Thresh(2, -180)
THRESHOLDS = [NORM_TILE_THRESH, R_ON_Y_TILE_THRESH, Y_ON_B_TILE_THRESH, Y_ON_B_TILE_THRESH2, Y_ON_PU_TILE_THRESH, VINES, BUBBLES,
              NORM_TILE_THRESH2]
fbm_count = 0

thresh_counter = co.Counter()
crop_counter = co.Counter()

parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
parser.add_argument('-i', '--input', help='Path to input image.', default='/users/michael/code/github/wordscan/test/IMG_1919.PNG')
parser.add_argument('-d', '--debug', help='Set logging level.', default='0')
parser.add_argument('-f', '--fbm', help='tile # to stop and show detail.', default='-1')
parser.add_argument('-q', '--quiet', help='quiet mode', default='false')
args = parser.parse_args()

filenames = args.input
fbm = int(args.fbm)
debug_level = 2 if fbm != -1 else int(args.debug)
quiet_mode = args.quiet != 'false'


class DebugWindow:

    def __init__(self, dbg_size=(2000, 1600)):
        self.size = dbg_size
        self.cx = 20
        self.tile_width = int(dbg_size[0] / self.cx)
        self.cy = 16
        self.tile_height = int(dbg_size[1] / self.cy)
        self.next_loc_dbg = -1
        self.img = np.zeros((dbg_size[1], dbg_size[0]), np.uint8)

    def get_next_loc(self):
        self.next_loc_dbg += 1
        loc = self.tile_width * (self.next_loc_dbg % self.cx), self.tile_height * int(
            self.next_loc_dbg / self.cx)
        d_print(f"next_loc: {loc}")
        return loc

    def get_tile_size(self):
        return self.tile_width, self.tile_height


debug_target_window = DebugWindow()
debug_candidate_window = DebugWindow()


# fidelity output; used to compare results with iOS
def f_print(out, level):
    if debug_level == -1 or level <= debug_level:
        print(out)


# diagnostic output; used to debug python implementation
def d_print(out, level=3):
    if level <= debug_level:
        print(out)


Tile = co.namedtuple('tile', 'letter end')


# determine the letter shape in the tile image from the board
def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p1 - p2).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def calc_moment(contour):
    moment = cv.moments(contour)
    divisor = moment["m00"]
    if divisor == 0:
        f_print("letter rejected due to null moment", 2)
        return sys.maxsize, sys.maxsize

    cx = int(moment["m10"] / divisor)
    cy = int(moment["m01"] / divisor)

    return cx, cy


def create_ab_font_baseline():
    # sorted in priority order
    # alphabet_common_letter = 'ETAOINSRHLDCUMFPGWYBVKXJQZ'
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    fontpath = "../wordscan/resources_ab/{0}.png"

    contours_map = []
    for i, letter in enumerate(alphabet):
        orig_img = cv.imread(fontpath.format(letter))
        # letter_img = cv.cvtColor(orig_img, cv.COLOR_BGR2GRAY)
        letter_img2 = cv.GaussianBlur(orig_img, (g_gaussian_blur_x, g_gaussian_blur_y), 0)
        channels = cv.split(letter_img2)
        letter_img3, moment = find_letter(channels[NORM_TILE_THRESH.channel], (NORM_TILE_THRESH.value, True))
        if letter_img3 is not None:
            contours_map.append((letter, letter_img3, moment))
        else:
            assert False

    # returned in lowest priority order
    return list(contours_map)


"""
candidate: tile from board
baseline: tile from baseline
"""


def cvmatch_shapes(candidate_info, baseline_info):
    letter = baseline_info[0]
    baseline = baseline_info[1]
    baseline_moment = baseline_info[2]

    candidate = candidate_info[0]
    if candidate is None:
        return -1, None

    candidate_moment = candidate_info[1]

    candidate_w = candidate.shape[1]
    candidate_h = candidate.shape[0]
    baseline_w = baseline.shape[1]
    baseline_h = baseline.shape[0]

    f_print(f"cvmatch_shapes w,h: {candidate_w}, {candidate_h}; {baseline_w}, {baseline_h}", 2)

    # check the ratios and return if they aren't close enough
    r1 = candidate_h / candidate_w
    r2 = baseline_h / baseline_w
    r_dim = abs(r2 / r1)
    if r_dim > 1:
        r_dim = abs(r1 / r2)
    if r_dim < .80:
        f_print(f"cvm: rejected due to dimension ratio {r_dim:.3f}", 2)
        return -1, None

    element = cv.getStructuringElement(cv.MORPH_RECT, (match_morph, match_morph))
    candidate = cv.morphologyEx(candidate, cv.MORPH_DILATE, element, iterations=1)

    _interpolation = cv.INTER_AREA if baseline_h < candidate_h else cv.INTER_LINEAR
    img2_resized = cv.resize(baseline, (candidate.shape[1], candidate.shape[0]), interpolation=_interpolation)
    diff_img = candidate ^ img2_resized

    # img1_resized = cv.resize(candidate, (baseline.shape[1], baseline.shape[0]), interpolation=cv.INTER_AREA)
    # diff_img = baseline ^ img1_resized

    diff_img_morph = cv.morphologyEx(diff_img, cv.MORPH_OPEN, element, iterations=1)
    _, diff_img_morph_thresh = cv.threshold(diff_img_morph, 127, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)

    # dist = np.sqrt(np.power(candidate_moment[0]/candidate_w - baseline_moment[0]/baseline_w, 2) +
    #                np.power(candidate_moment[1]/candidate_h - baseline_moment[1]/baseline_h, 2))
    # if dist > 1:
    #     dprint(f"rejected due to moment diff: {dist:.3f}",1)
    #     return -1

    diff = cv.countNonZero(diff_img_morph_thresh) / (candidate_w * candidate_h)

    f_print(f"cvm: match with diff: {diff:.3f}, r_dim: {r_dim:.3f}", 2)

    # if fbm_count == fbm and debug_level > 0:
    #     show_windows('cvm', candidate, baseline, diff_img, diff_img_morph, diff_img_morph_thresh)

    return diff, [candidate, baseline, diff_img, diff_img_morph, diff_img_morph_thresh]


def find_all_squares(src_img):
    scale = 2 if src_img.shape[1] < 800 else 1
    src_gray = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    src_gray = cv.resize(src_gray, (0, 0), fx=scale, fy=scale)
    all_squares, _, dbgs = find_squares(src_gray, 0, False)
    img_h, img_w = src_gray.shape

    if len(all_squares) <= 1:
        d_print("too few squares:")
        return []

    # show_windows('d', *dbgs)

    tile_size, left, top, right, bottom = get_tile_size(all_squares, img_w, img_h)
    # print(tile_size, left, top, right, bottom)
    tile_size = my_round(tile_size / scale)
    left = my_round(left / scale)
    top = my_round(top / scale)
    right = my_round(right / scale)
    bottom = my_round(bottom / scale)
    new_squares = []
    rows = 0
    buffer = 2
    # subtract just in case there were rounding issues.
    for y in range(top, bottom - 10, tile_size):
        rows += 1
        for x in range(left, right - 10, tile_size):
            new_rect = (int(x + buffer), int(y + buffer), tile_size - 2 * buffer, tile_size - 2 * buffer)
            f_print(f"fr: {new_rect}", 3)
            new_squares.append(new_rect)

    cols = len(new_squares) / rows
    return new_squares

    # Draw contours
    # drawing = np.zeros((img_w, img_w, 3), np.uint8)
    # color = (255, 255, 255)
    # color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    # for i in range(len(bounds)):
    #     color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    #     pt1 = (bounds[i][0], bounds[i][1])
    #     pt2 = (bounds[i][0] + bounds[i][2], bounds[i][1] + bounds[i][3])
    #     dprint(pt2)
    #     cv.rectangle(drawing, pt1, pt2, (255, 255, 255), thickness=2)

    # Draw centers
    # centers = np.int32(np.vstack(centers))
    # for (x, y), label in zip(centers, x_cluster_indicies.ravel()):
    #     c = list(map(int, colors[label]))
    #     cv.circle(drawing, (x, y), 2, c, -1)
    #
    # show_window('squares', cv.resize(drawing, (0, 0), fx=0.5, fy=0.5), wait=True)

    # return all_squares


def get_average_cluster_vals(values):
    values = sum(values, [])
    clusters = [[values[0]]]
    for value in values[1:]:
        found_group = False
        for cluster_list in clusters:
            dist = np.sqrt(np.power(value - cluster_list[0], 2))
            if dist < 30:
                cluster_list.append(value)
                found_group = True
                break
        if not found_group:
            clusters.append([value])

    return sorted([int(my_round(np.average(cluster))) for cluster in clusters])


def get_tile_size(all_squares, img_w, img_h):
    # Find centers
    x_centers = []
    y_centers = []
    for sq in all_squares:
        cx = my_round(sq[0] + sq[2] / 2)
        cy = my_round(sq[1] + sq[3] / 2)
        x_centers.append([cx])
        y_centers.append([cy])
    # print(x_centers)

    diff_counter = co.Counter()
    x_avgs = get_average_cluster_vals(x_centers)
    x_diffs = np.diff(x_avgs)
    diff_counter.update(x_diffs)
    y_avgs = get_average_cluster_vals(y_centers)
    y_diffs = np.diff(y_avgs)
    diff_counter.update(y_diffs)

    tile_size = int(diff_counter.most_common(1)[0][0])

    blue_edge = int(my_round(img_w / 50))
    top_offset = int(my_round(img_h / 45))
    screen_edge = int(my_round(img_w / 200))
    edge = blue_edge + screen_edge

    max_tile_section_height = min(tile_size * 8 + blue_edge, img_h * 2 / 3)
    top_bounds = my_round((img_h - max_tile_section_height) * 0.5) + top_offset
    f_print(f"tile diff: {tile_size}", 3)

    tile_top = my_round(y_avgs[0] - tile_size / 2)
    while tile_top - tile_size >= top_bounds - 1:
        tile_top -= tile_size

    # max_cols = int(np.floor((img_w - edge) / tile_size))
    left_bounds = edge / 2  # int(np.floor((img_w - (tile_size * max_cols)) / 2))
    tile_left = my_round(x_avgs[0] - tile_size / 2)
    while tile_left - tile_size >= left_bounds - 1:
        tile_left -= tile_size

    right = img_w - tile_left
    bottom = top_bounds + max_tile_section_height

    return tile_size, int(tile_left), int(tile_top), int(right), int(bottom)


def _find_best_match(candidate, baselines):
    global fbm_count
    best_distance = 999999
    best_candidate_index = -1
    dbg_best = None

    if candidate is not None and baselines is not None:
        # Try each candidate letter and keep the best match to the target.
        for candidateIndex, candidateLetter in enumerate(baselines):

            # Call OpenCV to determine the correlation between the letters.
            # dprint(f"target_letter: {target_letter}")
            # dprint(f"candidateLetter:{candidateLetter}")
            distance, dbgs = cvmatch_shapes(candidate, candidateLetter)
            # The smaller the value ('distance'), the better the match.
            if 0 <= distance <= best_distance:
                # dprint(f"_fbm->letter:{candidateLetter[0]} dist:{distance} count:{len(candidateLetter[1])} ")
                best_distance = distance
                best_candidate_index = candidateIndex
                dbg_best = dbgs

            if best_distance == 0:
                break

    if fbm_count == fbm and debug_level > 0 and dbg_best is not None:
        show_windows('_fbm', *dbg_best)

    return best_candidate_index, best_distance, dbg_best


def find_best_match(candidate, baselines):
    best_distance = 999999
    best_candidate_index = -1
    best_letter_image = None
    dbg_best = None

    candidate = cv.GaussianBlur(candidate, (g_gaussian_blur_x, g_gaussian_blur_y), 0)

    img_width = candidate.shape[1]
    img_height = candidate.shape[0]
    for crop in range(fbm_min_crop, fbm_max_crop + 1, fbm_crop_step):
        tile_image = candidate[crop:img_height - crop, crop:img_width - crop]
        channels = cv.split(tile_image)

        for threshold in THRESHOLDS:
            d_print(f"using threshold: {threshold}", 3)
            target_letter_info = find_letter(channels[threshold.channel], (threshold.value, False))
            if target_letter_info is None:
                continue

            target_letter, moment = target_letter_info

            tmp_best_candidate_index, tmp_best_distance, dbgs = _find_best_match(target_letter_info, baselines)
            tmp_best_image = target_letter

            f_print(f"fbm->i: {tmp_best_candidate_index} d: {tmp_best_distance:.3f}", 3)

            if 0 <= tmp_best_distance <= best_distance:
                best_distance = tmp_best_distance
                best_candidate_index = tmp_best_candidate_index
                best_letter_image = tmp_best_image
                dbg_best = dbgs

            if best_distance == 0:
                thresh_counter.update([str(threshold)])
                crop_counter.update([str(crop)])
                return best_letter_image, best_candidate_index, best_distance

    if dbg_best is not None:
        show_windows('fbm-cvm', *dbg_best)

    return best_letter_image, best_candidate_index, best_distance


def find_letter(tile_img, params):
    thresh = params[0]
    is_baseline_img = params[1]

    height = tile_img.shape[0]
    width = tile_img.shape[1]
    f_print(f"fl-ts: {width}, {height}", 2)

    if thresh > 0:
        _, tile_img = cv.threshold(tile_img, thresh, 255, cv.THRESH_BINARY)
    elif thresh < 0:
        _, tile_img = cv.threshold(tile_img, -thresh, 255, cv.THRESH_BINARY_INV)
    else:
        _, tile_img = cv.threshold(tile_img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    element = cv.getStructuringElement(cv.MORPH_RECT, (find_letter_morph, find_letter_morph))
    tile_img = cv.morphologyEx(tile_img, cv.MORPH_OPEN, element)

    # find the largest contour in the image
    contours, _ = cv.findContours(tile_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if contours is None or len(contours) == 0:
        return None

    # if len(contours) == 1 and is_baseline_img:
    #     return None

    drawing = np.zeros((height, width), np.uint8)
    # DIAG
    # for i in range(len(contours)):
    # color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    # cv.drawContours(drawing, contours, i, color, thickness=1)
    # #dprint(f"area: {cv.contourArea(contours[i])}")
    # END DIAG
    sorted_contours = sorted(contours, key=lambda _c: cv.contourArea(_c), reverse=True)

    idx = 0
    biggest_contour = sorted_contours[idx]
    moment = calc_moment(biggest_contour)
    if moment is None:
        return None

    cx, cy = moment
    dist = np.sqrt(np.power(width / 2 - cx, 2) + np.power(height / 2 - cy, 2))
    max_moment = (width * height) / 200
    if dist > max_moment:
        f_print("letter rejected due to moment: " + str(int(dist)), 2)
        return None

    # cv.drawContours(drawing, sorted_contours, idx, (255, 255, 255), thickness=3, lineType=cv.LINE_8)
    cv.drawContours(drawing, sorted_contours, idx, (255, 255, 255), thickness=cv.FILLED, lineType=cv.LINE_8)
    br = cv.boundingRect(biggest_contour)
    f_print(f"br: {br}", 3)

    if br[2] * br[3] > width * height * 0.95:
        # print(br[2]*br[3])
        # print( width * height)
        return None, None

    if br[2] < width * 0.1:
        f_print(f"letter rejected due to width: {br[2]}, {width}", 2)
        return None

    if br[3] < height * 0.4:
        f_print(f"letter rejected due to height: {br[3]}, {height}", 2)
        return None

    letter_img = tile_img[br[1]:br[1] + br[3], br[0]:br[0] + br[2]]
    f_print('found letter', 2)

    if fbm_count == fbm and not is_baseline_img:
        show_windows('diff', tile_img, drawing, letter_img)

    return letter_img, moment


def find_squares(gray, thresh, single_tile):
    blur = 3
    min_area = gray.shape[0] * gray.shape[1] / 200
    squares = []
    bounding_rects = []
    dbgs = []

    bin1 = sobel(gray, blur)

    if not single_tile or fbm == fbm_count:
        dbgs.append(bin1)

    if thresh > 0:
        _, bin1 = cv.threshold(bin1, thresh, 255, cv.THRESH_BINARY)
    elif thresh < 0:
        _, bin1 = cv.threshold(bin1, -thresh, 255, cv.THRESH_BINARY_INV)
    else:
        if single_tile:
            _, bin1 = cv.threshold(bin1, thresh, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        else:
            otsu, _ = cv.threshold(bin1, thresh, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            _, bin1 = cv.threshold(bin1, int(otsu / 2), 255, cv.THRESH_BINARY)

    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    bin1 = cv.morphologyEx(bin1, cv.MORPH_ERODE, element)

    if not single_tile or fbm == fbm_count:
        dbgs.append(bin1)

    contours, _ = cv.findContours(bin1, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda _c: cv.contourArea(_c), reverse=True)

    if not single_tile or fbm == fbm_count:
        dbgs.append(get_contours_img(sorted_contours, bin1))

    for orig_contour in sorted_contours:
        ca = cv.contourArea(orig_contour)
        if single_tile or min_area < ca < 5 * min_area:
            br = cv.boundingRect(orig_contour)
            r = br[2] / br[3]
            r = br[3] / br[2] if r > 1 else r
            if r > 0.95:
                bra = br[2] * br[3]
                # print(ca, br, bra)
                if ca > bra * 0.9:
                    squares.append(orig_contour)
                    bounding_rects.append(br)

    if not single_tile or fbm == fbm_count:
        dbgs.append(get_contours_img(squares, bin1))

    return bounding_rects, sorted_contours, dbgs


def find_squares_ab(shape, num_tile_rows, num_tile_cols):
    d_print(f"num_tile_cols: {num_tile_cols}", 1)
    d_print(f"num_tile_rows: {num_tile_rows}", 1)

    img_height, img_width, _ = shape
    if img_height > img_width:
        # portrait
        top_offset = int(my_round(img_height / 45))
        screen_edge = int(my_round(img_width / 200))
        blue_edge = int(my_round(img_width / 50))
        tile_gaps = int(my_round(img_width / 75))

        f_print(f"top_offset: {top_offset}", 1)
        f_print(f"blue_edge: {blue_edge}", 1)
        f_print(f"screen_edge: {screen_edge}", 1)
        f_print(f"tile_gaps: {tile_gaps}", 1)

        edge = screen_edge + blue_edge
        tile_width = (img_width - (2 * edge) - ((num_tile_cols - 1) * tile_gaps)) / num_tile_cols
        f_print(f"tile_width: {tile_width}", 1)
        tile_width = my_round(tile_width)
        letter_area_height = num_tile_rows * tile_width + (num_tile_rows - 1) * tile_gaps + (2 * blue_edge)
        top = int((img_height - letter_area_height) * 0.5) + top_offset
        f_print(f"top: {top}", 1)
        left = edge
        letters = []
        for row in range(0, num_tile_rows):
            for col in range(0, num_tile_cols):
                bounds = left + col * (tile_width + tile_gaps) - g_fs_gutter, \
                         top + blue_edge + row * (tile_width + tile_gaps) - g_fs_gutter, \
                         tile_width + g_fs_gutter * 2, \
                         tile_width + g_fs_gutter * 2
                d_print(f"fs: {bounds}", 3)
                letters.append(bounds)

        return letters
    else:
        # landscape
        left = int((img_width - img_height))
        d_print(left)


def get_contours_img(contours, img):
    if debug_level == 0:
        return None

    drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(len(contours)):
        if cv.contourArea(contours[i]) > 800:
            color = (rng.randint(128, 256), rng.randint(128, 256), rng.randint(128, 256))
            cv.drawContours(drawing, contours, i, color, thickness=2)
    return drawing


def sobel(src_cv, sobel_blur):
    ddepth = cv.CV_16S
    ksize = sobel_blur

    # [reduce_noise]
    # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    src_cv = cv.GaussianBlur(src_cv, (ksize, ksize), 0)
    # [reduce_noise]

    # [convert_to_gray]
    # Convert the image to grayscale
    # gray = cv.cvtColor(src_cv, cv.COLOR_BGR2GRAY)
    gray = src_cv
    # [convert_to_gray]

    # [sobel]
    # Gradient-X
    grad_x = cv.Sobel(gray, ddepth, 1, 0)

    # Gradient-Y
    grad_y = cv.Sobel(gray, ddepth, 0, 1)
    # [sobel]

    # [convert]
    # converting back to uint8
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    # [convert]

    # [blend]
    # Total Gradient (approximate)
    grad = cv.addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0)

    return grad


def thresh_callback(image_file, src_img):
    image_file_name = image_file[0:image_file.find("IMG_") + 8]
    name_expected_file = image_file_name + '.txt'

    expected = ''
    try:
        testfile = open(name_expected_file)
        lines = testfile.readlines()
        for line in lines:
            expected += line.replace('\n', '')
    except FileNotFoundError:
        expected = ''

    print('\n--------------\n' + image_file)

    start = time.process_time()
    global fbm_count
    # Find contours
    all_squares = find_all_squares(src_img)
    if len(all_squares) == 0:
        d_print("using fallback ab", 1)
        all_squares = find_squares_ab(src_img.shape, 8, 7)

    d_print(f"all_squares: {all_squares}", 3)
    images = [src_img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] for r in all_squares]

    final_match = ''
    for i, color_tile in enumerate(images):
        fbm_count = i
        f_print(f"considering tile: {i}", 2)
        # quick out when debugging a single tile
        if fbm >= 0 and fbm != fbm_count:
            final_match += '  '
            continue

        letter_image, idx, diff = find_best_match(color_tile, alphabet_baseline_contours)
        f_print(f"{i}: diff:{diff:.3f}", 2)

        possible_blank = False
        if letter_image is not None:
            if idx >= 0:
                candidate = str(alphabet_baseline_contours[idx][0])
                if diff <= 0.04:
                    if diff == 0:
                        d_print(f"{i}: exact match", 2)
                    else:
                        f_print(f"{i}: close enough match: {diff:.3f}", 1)
                    letter = candidate
                else:
                    f_print(f"{i}: match but not close enough: {diff:.3f}", 1)
                    # found a matching shape but not matching pixels
                    letter = ' '
                    possible_blank = True
            else:
                # didn't find a shape close enough
                f_print(f"{i}: shape but no match", 1)
                letter = ' '  # '?'
                possible_blank = True
        else:
            if fbm > 0 and fbm == fbm_count:
                f_print(f"{i}: no shape found", 1)
            letter = ' '  # '!'
            possible_blank = True

        if possible_blank and is_blank_tile(color_tile):
            letter = '*'

        final_match += ' ' + letter

    final_match = final_match.replace('!', ' ').replace('?', ' ')
    final_match = final_match.rstrip()
    expected = expected.rstrip()
    if final_match == expected:
        print("match!")
    elif len(expected) > 0:
        print('                                   1                   2                   3                   4                   5 ')
        print(
            '               0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5')
        print(f"---expected: [{expected}]")
        print(f"---got:    : [{final_match}]")
    else:
        print('test file not found')
        for l in range(0, len(final_match), 14):
            print(final_match[l:l + 14])

    d_print(f"time: {time.process_time() - start}", 1)
    for l in range(0, len(final_match), 14):
        d_print(final_match[l:l + 14])


def has_blank_center_only(color_tile):
    # ignore tiles which have no contours at all.
    gray_tile = cv.cvtColor(color_tile, cv.COLOR_BGR2GRAY)
    sobel_tile = sobel(gray_tile, 3)
    ranged = cv.inRange(sobel_tile, 64, 192)
    nz = cv.countNonZero(ranged)
    if nz < 10:
        d_print("hbco: fail 1", 3)
        return False

    w = sobel_tile.shape[1]
    h = sobel_tile.shape[0]

    # check if the outer edge has something everywhere
    blank_middle = sobel_tile.copy()
    edge = int(h / 8)
    blank_middle[edge:h - edge, edge:w - edge] = 0
    _, threshold_img = cv.threshold(blank_middle, 32, 255, cv.THRESH_BINARY)

    element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    morphed = cv.morphologyEx(threshold_img, cv.MORPH_CLOSE, element)
    nz = cv.countNonZero(morphed)
    if nz < h * w * 0.33:
        d_print("hbco: fail 2", 3)
        return False

    crop = int(h / 3)
    cropped = sobel_tile[crop:h - crop, crop:w - crop]
    cropped_ranged = cv.inRange(cropped, 32, 192)
    nz = cv.countNonZero(cropped_ranged)

    show_windows('cropped', sobel_tile, blank_middle, cropped_ranged)
    d_print("hbco: pass 3", 3) if nz < 10 else d_print("hbco: fail 4", 3)
    d_print(f"hbco: nz: {nz}", 3)
    return nz < 10


def has_squares_and_zero(color_tile):
    candidate_images = reversed(cv.split(color_tile))
    for candidate_img in candidate_images:
        ci_w = candidate_img.shape[1]
        ci_h = candidate_img.shape[0]
        ba = ci_w * ci_h
        for thresh in range(0, 256, 64):
            squares, contours, dbgs = find_squares(candidate_img, thresh, True)
            # show_windows('fsq', *dbgs)
            d_print("hsaz: pre-squares", 3)

            if 2 <= len(squares) <= 4 and len(contours) > 2:
                ca1 = cv.contourArea(contours[0])
                ca2 = cv.contourArea(contours[1])
                moment = calc_moment(contours[2])
                if moment is None:
                    continue

                cx, cy = moment
                moment_dist = np.sqrt(np.power(int(ci_w * .76) - cx, 2) + np.power(int(ci_h * .83) - cy, 2))
                d_print("hsaz: pre-moment", 3)
                if moment_dist < 5:
                    d_print(f"blank number moment dist: {moment_dist}")
                    outer_ratio = ca1 / ba
                    inner_ratio = ca2 / ca1
                    # print(ca1, ca2, ba, int(ca1 * 100 / ba) / 100, int(ca2 * 100 / ca1) / 100)
                    d_print("hsaz: pre-ratio", 3)
                    if outer_ratio > 0.75 and inner_ratio > 0.75:
                        d_print("hsaz: found blank", 3)
                        return True
    return False


def is_blank_tile(color_tile):
    if has_blank_center_only(color_tile):
        return True

    return has_squares_and_zero(color_tile)


fl_next_loc = (0, 0)
windows_to_position = []
windows_moved = []
fl_screen_size = (0, 0)


def init_windows():
    global fl_screen_size
    if debug_level >= 1:
        n = 'screensize'
        cv.namedWindow(n, cv.WINDOW_NORMAL)
        cv.setWindowProperty(n, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        img = np.zeros((100, 100), np.uint8)
        cv.imshow(n, img)
        fl_screen_size = cv.getWindowImageRect(n)
        fl_screen_size = (fl_screen_size[0] + fl_screen_size[2], fl_screen_size[1] + fl_screen_size[3])
        subprocess.call(["/usr/bin/osascript", "-e", 'tell app "Finder" to set frontmost of process "Python" to true'])
        # cv.setWindowProperty(n, cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
        cv.destroyWindow('screensize')
        # dprint('screensize:' + str(fl_screen_size))


windows_to_skip = []


def show_window(window_name, img, wait=True):
    if quiet_mode:
        return
    if debug_level == 0:
        return
    if window_name in windows_to_skip:
        return

    cv.namedWindow(window_name)
    cv.imshow(window_name, img)
    if wait:
        k = cv.waitKey()
        if k == ord('s'):
            print('skipping')
            windows_to_skip.append(window_name)


def show_windows(name, *_args):
    if quiet_mode:
        return
    if debug_level < 2:
        return
    if len(_args) == 0:
        return

    scale = 1.0
    new_height = max([a.shape[0] for a in _args])
    if new_height > 768:
        scale = 768.0 / new_height

    new_images = []
    for i, img in enumerate(_args):
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        new_width = img.shape[1]
        new_img = np.ones((new_height, new_width, 3), np.uint8)
        new_img[:] = (int(127), 127, 127)
        height_margin = int((new_height - img.shape[0]) / 2)

        new_img[height_margin: height_margin + img.shape[0], :] = img
        new_images.append(new_img)
        if i < len(_args) - 1:
            divider = np.zeros((new_height, 10, 3), np.uint8)
            divider[:] = (128, 200, 200)
            new_images.append(divider)

    stack = np.hstack(new_images)
    show_window(name, cv.resize(stack, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA))


#
#
#
#
#

init_windows()

alphabet_baseline_contours = create_ab_font_baseline()

cluster_n = 130
colors = np.zeros((1, cluster_n, 3), np.uint8)
colors[0, :] = 255
colors[0, :, 0] = np.arange(0, 180, 180.0 / cluster_n)
colors = cv.cvtColor(colors, cv.COLOR_HSV2BGR)[0]


def main(filename):
    # Load source image
    _src_img = cv.imread(filename)
    if _src_img is None:
        print(f"Could not open or find the image: {args.input}")
        exit(1)

    # Create Window
    source_window = 'Source'
    show_window(source_window, cv.resize(_src_img, (0, 0), fx=0.25, fy=0.25), False)
    thresh_callback(filename, _src_img)

    d_print('thresh count: ' + str(thresh_counter.most_common(4)), 1)
    d_print('crop count: ' + str(crop_counter.most_common(6)), 1)


files = [f for f in glob.glob(filenames)]
if len(files) == 0:
    print(f"no files found for: {filenames}")
    exit(1)

for file in files:
    try:
        main(file)
    except Exception as e:
        print(traceback.format_exc())

# move_windows()
if debug_level >= 2 and not quiet_mode:
    while True:
        ch = cv.waitKey()
        if ch == ord('q'):
            break
