import cv2, random
import numpy as np
from os import listdir
from os.path import isfile, join

MAX_DIFF_HEIGHT = 40
TRUTH_DIR = "truth"
IMAGE_DIR = "images"
TRAINING_LABEL_FILE = 'images/label/label.jpg'

FIRST_PART_HEIGHT = 22
PARTITION_HEIGHTS = [0, 23, 40, 70]

SPLIT_DIR = "split"


def get_files(dir_name):
    files = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]
    files.sort()
    return files


def euclidean(pixel1, pixel2):
    return sum([(int(pixel1[i]) - int(pixel2[i])) ** 2 for i in range(len(pixel1))])


def manhattan(pixel1, pixel2):
    return sum([abs(int(pixel1[i]) - int(pixel2[i])) for i in range(len(pixel1))])


def get_heights(lines):
    heights = [0]
    for line in lines:
        avg_height = sum([pt[1] for pt in line]) / len(line)
        heights.append(avg_height)

    heights.sort()
    return heights


def get_matches(big_file_path, label_path, threshold=0.28):
    img_rgb = cv2.imread(big_file_path)
    label_template = cv2.imread(label_path)
    return get_matches_rgb(img_rgb, label_template, threshold)


def get_matches_rgb(img_rgb, img_template, threshold=0.28, gray=True):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    if not gray:
        img_gray = img_rgb
        template = img_template

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_diff = 50
    loc = np.where(res >= threshold)
    pts = list(zip(*loc[::-1]))
    legit_pts = [True] * len(pts)

    for i in range(len(pts)):
        if legit_pts[i]:
            for j in range(i + 1, len(pts)):
                if not legit_pts[j]:
                    continue
                if manhattan(pts[i], pts[j]) < min_diff:
                    legit_pts[j] = False

    real_pts = [pts[i] for i in range(len(legit_pts)) if legit_pts[i]]

    ctr = 0
    for pt in real_pts:
        # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        segment_img = img_rgb[pt[1]:pt[1] + h, pt[0]:pt[0] + w]
        cv2.imwrite('labels/' + "%03d.jpg" % ctr, segment_img)
        ctr += 1

    # cv2.imwrite('res/' + big_file_path, img_rgb)

    return real_pts


def get_labels(big_file, training_label):
    pts = get_matches(big_file, training_label)
    img_rgb = cv2.imread(big_file)
    template = cv2.imread(training_label, 0)
    w, h = template.shape[::-1]
    labels = [img_rgb[pt[1]:pt[1] + h, pt[0]:pt[0] + w] for pt in pts]
    return labels


def split_write_labels():
    ctr = 0
    for big_file in get_files(IMAGE_DIR):
        labels = get_labels(big_file, TRAINING_LABEL_FILE)
        for label, index in zip(labels, range(len(labels))):
            for i in range(len(PARTITION_HEIGHTS) - 1):
                y1 = PARTITION_HEIGHTS[i]
                y2 = PARTITION_HEIGHTS[i + 1]
                split_img = label[y1:y2]
                cv2.imwrite(
                    SPLIT_DIR + "/" + str(i) + "/" + big_file.split("/")[-1].split(".")[0] + str(index) + ".jpg",
                    split_img)
        ctr += 1
        print(ctr)


def get_horizontal_lines(real_pts, label_height, label_width, big_img_width):
    max_diff_height = 40

    lines = []
    curr_line = []
    prev_pt = None

    real_pts = [[pt[0], pt[1] + label_height] for pt in real_pts]

    for pt in sorted(real_pts, key=lambda x: x[1]):
        if prev_pt is None:
            prev_pt = pt
            curr_line = [pt]
            continue
        diff_height = abs(prev_pt[1] - pt[1])
        prev_pt = pt
        if diff_height > max_diff_height:
            lines.append(curr_line)
            curr_line = [pt]
            continue
        else:
            curr_line.append(pt)
    lines.append(curr_line)
    for line in lines:
        line.insert(0, [-label_width, line[0][1]])
        line.append([big_img_width, line[-1][1]])
        line.sort(key=lambda x: x[0])
    return lines


def get_segment_rects_labels(test_fname, label_fname):
    img_rgb = cv2.imread(test_fname)

    template = cv2.imread(label_fname, 0)
    label_w, label_h = template.shape[::-1]
    _, big_w, big_h = img_rgb.shape[::-1]
    real_pts = get_matches(test_fname, label_fname)
    lines = get_horizontal_lines(real_pts, label_h, label_w, big_w)
    heights = get_heights(lines)
    heights = heights[::-1]

    result = []

    for line in lines:
        for i in range(0, len(line) - 2):
            pt1, mid_pt, pt2 = line[i:i + 3]
            left_border = pt1[0] + label_w
            right_border = pt2[0]
            bottom_limit = min(pt1[1], pt2[1]) - label_h
            upper_limit = 0
            for height in heights:
                if pt1[1] - height > MAX_DIFF_HEIGHT:
                    upper_limit = height
                    break
            result.append(([(left_border, upper_limit), (right_border, bottom_limit)],
                           [(mid_pt[0], mid_pt[1]-label_h), (mid_pt[0] + label_w, mid_pt[1])]))

    return result


def get_segmented_images_and_labels(test_fname, label_fname):
    img_rgb = cv2.imread(test_fname)
    rect_labels = get_segment_rects_labels(test_fname, label_fname)
    imgs = []
    for rectangle, label in rect_labels:
        segment_img = img_rgb[rectangle[0][1]:rectangle[1][1], rectangle[0][0]:rectangle[1][0]]
        label_img = img_rgb[label[0][1]:label[1][1], label[0][0]:label[1][0]]
        imgs.append([segment_img, label_img])
    return imgs


def get_truth_images_brand_sku():
    with open(TRUTH_DIR + "/brands", "r") as f:
        brands = [line.strip() for line in f.readlines()]
    with open(TRUTH_DIR + "/skus", "r") as f:
        skus = [line.strip() for line in f.readlines()]
    return [(cv2.imread(fname), brand, sku) for fname, brand, sku in zip(get_files(TRUTH_DIR), brands, skus)]


def get_brand_sku_by_truth(truth_imgs_brand_sku, test_img):
    threshold = 0.8
    while True:
        for img, brand, sku in truth_imgs_brand_sku:
            if len(test_img) < len(img) or len(test_img[0]) < len(img[0]):
                continue
            if len(get_matches_rgb(test_img, img, threshold=threshold, gray=False)) > 0:
                return brand, sku
        threshold -= 0.05
    return "Unknown", "Unknown"


def get_real_name_label_name():
    pass


def write_segmented_images(test_fname, label_fname, prefix_num):
    ctr = 0
    for img, label in get_segmented_images_and_labels(test_fname, label_fname):
        cv2.imwrite('segments/' + "%03d_%03d.jpg" % (prefix_num, ctr), img)
        ctr += 1


def group_related_images(label_fname):
    dict_related = {}
    files = get_files("images")
    for fname in files:
        print(fname)
        segmented_imgs = [item[0] for item in get_segmented_images_and_labels(fname, label_fname)]
        for segment, index in zip(segmented_imgs, range(len(segmented_imgs))):
            if index not in dict_related:
                dict_related[index] = segment
            else:
                dict_related[index] = min(dict_related[index], segment, key=lambda x: len(x[0]))

    for key, img in dict_related.items():
        cv2.imwrite('test/%03d.jpg' % key, img)


def test_labelling_accuracy():
    count_dict = {}
    files = get_files("images")
    for f, index in zip(files, range(len(files))):
        real_pts = get_matches(f, TRAINING_LABEL_FILE)
        match_count = len(real_pts)
        if match_count != 22:
            print(f)
        count_dict[match_count] = count_dict.get(match_count, 0)
        print(count_dict)
        # write_segmented_images(f, TRAINING_LABEL_FILE, index)


def main():
    print(get_matches("split/0/Supermarket_2015-01-1817.jpg", "split/0/Supermarket_2016-12-2516.jpg", threshold=0.8))


if __name__ == '__main__':
    main()
