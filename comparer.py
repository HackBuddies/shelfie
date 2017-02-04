import cv2, random
import numpy as np
from os import listdir
from os.path import isfile, join


def euclidean(pixel1, pixel2):
    return sum([(int(pixel1[i]) - int(pixel2[i])) ** 2 for i in range(len(pixel1))])


def manhattan(pixel1, pixel2):
    return sum([abs(int(pixel1[i]) - int(pixel2[i])) for i in range(len(pixel1))])


def get_matches(big_file_path, label_path, gray=True):
    img_rgb = cv2.imread(big_file_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(label_path)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]
    if gray:
        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    else:
        res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    min_diff = 50
    loc = np.where(res >= threshold)
    pts = list(zip(*loc[::-1]))
    legit_pts = [True] * len(pts)

    # for i in range(len(pts)):
    #     if legit_pts[i]:
    #         for j in range(i + 1, len(pts)):
    #             if not legit_pts[j]:
    #                 continue
    #             if manhattan(pts[i], pts[j]) < min_diff:
    #                 legit_pts[j] = False

    real_pts = [pts[i] for i in range(len(legit_pts)) if legit_pts[i]]

    ctr = 0
    for pt in real_pts:
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        segment_img = img_rgb[pt[1]:pt[1] + h, pt[0]:pt[0] + w]
        cv2.imwrite('labels/' + "%03d.jpg" % ctr, segment_img)
        ctr += 1

    cv2.imwrite('res/' + big_file_path, img_rgb)

    return real_pts


def main():
    get_matches("test/001_001.jpg", "test/000_001.jpg")


if __name__ == '__main__':
    main()
