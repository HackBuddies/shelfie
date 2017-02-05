import cv2
import numpy as np


def manhattan(pixel1, pixel2):
    return sum([abs(int(pixel1[i]) - int(pixel2[i])) for i in range(len(pixel1))])


def get_matches_rgb(img_rgb, img_template, threshold=0.28, gray=True):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    if not gray:
        img_gray = img_rgb
        template = img_template

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_diff = 5
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


def image_to_num(im1):
    matches = [0] * 5
    for i in range(5):
        im2 = cv2.imread("digits/" + str(i) + ".jpg")
        matches[i] = get_matches_rgb(im1, im2, 0.9)
    dec = [0, 0]
    sm2 = cv2.imread("digits/small4.jpeg")
    dec[0] = get_matches_rgb(im1, sm2, 0.9)

    sm2 = cv2.imread("digits/small9.jpeg")
    dec[1] = get_matches_rgb(im1, sm2, 0.9)
    if (len(dec[1]) == 2):
        last = "99"
    else:
        last = "49"
    nums = []
    for i, j in enumerate(matches):
        while (len(j) > 0):
            nums.append([j[0], i])
            j.pop(0)
    nums.sort(key=lambda x: x[0])
    final = ""
    for num in nums:
        final += str(num[1])
    final += "." + last
    return float(final)
