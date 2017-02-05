import requests
import time
import collections
import cv2
import pandas as pd
import kai_searcher
import ocr
import templatematching

BRAND_DIR = kai_searcher.SPLIT_DIR + "/0"
SKU_DIR = kai_searcher.SPLIT_DIR + "/1"

KEY_WHITE = "White"
KEY_YELLOW = "Yellow"
KEY_BRAND = "Brand"
KEY_SKU = "Sku"


def get_brand(img):
    df_brand = pd.read_csv("brands.csv")
    brand = "Unknown"
    found = False
    threshold = 0.9
    while not found:
        for i in range(df_brand.shape[0]):
            if found:
                break
            image_paths = [df_brand.iloc[i][KEY_WHITE], df_brand.iloc[i][KEY_YELLOW]]
            for image_path in image_paths:
                img_template = cv2.imread(BRAND_DIR + "/" + image_path)
                if len(kai_searcher.get_matches_rgb(img, img_template,
                                                    threshold=threshold, gray=True)) != 0:
                    brand = df_brand.iloc[i][KEY_BRAND]
                    found = True
                    break
        threshold -= 0.05
    return brand


def get_sku(img):
    df_sku = pd.read_csv("sku.csv")
    sku = "Unknown"
    found = False
    threshold = 0.9
    while not found:
        for i in range(df_sku.shape[0]):
            if found:
                break
            image_paths = [df_sku.iloc[i][KEY_WHITE], df_sku.iloc[i][KEY_YELLOW]]
            for image_path in image_paths:
                img_template = cv2.imread(SKU_DIR + "/" + image_path)
                if len(kai_searcher.get_matches_rgb(img, img_template,
                                                    threshold=threshold, gray=False)) != 0:
                    sku = df_sku.iloc[i][KEY_SKU]
                    found = True
                    break
        threshold -= 0.05
    return sku


def get_price(img):
    cash_part = img[kai_searcher.PARTITION_HEIGHTS[2]:kai_searcher.PARTITION_HEIGHTS[3]]
    price = templatematching.image_to_num(cash_part)
    return price


def main():
    fd = collections.Counter([get_brand(label) for label in kai_searcher.get_labels("images/2.jpg",
                                                                                    kai_searcher.TRAINING_LABEL_FILE)])

    print(fd)
    return
    for i in range(1, 22):
        path = "labels/%03d.jpg" % i
        img = cv2.imread(path)
        print(get_price(img))


if __name__ == '__main__':
    main()
