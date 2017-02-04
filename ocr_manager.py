import pickle
import os
import ocr_ne
import cv2

OCR_CACHE_FILE = "ocr_cache.pkl"

cache = None


def get_text(key, img):
    global cache

    if cache is None:
        if os.path.exists(OCR_CACHE_FILE):
            with open(OCR_CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
        else:
            cache = {}

    resized_img = cv2.resize(img, (100, 50))

    tuple_img = tuple([tuple([tuple(pixel) for pixel in row]) for row in resized_img])
    if tuple_img in cache:
        return cache[tuple_img]
    result = ocr_ne.get_text_label(img)
    cache[tuple_img] = result
    return result


def store_cache():
    with open(OCR_CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
