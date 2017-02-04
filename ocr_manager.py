import pickle
import os
import ocr_ne

OCR_CACHE_FILE = "ocr_cache.pkl"


def get_text(img):
    tuple_img = tuple([tuple([tuple(pixel) for pixel in row]) for row in img])
    if os.path.exists(OCR_CACHE_FILE):
        with open(OCR_CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}
    if tuple_img in cache:
        return cache[tuple_img]
    text = ocr_ne.get_text_label(img)
    cache[tuple_img] = text
    return text

