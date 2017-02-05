import datetime
import pandas as pd
import kai_searcher
import ocr_ne
import pickle
import collections
import cv2
import json

KEY_DATE = "Date"
KEY_BRAND = "Brand"
KEY_SKU = "SKU"
KEY_RECOMMENDED_PRICE = "Recommended Price"

PKL_FILE_MISPRICE = "tasks_mispriced.pkl"
PKL_FILE_MISPLACE = "tasks_misplaced.pkl"

ALERT_PRICE = 1
ALERT_MISMATCH = 2


def get_misprice(date):
    with open(PKL_FILE_MISPRICE, "rb") as f:
        alert_dict = pickle.load(f)
    date = date.strftime("%d/%m/%Y")
    print(alert_dict)
    return alert_dict[date]


def get_misplace(date):
    with open(PKL_FILE_MISPLACE, "rb") as f:
        alert_dict = pickle.load(f)
    date = date.strftime("%d/%m/%Y")
    date = str(date)
    return alert_dict[date]


def read_misplace():
    with open(PKL_FILE_MISPLACE, "rb") as f:
        m_dict = pickle.load(f)
    for key in m_dict:
        print(key, m_dict[key])


def get_filename_at_date(date_str):
    date = datetime.datetime.strptime(date_str, "%d/%m/%Y").date()
    return "images/Supermarket_" + str(date) + ".jpg"


def count_mismatch():
    df = pd.read_csv("sheet.csv")
    dates = df[KEY_DATE].unique()
    alert_dict = {}
    ctr = 0
    img_brand_skus = kai_searcher.get_truth_images_brand_sku()
    for date in dates:
        alerts = []
        for img, label in kai_searcher.get_segmented_images_and_labels(get_filename_at_date(date),
                                                                       kai_searcher.TRAINING_LABEL_FILE):
            brand1, sku1 = kai_searcher.get_brand_sku_by_truth(img_brand_skus, img)
            brand2, sku2 = ocr_ne.get_brand(label), ocr_ne.get_sku(label)

            cv2.imwrite("temp.jpg", label)
            cv2.imwrite("temp2.jpg", img)

            if brand1 != brand2 or sku1 != sku2:
                alerts.append([date, brand1, sku1, brand2, sku2])
                print(brand1, brand2, sku1, sku2, date)
        ctr += 1
        print(ctr)
        alert_dict[date] = alerts
    with open(PKL_FILE_MISPLACE, "wb") as f:
        pickle.dump(alert_dict, f)


def analyse_misprice():
    with open(PKL_FILE_MISPRICE, "rb") as f:
        alert_dict = pickle.load(f)
    tot_alerts = sum([len(value) for value in alert_dict.values()])
    avg_alerts = tot_alerts * 1.0 / len(alert_dict)
    print(tot_alerts)
    print(avg_alerts)


def get_price_alert(label, brand, sku, actual_price, expected_price):
    return [ALERT_PRICE, brand, sku, actual_price, expected_price]


def test_labels():
    df = pd.read_csv("sheet.csv")
    dates = df[KEY_DATE].unique()
    truth = None
    ctr = 0
    for date in dates:
        fname = get_filename_at_date(date)
        labels = kai_searcher.get_labels(fname, kai_searcher.TRAINING_LABEL_FILE)
        brands = [ocr_ne.get_brand(img) for img in labels]
        fd = collections.Counter(brands)
        if truth is None:
            truth = fd
            print(truth)
        elif truth != fd:
            print(fd)
            print(date)
        ctr += 1
        print(ctr)


def main():
    df = pd.read_csv("sheet.csv")
    dates = df[KEY_DATE].unique()
    alert_dict = {}
    ctr = 0
    for date in dates:
        alerts = []
        dated_df = df[df[KEY_DATE] == date]
        fname = get_filename_at_date(date)
        labels = kai_searcher.get_labels(fname, kai_searcher.TRAINING_LABEL_FILE)
        for label in labels:
            brand, sku, price = ocr_ne.get_brand(label), ocr_ne.get_sku(label), ocr_ne.get_price(label)
            item_df = dated_df[
                (dated_df[KEY_BRAND] == brand) & (dated_df[KEY_SKU] == sku)]
            if item_df.shape[0] >= 1:
                expected_price = item_df[KEY_RECOMMENDED_PRICE].iloc[0]
            else:
                continue
            if price != expected_price:
                alerts.append(get_price_alert(label, brand, sku, price, expected_price))
        alert_dict[date] = alerts
        print(ctr)
        ctr += 1
        print(alerts, date)

    with open(PKL_FILE_MISPRICE, "wb") as f:
        pickle.dump(alert_dict, f)
    for date in alert_dict:
        if len(alert_dict[date]) > 0:
            print(len(alert_dict[date]), "mismatched prices during week", date)


if __name__ == '__main__':
    read_misplace()
