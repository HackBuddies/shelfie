import datetime
import pandas as pd
import kai_searcher
import ocr_manager

KEY_DATE = "Date"
KEY_BRAND = "Brand"
KEY_SKU = "SKU"
KEY_RECOMMENDED_PRICE = "Recommended Price"

ALERT_PRICE = 1
ALERT_MISMATCH = 2


def get_filename_at_date(date_str):
    date = datetime.datetime.strptime(date_str, "%d/%m/%Y").date()
    return "images/Supermarket_" + str(date) + ".jpg"


def get_price_alert(label, actual_price, expected_price):
    return [ALERT_PRICE, label, actual_price, expected_price]


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
            brand, sku, price = ocr_manager.get_text(label)
            expected_price = dated_df[
                (dated_df[KEY_BRAND] == brand) & (dated_df[KEY_SKU] == sku)][KEY_RECOMMENDED_PRICE].iloc[0]
            if price != expected_price:
                alerts.append(get_price_alert(label, price, expected_price))
        alert_dict[date] = alerts
        print(ctr)
        ctr += 1
        print(len(alerts), date)


if __name__ == '__main__':
    main()
