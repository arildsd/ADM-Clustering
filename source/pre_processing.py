import pandas as pd
import numpy as np

# Global constants
RAW_INPUT_FILE_PATH = "../data/DC_Properties.csv"
OUTPUT_FILE_PATH = "../data/pre_processed_data.csv"
SELECTED_FEATURES = ["BATHRM", "HF_BATHRM", "AC", "NUM_UNITS", "ROOMS", "BEDRM", "STORIES",
                     "PRICE", "GBA", "BLDG_NUM", "GRADE", "CNDTN", "KITCHENS", "FIREPLACES"]

def text_rating_to_numeric(text):
    text = text.lower()
    if text == "excellent":
        return 7
    elif text == "very good":
        return 6
    elif text == "good":
        return 5
    elif text == "fair":
        return 4
    elif text == "average":
        return 3
    elif text == "poor":
        return 2
    else:
        return 1


def normalize(df):
    # Normalize every column
    for key in df.keys():
        column = df[key]
        standard_deviation = np.std(column)
        mean = np.mean(column)
        df[key] = column.apply(lambda val: (val - mean)/standard_deviation)

    return df


def pre_process():
    df = pd.read_csv(RAW_INPUT_FILE_PATH)

    # Select features
    df = df[SELECTED_FEATURES]

    # Drop rows with nan values
    df = df.dropna()

    df["GRADE"] = df["GRADE"].apply(text_rating_to_numeric)
    df["CNDTN"] = df["CNDTN"].apply(text_rating_to_numeric)

    df["AC"] = df["AC"].apply(lambda x: 1 if x == "Y" else -1)

    df = normalize(df)
    return df




if __name__ == '__main__':
    df = pre_process()
    df.to_csv(OUTPUT_FILE_PATH, index=False)







