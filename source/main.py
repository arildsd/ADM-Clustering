from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE_PATH = "../data/pre_processed_data.csv"
PLOT_PATH = "../data/error_plot"


def find_k():
    scores = []
    errors = []
    for k in range(1, 10):
        print(f"K = {k}")
        X = pd.read_csv(INPUT_FILE_PATH)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        errors.append(kmeans.inertia_)
        scores.append(kmeans.score(X))

    print(scores)
    print(errors)

    plt.plot(list(range(1, 10)), errors)
    plt.xlabel("Number of clusters")
    plt.ylabel("Error")
    plt.savefig(PLOT_PATH)
    plt.show()

    print(kmeans.cluster_centers_)


def interpret_clusters(k):
    X = pd.read_csv(INPUT_FILE_PATH)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

    expensive_big_property = X.iloc[3, :]
    cheap_property = X.iloc[9, :]
    small_property = X.iloc[5, :]
    no_air_conditioning = X.iloc[13, :]
    low_grade = X.iloc[52, :]
    many_kitchens = X.iloc[54, :]


    trail_df = pd.DataFrame(data=[expensive_big_property, cheap_property, small_property, no_air_conditioning,
                                  low_grade, many_kitchens])
    predictions = kmeans.predict(trail_df)
    print(predictions)



if __name__ == '__main__':
    k = 5
    interpret_clusters(k)
