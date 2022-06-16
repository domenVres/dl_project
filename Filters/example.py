import matplotlib.pyplot as plt
import cv2
import pandas as pd

from Filters.glasses import GlassesFilter
from Filters.beard import BeardFilter
from Filters.hat import HatFilter


def run_example():
    df = pd.read_csv("../Data/train/labels.csv")
    x = df.values[0]
    img = cv2.imread("../Data/train/images/" + x[0])
    points = [int(pt) for pt in x[1:]]

    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original image")

    g_filter = GlassesFilter()
    filtered_img = g_filter.place_filter(img, points)
    plt.subplot(2, 2, 2)
    plt.axis("off")
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
    plt.imshow(filtered_img)
    plt.title("Glasses filter")

    b_filter = BeardFilter()
    filtered_img = b_filter.place_filter(img, points)
    plt.subplot(2, 2, 3)
    plt.axis("off")
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
    plt.imshow(filtered_img)
    plt.title("Beard filter")

    h_filter = HatFilter()
    filtered_img = h_filter.place_filter(img, points)
    plt.subplot(2, 2, 4)
    plt.axis("off")
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
    plt.imshow(filtered_img)
    plt.title("Hat filter")

    plt.show()


if __name__ == "__main__":
    run_example()