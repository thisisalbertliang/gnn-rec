from matplotlib import pyplot as plt


def plot():
    plt.bar(
        x=["SVD", "NMF", "SlopeOne", "KNN"],
        height=[0.934, 0.963, 0.946, 0.980]
    )

    plt.xlabel("Baseline")
    plt.ylabel("Mean RMSE")
    plt.title("Mean RMSE v.s. Baseline")
    plt.savefig("figs/rmse-baselines.png")
    plt.clf()

    plt.bar(
        x=["SVD", "NMF", "SlopeOne", "KNN"],
        height=[0.737, 0.758, 0.743, 0.774]
    )

    # plt.savefig("rmse.png")
    plt.xlabel("Baseline")
    plt.ylabel("Mean MAE")
    plt.title("Mean MAE v.s. Baseline")
    plt.savefig("figs/mae-baselines.png")


if __name__ == "__main__":
    plot()
