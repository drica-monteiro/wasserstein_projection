import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()
import sys
sys.path.append('../')
from src.mean_stress import stress_twomeans


# global style for all figures
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 300
})


def plot_brightness_stress_all_classes(taus, portions, class_names):
    """
    Plot portion of predicted class c vs tau
    """

    num_classes = portions.shape[1]

    rows = int(np.ceil(num_classes / 2))
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(6, 2.8*rows),
                             constrained_layout=True)

    axes = axes.flatten()

    zero_idx = np.where(taus == 0)[0][0]

    for c in range(num_classes):

        ax = axes[c]

        ax.plot(taus, portions[:, c])
        ax.plot(taus[zero_idx], portions[zero_idx, c],
                marker="*", color="red")

        ax.set_title(class_names[c])

        if c % cols == 0:
            ax.set_ylabel("Portion of predictions")

        if c // cols == rows-1:
            ax.set_xlabel(r"$\tau$")
        else:
            ax.set_xticks([])

    plt.show()


def plot_mean_stress(projected_means, values, base_mean, ylabel="PP1s"):
    
    plt.figure(figsize=(5,4))

    plt.plot(projected_means, values)

    # find where tau = 0
    idx = np.argmin(np.abs(projected_means - base_mean))

    plt.plot(
        projected_means[idx],
        values[idx],
        marker="*",
        color="red",
        markersize=8,
        label="Original data"
    )

    #plt.xlabel(r"average feature value")
    plt.ylabel(ylabel)

    #plt.legend()
    plt.grid(alpha=0.3)

    plt.show()

def plot_twomeans(
        x_test,
        col_name1,
        col_name2,
        taus,
        model,
        alpha
        ):

    def predict_fn(df_shift):

        #X = df_shift.drop(columns="class")
        preds = model.predict(df_shift)

        return preds
    
    list_t1, list_t2, m1, m2 = stress_twomeans(
        x_test,
        col_name1,
        col_name2,
        taus,
        alpha
    )

    portions = []

    for t1, t2 in zip(list_t1, list_t2):

        df_shift = x_test.copy()

        df_shift[col_name1] += (t1 - m1)
        df_shift[col_name2] += (t2 - m2)

        preds = predict_fn(df_shift)

        portion = (preds == 1).mean()

        portions.append(portion)

    portions = np.array(portions)

    fig, ax = plt.subplots(figsize=(5,3), dpi=400)

    sc = ax.scatter(
        list_t1,
        list_t2,
        c=portions,
        cmap="viridis",
        edgecolor="k",
        s=60
    )

    # highlight original dataset
    # ax.scatter(
    #     m1,
    #     m2,
    #     marker="*",
    #     color="red",
    #     s=8,
    #     label="Original data"
    # )

    fig.colorbar(sc, ax=ax, label="Portion of 1s")

    ax.set_xlabel(f"Average {col_name1}")
    ax.set_ylabel(f"Average {col_name2}")

    #ax.legend()

    plt.show()

def plot_di(projected_means, dis, lower, upper, col_name, taus):

    plt.figure(figsize=(5,4), dpi=300)

    plt.plot(projected_means, dis, color="k")

    plt.fill_between(
        projected_means,
        lower,
        upper,
        color="gray",
        alpha=0.25
    )

    taus = np.asarray(taus)
    zero_idx = np.argmin(np.abs(taus))

    plt.plot(
        projected_means[zero_idx],
        dis[zero_idx],
        marker="*",
        markersize=10,
        color="red"
    )

    plt.xlabel(f"Average {col_name}")
    plt.ylabel("DI")

    plt.tight_layout()
    plt.show()

def save_figure(name):
    """
    Save figure in figures folder
    """

    plt.savefig(f"figures/{name}.pdf", bbox_inches="tight")