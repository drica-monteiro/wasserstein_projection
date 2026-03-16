import sys
sys.path.append('../')
from src.stress_base import compute_quantile_bounds, run_stress_experiment, tau_to_t
import numpy as np
import matplotlib.pyplot as plt
from LEFkit.bias_measure.bias_measure_fcts import Cpt_DI
import seaborn as sns
sns.set_theme()

def mean_stress_experiment(
    model,
    df,
    feature,
    taus,
    alpha=0.05
):

    values = df[feature].values

    m, q_low, q_high = compute_quantile_bounds(
        values, alpha
    )

    def transform_fn(delta):

        df_new = df.copy()
        df_new[feature] += delta

        return df_new


    def predict_fn(df_shift):

        #X = df_shift.drop(columns="class")
        preds = model.predict(df_shift)

        return preds


    return run_stress_experiment(
        taus,
        m,
        q_low,
        q_high,
        transform_fn,
        predict_fn,
        num_classes=2
    )


def stress_twomeans(x_test, col_name1, col_name2, taus, alpha=0.05):
    """
    Compute projected means for two stressed features.

    Returns:
        list_t1 : projected means for col_name1
        list_t2 : projected means for col_name2
        m1, m2 : baseline means
    """



    values1 = x_test[col_name1].values
    values2 = x_test[col_name2].values

    m1, q_low1, q_high1 = compute_quantile_bounds(values1, alpha)
    m2, q_low2, q_high2 = compute_quantile_bounds(values2, alpha)

    list_t1 = []
    list_t2 = []

    for tau in taus:

        t1 = tau_to_t(tau, m1, q_low1, q_high1)
        t2 = tau_to_t(tau, m2, q_low2, q_high2)

        list_t1.append(t1)
        list_t2.append(t2)

    return np.array(list_t1), np.array(list_t2), m1, m2

def plot_multiplemean(
        x_test,
        taus,
        model,
        alpha,
        #predict_fn,
        col_names=None):
    """
    Plot tau vs portion of positive classifications
    when stressing multiple feature means.

    Inputs
    ------
    x_test : dataframe
    taus : array of tau values
    alpha : quantile parameter
    predict_fn : function df -> predictions
    col_names : list of columns to stress
    """

    def predict_fn(df_shift):

        #X = df_shift.drop(columns="class")
        preds = model.predict(df_shift)

        return preds

    if col_names is None:
        #col_names = ['education-num', 'age', 'capital-gain', 'capital-loss', 'hours-per-week']
        col_names = ['Age', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']
    fig, ax = plt.subplots(figsize=(6,4), dpi=300)

    for col_name in col_names:

        results, _ = mean_stress_experiment(
        model,
        df=x_test,
        feature=col_name,
        taus=taus,
        alpha=alpha
    )

        portions = results[:,1]   # positive class portion

        ax.plot(
            taus,
            portions,
            label=col_name
        )

        # baseline marker (tau = 0)
        #zero_idx = np.where(taus == 0)[0][0]

        # ax.plot(
        #     taus[zero_idx],
        #     portions[zero_idx],
        #     marker="*",
        #     color="red",
        #     markersize=8
        # )

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Portion of 1s")

    ax.legend(
        loc='best',
        frameon=True,
        fontsize='small'
    )

    fig.tight_layout()

    return fig


def di_stress_experiment(
        x_test,
        col_name,
        taus,
        model,
        alpha,
        #predict_fn,
        S):


    values = x_test[col_name].values
    m, q_low, q_high = compute_quantile_bounds(values, alpha)

    dis = []
    lower = []
    upper = []
    projected_means = []

    for tau in taus:

        t_tau = tau_to_t(tau, m, q_low, q_high)
        delta = t_tau - m

        df_shift = x_test.copy()
        df_shift[col_name] += delta

        def predict_fn(df_shift):
            #X = df_shift.drop(columns="class")
            preds = model.predict(df_shift)

            return preds
    
        preds = predict_fn(df_shift)

        di, err = Cpt_DI(
            S,
            preds,
            w=1,
            alpha=alpha,
            boxplot=False,
            wedge=False
        )

        dis.append(di)
        lower.append(err[0])
        upper.append(err[1])
        projected_means.append(t_tau)

    return (
        np.array(projected_means),
        np.array(dis),
        np.array(lower),
        np.array(upper)
    )