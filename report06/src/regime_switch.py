import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set('talk', 'white', 'dark', font_scale=1,
        rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

from numba import njit

n_trial = 1000000
epsilon = 0.05


@njit('f8(f8[:],f8[:])')
def cal_log_likelihood(y, theta):
    mu0 = theta[0]
    mu1 = theta[1]
    sigma0 = np.exp(theta[2])
    sigma1 = np.exp(theta[3])
    p11 = np.exp(theta[4]) / (1 + np.exp(theta[4]))
    p22 = np.exp(theta[5]) / (1 + np.exp(theta[5]))

    prior0 = (1 - p22) / (2 - p11 - p22)
    prior1 = (1 - p11) / (2 - p11 - p22)

    l0 = 1 / (2 * np.pi * sigma0**2)**0.5 * np.exp(-(y[0] - mu0)**2 / (2 * sigma0 ** 2))
    l1 = 1 / (2 * np.pi * sigma1**2)**0.5 * np.exp(-(y[0] - mu1)**2 / (2 * sigma1 ** 2))
    l0_p0 = l0 * prior0
    l1_p1 = l1 * prior1

    post0 = l0_p0 / (l0_p0 + l1_p1)
    post1 = l1_p1 / (l0_p0 + l1_p1)
    loglikelihood = np.log(l0_p0 + l1_p1)
    for t in range(1, len(y)):
        obs = y[t]
        l0 = 1 / (2 * np.pi * sigma0**2)**0.5 * np.exp(-(obs - mu0)**2 / (2 * sigma0 ** 2))
        l1 = 1 / (2 * np.pi * sigma1**2)**0.5 * np.exp(-(obs - mu1)**2 / (2 * sigma1 ** 2))

        prior0 = post0 * p11 + post1 * (1 - p22)
        prior1 = post0 * (1 - p11) + post1 * p22

        l0_p0 = l0 * prior0
        l1_p1 = l1 * prior1
        loglikelihood += np.log(l0_p0 + l1_p1)

        post0 = l0_p0 / (l0_p0 + l1_p1)
        post1 = l1_p1 / (l0_p0 + l1_p1)

    return loglikelihood

@njit('f8[:,:](f8[:],f8[:])')
def mcmc(y, theta_init):
    theta_pre = theta_init
    theta_history = np.zeros((n_trial, len(theta_init)))

    for i in range(n_trial):
        u = np.random.uniform(0, 1, len(theta_pre))
        theta_now = theta_pre + (2 * u - 1) * epsilon

        if theta_now[2] > theta_now[3]:
            theta_now[2] = theta_pre[2]
            theta_now[3] = theta_pre[3]

        log_likelihood_pre = cal_log_likelihood(y, theta_pre)
        log_likelihood_now = cal_log_likelihood(y, theta_now)
        likelihood_ratio = np.exp(log_likelihood_now - log_likelihood_pre)

        u = np.random.uniform(0, 1)
        if likelihood_ratio > u:
            theta_pre = theta_now

        theta_history[i] = theta_pre
    return theta_history


def get_prob(y, theta):
    p0_history = np.zeros(y.shape[0])

    mu0 = theta[0]
    mu1 = theta[1]
    sigma0 = np.exp(theta[2])
    sigma1 = np.exp(theta[3])
    p11 = np.exp(theta[4]) / (1 + np.exp(theta[4]))
    p22 = np.exp(theta[5]) / (1 + np.exp(theta[5]))

    prior0 = (1 - p22) / (2 - p11 - p22)
    prior1 = (1 - p11) / (2 - p11 - p22)

    l0 = 1 / (2 * np.pi * sigma0**2)**0.5 * np.exp(-(y[0] - mu0)**2 / (2 * sigma0 ** 2))
    l1 = 1 / (2 * np.pi * sigma1**2)**0.5 * np.exp(-(y[0] - mu1)**2 / (2 * sigma1 ** 2))
    l0_p0 = l0 * prior0
    l1_p1 = l1 * prior1

    post0 = l0_p0 / (l0_p0 + l1_p1)
    post1 = l1_p1 / (l0_p0 + l1_p1)
    p0_history[0] = post0

    loglikelihood = np.log(l0_p0 + l1_p1)
    for t in range(1, len(y)):
        obs = y[t]
        l0 = 1 / (2 * np.pi * sigma0**2)**0.5 * np.exp(-(obs - mu0)**2 / (2 * sigma0 ** 2))
        l1 = 1 / (2 * np.pi * sigma1**2)**0.5 * np.exp(-(obs - mu1)**2 / (2 * sigma1 ** 2))

        prior0 = post0 * p11 + post1 * (1 - p22)
        prior1 = post0 * (1 - p11) + post1 * p22

        l0_p0 = l0 * prior0
        l1_p1 = l1 * prior1
        loglikelihood += np.log(l0_p0 + l1_p1)

        post0 = l0_p0 / (l0_p0 + l1_p1)
        post1 = l1_p1 / (l0_p0 + l1_p1)

        p0_history[t] = post0

    return loglikelihood, p0_history


if __name__ == "__main__":
    data = pd.read_csv("../data/jal.csv")[510:]
    data["date"] = pd.to_datetime(data["date"])

    x = data["return"].values
    theta_init = np.array([0, 0, -5.0, -3.0, 5.0, 3.0])

    theta_history = mcmc(x, theta_init)
    theta_estimated = []
    for i in range(len(theta_init)):
        n, bins, patches = plt.hist(theta_history[:, i], bins=100, orientation="horizontal", color="#298ED3")
        theta_estimated.append((bins[n.argmax()] + bins[n.argmax()+1]) * 0.5)

    print(theta_estimated)

    _, p0_history = get_prob(x, theta_estimated)

    # plot regime
    sns.set('talk', 'white', 'dark', font_scale=1.1,
        rc={"lines.linewidth": 1.5, 'patch.linewidth': 0.0})

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(data["date"], data["return"], c="black")

    ax2 = ax1.twinx()
    for i in range(len(data)-1):
        ax2.axvspan(data["date"].iloc[i], data["date"].iloc[i+1], 0, p0_history[i], alpha=0.2, facecolor="#298ED3")
        ax2.axvspan(data["date"].iloc[i], data["date"].iloc[i+1], p0_history[i], 1, alpha=0.2, facecolor="#FBBD31")

    plt.xlim(data["date"].iloc[0], data["date"].iloc[-1])
    plt.savefig("../result/regime.pdf", bbox_inches="tight")