import pdb
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from DT_sampler import DT_sampler
import encode


# Given the score matrix "s" and the true class indices vector "l", the following method returns the margin values
def comp_margin(s, l):
    m, p = s.shape
    # pdb.set_trace()
    indx = np.zeros((m, p))
    indx[np.arange(m), l] = 1
    data_other_class = np.ma.array(s, mask=indx)
    data_true_class = np.ma.array(s, mask=~data_other_class.mask)
    s_other = data_other_class.compressed().reshape((m, p - 1))
    s_true = data_true_class.compressed().reshape((m, 1))
    margin = s_true - np.max(s_other, axis=1, keepdims=True)
    return margin


def nc_margin(s_cal, s_val, l):
    m_cal = comp_margin(s_cal, l)
    m_val = np.empty(s_val.shape)
    for j in range(s_val.shape[1]):
        m_val[:, j] = comp_margin(s_val, np.ones(s_val.shape[0], dtype=np.int8) * j).flatten()
    return m_cal, m_val


# Load fluorescence data
fluorescence_data = genfromtxt('../data/fluorescence.csv', delimiter=',')
X_all, y_all = fluorescence_data[:, :-1], np.int8(fluorescence_data[:, -1])
# pdb.set_trace()

# Set a fixed random seed for the training data
X_tr, X, y_tr, labels = train_test_split(X_all, y_all, train_size=100, random_state=42)

# Generate "cnf" file.
cnf_path = "../cnf/fluorescence_7_seed_42.cnf"
# sol = encode.get_solution(X_tr, y_tr, 7, 13, export_path=cnf_path, is_leaf_sampling=False) # uncomment it if required


# Run DT-sampler and return 50 samples (trees)
dt_sampler = DT_sampler(X_tr, y_tr, 7, threshold=80, cnf_path=cnf_path)
dt_sampler.run(50, method="unigen", sample_seed=0)

# Returns a 3D array containing probabilities of two classes for all 50 trees.
pred_all = dt_sampler.predict_proba_all_trees(X)

# Post-calibration using Inductive Conformal Prediction (ICP)
n = 200  # number of calibration points
alphas = np.arange(0.05, 0.5 , 0.05)  # 1-alpha is the desired coverage
errors = []
accs = []

MARGIN = True

for alpha in alphas:

    err_all = []
    acc_all = []

    for j, _ in enumerate(dt_sampler.trees):

        # Repeat experiments over 5/10 samples
        n_run = 5
        err = []
        OneAcc = []
        for run in range(n_run):

            smx = pred_all[:, :, j]

            # Split the softmax scores into calibration and validation sets (save the shuffling)
            idx = np.array([1] * n + [0] * (smx.shape[0] - n)) > 0
            np.random.shuffle(idx)
            cal_smx, val_smx = smx[idx, :], smx[~idx, :]
            cal_labels, val_labels = labels[idx], labels[~idx]

            # # 1: get conformal scores. n = calib_Y.shape[0]

            if MARGIN:
                # Conformal scores using "margin
                cal_mgx, val_mgx = nc_margin(cal_smx, val_smx, cal_labels)
                cal_scores = 1 - cal_mgx
            else:
                cal_scores = 1 - cal_smx[np.arange(n), cal_labels]

            # 2: get adjusted quantile
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            qhat = np.quantile(cal_scores, q_level, method='higher')

            if MARGIN:
                prediction_sets = val_mgx >= (1 - qhat)  # 3: form prediction sets
            else:
                prediction_sets = val_smx >= (1 - qhat)  # 3: form prediction sets

            # Coverage
            empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()

            # OneAcc
            ps = prediction_sets.sum(axis=1)
            idx_oneC = np.where(ps == 1)[0]

            if len(idx_oneC) == 0:
                one_class_accuracy = 0
                print("Tree:{}, run: {}, alpha: {}, one_class_accuracy is zero! \n".format(j, run, alpha))
            else:
                one_class_accuracy = prediction_sets[idx_oneC, val_labels[idx_oneC]].mean()

            # Stack "errors" and "accuracies" of a single tree classifiers over 5 runs
            OneAcc += [one_class_accuracy]
            err += [1 - empirical_coverage]

        # Stack "mean errors" and "mean accuracies" of a single tree classifiers over 5 runs
        err_all += [np.mean(err)]
        acc_all += [np.mean(OneAcc)]

    # Stack "mean errors" and "mean accuracies" of all tree classifiers (mean of mean values)
    errors += [np.mean(err_all)]
    accs += [np.mean(acc_all)]

if MARGIN:
    title = "calibration plot (Margin)"
    filename = "../results/ds_pre_cal_margin.pdf"
else:
    title = "calibration plot (Probability)"
    filename = "../results/ds_pre_cal_probability.pdf"

fig, ax = plt.subplots()
ax.plot(alphas, errors, label="error")
ax.plot(alphas, accs, label="OneAcc")
plt.xlabel("significance")
plt.ylabel("error")
plt.ylim([0, 1])
plt.legend()
plt.title(title)
plt.savefig(filename)
plt.show()
