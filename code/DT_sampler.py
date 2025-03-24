import pdb

import numpy as np
import pandas as pd
import time
import copy
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from pyunigen import Sampler
import os
from decode import get_sample_solution
from decision_tree import DecisionTree
from encode import get_solution


class DT_sampler:
    def __init__(self, X, y, node_n, threshold, cnf_path, is_leaf_sampling=False):
        self.X = X
        self.y = y
        self.node_n = node_n
        self.threshold = threshold
        self.cnf_path = cnf_path
        self.trees = []
        self.is_leaf_sampling = is_leaf_sampling
        self.sampled = False
        if get_solution(X, y, node_n, threshold, None, is_leaf_sampling) == None:
            print("No satisfied decision trees")
            exit()
        get_solution(X, y, node_n, threshold, cnf_path, is_leaf_sampling)

    def run(self, sample_n, method, sample_seed=0):
        if method == "unigen":
            sample_clauses = []
            with open(self.cnf_path) as f:
                for line in f.readlines():
                    if line[:5] == "c ind":
                        sampling_set = set([int(i) for i in line.strip()[5:-1].split()])
                    elif line[0] == "c":
                        name_to_index = eval(line[2:].strip())
                    elif line[:5] == "p cnf":
                        n_value, n_constraints = [int(i) for i in line[5:].strip().split()]
                    else:
                        sample_clauses.append([int(i) for i in line.strip()[:-1].split()])

            sampler = Sampler(seed=sample_seed)
            for clause in sample_clauses:
                sampler.add_clause(clause)

            start = time.time()
            raw_sample = sampler.sample(num=sample_n, sampling_set=sampling_set)
            end = time.time()
            samples = raw_sample[2]

            # print("time:",end-start)
            for index, sample in enumerate(samples):
                name_to_index_copy = copy.copy(name_to_index)
                for v in sample:
                    i = abs(v)
                    name_to_index_copy[list(name_to_index.keys())[i - 1]] = 1 if v > 0 else -1

                tree = DecisionTree()
                solution = get_sample_solution(name_to_index_copy)
                tree.fit_solution(solution, self.X, self.y, self.is_leaf_sampling)
                self.trees.append(tree)

        elif method == None:
            pre_sol = []
            for i in range(sample_n):
                tree = DecisionTree()
                solution, m = get_solution(self.X, self.y, self.node_n, self.threshold, None, self.is_leaf_sampling,
                                           seed=sample_seed, pre_sol=pre_sol)
                tree.fit_solution(solution, self.X, self.y, self.is_leaf_sampling)
                self.trees.append(tree)
                pre_sol.append(m)

        # elif method == "quicksampler":

        else:
            print("No sampling method named \"%s\"" % method)
            exit()

        self.sampled = True

    '''
    Returns class probabilities of all examples in "X_test" for each tree in a 3-D array called "pred_all" of shape
    (n X 2 X m), where n:# of test examples, 2: # of classes (binary classification), m: # of trees. In pred_all, the
    class order is - 0th col : class-0, 1th col: class 1.
    '''

    def predict_proba_all_trees(self, X_test):
        if not self.sampled:
            print("Please run sampling first!")
            exit()

        pred_all = np.empty((len(X_test), 2, len(self.trees)))

        for i, example in enumerate(X_test):

            for j, tree in enumerate(self.trees):

                pred, prob = tree.predict(example)

                if pred == 0:
                    pred_all[i, 0, j] = prob
                    pred_all[i, 1, j] = (1 - prob)

                if pred == 1:
                    pred_all[i, 0, j] = (1 - prob)
                    pred_all[i, 1, j] = prob
                # pdb.set_trace()
        return pred_all

    def predict_proba_ensemble(self, X_test):
        # pdb.set_trace()

        if self.sampled is False:
            print("Please run sampling first!")
            exit()

        proba = np.empty(shape=(len(X_test), 2))
        # pdb.set_trace()
        for i, example in enumerate(X_test):
            prob_0 = 0
            prob_1 = 0
            for tree in self.trees:
                pred, prob = tree.predict(example)
                if pred == 0:
                    prob_0 += prob
                    prob_1 += (1 - prob)
                if pred == 1:
                    prob_0 += (1 - prob)
                    prob_1 += prob
            # pdb.set_trace()
            proba[i, 0] = prob_0 / len(self.trees)
            proba[i, 1] = prob_1 / len(self.trees)

        return proba

    def predict(self, X_test):
        # pdb.set_trace()
        print("#trees: {}\n".format(len(self.trees)))
        if self.sampled == False:
            print("Please run sampling first!")
            exit()
        y_predicted = np.empty(shape=len(X_test), dtype=np.int8)
        for i, example in enumerate(X_test):
            prob_0 = 0
            prob_1 = 0
            for tree in self.trees:
                pred, prob = tree.predict(example)
                if pred == 0:
                    prob_0 += prob
                    prob_1 += (1 - prob)
                if pred == 1:
                    prob_0 += (1 - prob)
                    prob_1 += prob
            # pdb.set_trace()
            y_predicted[i] = 0 if prob_0 > prob_1 else 1
        return y_predicted

    def feature_prob(self):
        if self.sampled == False:
            print("Please run sampling first!")
            exit()
        f_prob = {}
        for tree in self.trees:
            for k, node in tree.nodes.items():
                f_prob[node.x] = f_prob.get(node.x, 0) + 1

        f_count = sum(f_prob.values())
        for f in f_prob:
            f_prob[f] = f_prob[f] / f_count
        return f_prob
