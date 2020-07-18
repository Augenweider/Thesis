import sys
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import math
from random import sample
from em_algo import em_algo


def leave_one_participant_out(initial_theta, observations, type):
    """
    This function uses leava one out cross validation to split the train set and the test set
    calculating the goodness-of-fit for the trained model, then stores the trained
    parameters as keys and the goodness-of-fit as values in a dictionary.

    Args:
        initial_theta(list): A list of initial parameters
        observations(dict): the data set
        type(string): "-i" means the inference model "-m" means the model theory

    Return:
        g_2(dictionary): A dictionary contains the trained parameters(keys) and
        the goodness-of-fit(values) of that.
    """
    # Leave one participant out
    metric = {}
    roc_fpr = []
    roc_tpr = []
    labels = []
    precisions = {}
    accuracies = {}
    recalls = {}
    groups = list(observations.keys())
    for group, data in observations.items():
        conf_matrix = 0
        accuracy = []
        precision = []
        recall = []
        for i in range(len(data)):
            train_set = data[:i - 1] + data[i:]
            # Training
            theory = em_algo(initial_theta, train_set)
            trained_parameters = theory.run()
            if type == "-i":
                probs = inference_model(
                    trained_parameters[0], trained_parameters[1],
                    trained_parameters[2], trained_parameters[3],
                    trained_parameters[4])
            else:
                probs = model_theory(trained_parameters[0],
                                     trained_parameters[1],
                                     trained_parameters[2])
            leave, prediction, label = metrics_function(data[i], probs, type)
            labels = label
            # print metrics of the results
            # a = metrics.classification_report(leave, prediction, digits=3)
            # print(a)
            # confusion matrix
            conf_mat = metrics.confusion_matrix(leave, prediction)
            conf_matrix += conf_mat
        for i, row in enumerate(conf_matrix):
            fp = 0
            tp = 0
            tn = 0
            fn = 0
            for j, element in enumerate(row):
                if i == j:
                    tp += element
                else:
                    fn += element
            fp = sum(list(conf_matrix[:i, i]) + list(conf_matrix[i + 1:, i]))
            tn_list = list(conf_matrix[:i, :i]) + list(conf_matrix[i + 1:, i + 1:])
            for t in tn_list:
                tn += sum(t)
            acc = (tp + tn) / (tp + tn + fp + fn)
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            if not math.isnan(acc):
                accuracy.append(acc)
            else:
                accuracy.append(0.0)
            if not math.isnan(prec):
                precision.append(prec)
            else:
                precision.append(0.0)
            if not math.isnan(rec):
                recall.append(rec)
            else:
                recall.append(0.0)
        accuracies[group] = accuracy
        precisions[group] = precision
        recalls[group] = recall
    plt.figure()
    for g in groups:
        plt.plot(labels, precisions[g], label=("%s" % g))
    plt.xlabel("patterns")
    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.figure()
    for g in groups:
        plt.plot(labels, recalls[g], label="%s" % g)
    plt.xlabel("patterns")
    plt.ylabel("Recall")
    plt.legend(loc="upper right")
    plt.ylim(0, 1)
    plt.figure()
    for g in groups:
        plt.plot(labels, accuracies[g], label="%s" % g)
    plt.xlabel("patterns")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim(0, 1)

    # plt.show()



def leave_one_experiment_out(initial_theta, observations, type):
    # Leave one experiment out
    g_tests = {}
    cos = {}
    root_mean_squared_error = {}
    for group, data in observations.items():
        g_2 = 0
        cos_sim = 0
        rmse = 0
        scores = {}
        for i, observation in enumerate(data):
            score = 0
            leave_one = data[i]
            others = data[:i] + data[i + 1:]
            for experiment in others:
                score += similarity(leave_one, experiment)
            scores[i] = score

        # Select 10 experiments with the highest similarity
        score_list = sorted(list(scores.values()), reverse=True)
        chosen_scores = score_list[:10]
        chosen_expers = []
        for s in chosen_scores:
            for key, value in scores.items():
                if value == s:
                    chosen_expers.append(key)

        for i in chosen_expers:
            # Select one sample as the test set.
            test_set = data[i]
            # Others is regarded as the train set.
            train_set = data[:i - 1] + data[i:]
            # Training
            theory = em_algo(initial_theta, train_set)
            trained_parameters = theory.run()
            if type == "-i":
                probs = inference_model(
                    trained_parameters[0], trained_parameters[1],
                    trained_parameters[2], trained_parameters[3],
                    trained_parameters[4])
            else:
                probs = model_theory(trained_parameters[0],
                                     trained_parameters[1],
                                     trained_parameters[2])
            # Calculate the G-test statistic.
            g_2 += calculate_g_2(trained_parameters, test_set, type)
            # Calculate Cosine similarity
            distribution = [x / sum(test_set) for x in test_set]
            dot_product = 0
            for i in range(len(probs)):
                dot_product += probs[i] * distribution[i]

            # Calculate the magnitude of the experiment
            magnitude1 = math.sqrt(sum([i**2 for i in probs]))
            magnitude2 = math.sqrt(sum([i**2 for i in distribution]))
            cos_sim += dot_product / (magnitude1 * magnitude2)
            # Calculate RMSE
            rmse += metrics.mean_squared_error(probs, distribution, squared=False)
        g_2 = round(g_2 / 10, 3)
        cos_sim = round(cos_sim / 10, 3)
        rmse = round(rmse / 10, 3)
        g_tests[group] = g_2
        cos[group] = cos_sim
        root_mean_squared_error[group] = rmse

    return g_tests, cos, root_mean_squared_error





def similarity(experiment1, experiment2):
    """
    Return a score that represents the distribution similarity of both experiments
    The similarity is computed by the cosine similarity and a bias value.

    Parameters:
        experiment1(list): The list contains the number of participants of each patterns in a experiment
        experiment2(list): The list contains the number of participants of each patterns in a experiment
    """
    # Calculate the dot produt of both experimental distribution
    dot_product = 0
    for i in range(len(experiment1)):
        dot_product += experiment1[i] * experiment2[i]

    # Calculate the magnitude of the experiment
    magnitude1 = math.sqrt(sum([i**2 for i in experiment1]))
    magnitude2 = math.sqrt(sum([i**2 for i in experiment2]))

    cos = dot_product / magnitude1 * magnitude2

    # Calculate the bias value using the difference of sums
    # of participant numbers of both experiments
    sum1 = sum(experiment1)
    sum2 = sum(experiment2)
    if sum1 >= sum2:
        diff = sum2 - sum1
    else:
        diff = sum1 - sum2

    # If the difference is 20, the bias value is 0.
    # If the difference is larger than 20, the bias value is negative.
    # If the difference is smaller than 20, the bias value is positive.
    if diff != 0:
        domain = abs(diff) / 20
        bias = - math.log(domain)
    else:
        bias = 30
    # Compute the sum between cosine similarity multiplying by 100 and bias
    # the higher cosine similarity the experiment has, the higher score it gets
    # the lower bias value the experiment has, the higher score it obtains
    score = cos * 100 + bias
    return score


def metrics_function(experiment, probs, type):
    prediction = []
    leave = []
    result = {}
    if type == "-i":
        result[0] = "0000"
        result[1] = "0001"
        result[2] = "0010"
        result[3] = "0011"

        result[4] = "0100"
        result[5] = "0101"
        result[6] = "0110"
        result[7] = "0111"

        result[8] = "1000"
        result[9] = "1001"
        result[10] = "1010"
        result[11] = "1011"

        result[12] = "1100"
        result[13] = "1101"
        result[14] = "1110"
        result[15] = "1111"

    else:
        result[0] = "1000"
        result[2] = "1010"
        result[1] = "1001"
        result[3] = "1011"

    for index, number in enumerate(experiment):
        for i in range(number):
            leave.append(result[index])

    p = np.array(probs)
    for i in range(len(leave)):
        prediction.append(result[np.random.choice([int(x) for x in range(len(result))], p = p.ravel())])


    return leave, prediction, list(result.values())



def bootstrap(initial_theta, observations, type):
    """
    This function uses the boostraping method to split the train set and the test set
    calculating the goodness-of-fit for the trained model, and then stores the trained
    parameters as keys and the goodness-of-fit as values in a dictionary.

    Args:
        initial_theta(list): A list of initial parameters
        observations(list): the data set
        type(string): "-i" means the inference model "-m" means the model theory

    Return:
        g_2(dictionary): A dictionary contains the trained parameters(keys) and
        the goodness-of-fit(values) of that.
    """
    g_2 = {}
    rmse = {}
    cos_sim = {}
    # Choose the training samples with replacements
    for group, data in observations.items():
        n = 0
        test_num = 0
        root_mean_squared_error = 0
        cos = 0
        mean_of_g_2 = 0
        while n < 10:
            n += 1
            train_set = []
            length = len(data)
            for i in range(length):
                train_set.append([x for x in sample(data, 1)[0]])
            temp = train_set
            # All other samples not be chosen is in the test set.
            for train in train_set:
                i = temp.index(train)
                test_set = temp[:i - 1] + temp[i:]
                temp = test_set
            test_num += len(test_set)
            # Training
            theory = em_algo(initial_theta, train_set)
            trained_parameters = theory.run()
            if type == "-i":
                probs = inference_model(
                    trained_parameters[0], trained_parameters[1],
                    trained_parameters[2], trained_parameters[3],
                    trained_parameters[4])
            else:
                probs = model_theory(trained_parameters[0],
                                     trained_parameters[1],
                                     trained_parameters[2])
            # Computing g2
            for test in test_set:
                mean_of_g_2 += calculate_g_2(trained_parameters, test, type)
            # Compute rmse and Cosine Similarity
            for test in test_set:
                distribution = [x / sum(test) for x in test]
                dot_product = 0
                for i in range(len(probs)):
                    dot_product += probs[i] * distribution[i]

                # Calculate the magnitude of the experiment
                magnitude1 = math.sqrt(sum([i**2 for i in probs]))
                magnitude2 = math.sqrt(sum([i**2 for i in distribution]))
                cos += dot_product / (magnitude1 * magnitude2)
                root_mean_squared_error += metrics.mean_squared_error(probs, distribution, squared=False)
        root_mean_squared_error = root_mean_squared_error / test_num
        cos = cos / test_num
        mean_of_g_2 = mean_of_g_2 / test_num
        rmse[group] = round(root_mean_squared_error, 3)
        cos_sim[group] = round(cos, 3)
        g_2[group] = round(mean_of_g_2, 3)

    return g_2, cos_sim, rmse



def calculate_g_2(params, test, type):
    """
    Calculate the goodness-of-fit of a model using samples in the test set.

    Args:
        params(list): the trained parameters
        test(list): the test set
        type(string): "-i" means the inference model "-m" means the model theory
    """
    total = sum(test)
    distribution = [x / total for x in test]
    if type == "-i":
        new_distribution = inference_model(
            params[0], params[1], params[2], params[3], params[4])
    elif type == "-m":
        new_distribution = model_theory(params[0], params[1], params[2])

    g = 0
    for (i, j) in enumerate(new_distribution):
        if j != 0 and test[i] != 0:
            g += 2 * test[i] * math.log(test[i] / (total * j))

    return g


def inference_model(c, d, x, s, i):
    """
    Calculate the probabilities of all categories in the inference model
    using the given parameters.
    """
    probs = []
    # 0000
    probs.append(0)
    # 0001
    p_0001 = c * (1 - d) * (1 - s) * i
    probs.append(p_0001)
    # 0010
    p_0010 = c * (1 - d) * s * i
    probs.append(p_0010)
    # 0011
    p_0011 = (1 - c) * (1 - x) * (1 - d) * i
    probs.append(p_0011)
    # 0100
    p_0101 = c * d * (1 - s) * i
    probs.append(p_0101)
    # 0101
    p_0101 = (1 - c) * x * (1 - s) * i
    probs.append(p_0101)
    # 0110
    p_0110 = c * d * (1 - s) * (1 - i)
    p_0110 += c * (1 - d) * s * (1 - i)
    probs.append(p_0110)
    # 0111
    probs.append(0)
    # 1000
    p_1000 = c * d * s * i
    probs.append(p_1000)
    # 1001
    p_1001 = c * d * s * (1 - i)
    p_1001 += c * (1 - d) * (1 - s) * (1 - i)
    probs.append(p_1001)
    # 1010
    p_1010 = (1 - c) * x * s * i
    probs.append(p_1010)
    # 1011
    probs.append(0)
    # 1100
    p_1100 = (1 - c) * (1 - x) * d * i
    probs.append(p_1100)
    # 1101
    probs.append(0)
    # 1110
    probs.append(0)
    # 1111
    p_1111 = (1 - c) * x * s * (1 - i)
    p_1111 += (1 - c) * x * (1 - s) * (1 - i)
    p_1111 += (1 - c) * (1 - x) * d * (1 - i)
    p_1111 += (1 - c) * (1 - x) * (1 - d) * (1 - i)
    probs.append(p_1111)

    return probs


def model_theory(c, e, f):
    """
    Calculate the probabilities of all categories in the model theory
    using the given parameters.
    """
    probs = []
    # 1000
    p_1000 = (1 - c) * (1 - e)
    probs.append(p_1000)
    # 1010
    p_1010 = c * (1 - e)
    p_1010 += (1 - c) * e * (1 - f)
    probs.append(p_1010)
    # 1001
    p_1001 = (1 - c) * e * f
    p_1001 += c * e * f
    probs.append(p_1001)
    # 1011
    p_1011 = c * e * (1 - f)
    probs.append(p_1011)

    return probs


if __name__ == '__main__':
    if ((len(sys.argv) != 5 and len(sys.argv) != 7) or
        (sys.argv[1] == '-i' and len(sys.argv[2:]) != 5) or
            (sys.argv[1] == '-m' and len(sys.argv[2:]) != 3)):
        print(
            """Usage: python3 main.py [-i | -m] [parameter1 [parameter2 ...]]
optional arguments:
  -i, --the inference model with five parameters
  -m, --the model theory with three parameters""")
        exit(0)
    initial_theta = []
    obs = {}
    for parameter in sys.argv[2:]:
        initial_theta.append(float(parameter))
    if sys.argv[1] == "-i":
        file = pd.ExcelFile('Studies16patterns.xlsx')
        observations = pd.read_excel(file)
        obs["abstract"] = observations[observations["Content"] == "abstract"]
        obs["deontic"] = observations[observations["Content"] == "deontic"]
        obs["everyday"] = observations[observations["Content"] == "generalization"]
        for key, value in obs.items():
            obs[key] = value.iloc[:, -16:].astype('int') + 1
            obs[key] = obs[key].values.tolist()
    elif sys.argv[1] == "-m":
        file = pd.ExcelFile('Studies4patterns.xlsx')
        observations = pd.read_excel(file)
        obs["abstract"] = observations[observations["Content"] == "abstract"]
        obs["deontic"] = observations[observations["Content"] == "deontic"]
        obs["everyday"] = observations[observations["Content"] == "generalization"]
        for key, value in obs.items():
            obs[key] = value.iloc[:, -5:-1].astype('int') + 1
            obs[key] = obs[key].values.tolist()
    leave_one_participant_out(initial_theta, obs, sys.argv[1])
    g_2_loo, cos_min_loo, rmse_loo = leave_one_experiment_out(initial_theta, obs, sys.argv[1])
    g_2_bootstrap, cos_min_bootstrap, rmse_bootstrap = bootstrap(initial_theta, obs, sys.argv[1])
    groups = list(g_2_loo.keys())
    plt.figure()
    plt.plot(groups, list(g_2_loo.values()), label="loo-g2")
    plt.plot(groups, list(g_2_bootstrap.values()), label="bootstrap-g2")
    plt.xlabel("Generalizations")
    plt.ylabel("G-test Statistic")
    plt.ylim(0, 400)
    plt.legend(loc="upper right")
    plt.figure()
    plt.plot(groups, list(cos_min_loo.values()), label="loo-cosine similarity")
    plt.plot(groups, list(cos_min_bootstrap.values()), label="bootstrap-cosine similarity")
    plt.xlabel("Generalizations")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.figure()
    plt.plot(groups, list(rmse_loo.values()), label="loo-rmse")
    plt.plot(groups, list(rmse_bootstrap.values()), label="bootstrap-rmse")
    plt.xlabel("Generalizations")
    plt.ylabel("RMSE")
    plt.ylim(0, 0.5)
    plt.legend(loc="lower right")

    plt.show()
