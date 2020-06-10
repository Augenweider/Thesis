import sys
import pandas as pd
import math
from random import sample
from em_algo import em_algo

def leave_one_out(initial_theta, observations, type):
    """
    This function uses leava one out cross validation to split the train set and the test set
    calculating the goodness-of-fit for the trained model, then stores the trained
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
    for i in range(len(observations)):
        # Select one sample as the test set.
        test_set = observations[i]
        # Others is regarded as the train set.
        train_set = observations[:i - 1] + observations[i:]
        # Training
        theory = em_algo(initial_theta, train_set)
        trained_parameters = theory.run()

        # Calculate the G-test statistic.
        result = calculate_g_2(trained_parameters, test_set, type)
        g_2[str(trained_parameters)] = result
    return g_2


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
    # Operating 10 times
    n = 10
    length = len(observations)
    while n > 0:
        train_set = []
        # Choose the training samples with replacements
        for i in range(length):
            train_set.append([x for x in sample(observations, 1)[0]])
        temp = train_set
        # All other samples not be chosen is in the train set.
        for train in train_set:
            i = temp.index(train)
            test_set = temp[:i - 1] + temp[i:]
            temp = test_set
        # Training
        theory = em_algo(initial_theta, train_set)
        trained_parameters = theory.run()
        # Computing the standard deviation.
        mean_of_g_2 = 0
        var_of_g_2 = 0
        for test in test_set:
            mean_of_g_2 += (1 / len(test_set)) * calculate_g_2(trained_parameters, test, type)
        for test in test_set:
            var_of_g_2 += (1 / len(test_set)) * ((calculate_g_2(trained_parameters, test, type) - mean_of_g_2) ** 2)
        g_2[str(trained_parameters)] = (round(mean_of_g_2, 2), round(math.sqrt(var_of_g_2), 2))
        n -= 1

    return g_2


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
        new_distribution = inference_model(params[0], params[1], params[2], params[3], params[4])
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
    p_0101 = (1 - c) * x * (1 - s) * i
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

if __name__=='__main__':
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
    for parameter in sys.argv[2:]:
        initial_theta.append(float(parameter))
    if sys.argv[1] == "-i":
        file = pd.ExcelFile('Studies16patterns.xlsx')
        observations = pd.read_excel(file)
        observations = observations.iloc[:, -16:].astype('int') + 1
        observations = observations.values.tolist()
    elif sys.argv[1] == "-m":
        file = pd.ExcelFile('Studies4patterns.xlsx')
        observations = pd.read_excel(file)
        observations = observations.iloc[:, -5:-1].astype('int') + 1
        observations = observations.values.tolist()

    result1 = leave_one_out(initial_theta, observations, sys.argv[1])
    result2 = bootstrap(initial_theta, observations, sys.argv[1])

    data1 = []
    for (key, value) in result1.items():
        data1.append(eval(key) + [value])
    data2 = []
    for (key, value) in result2.items():
        data2.append(eval(key) + [value])
    if sys.argv[1] == '-i':
        df_loo = pd.DataFrame(data1, columns=['theta_c', 'theta_d', 'theta_x', 'theta_s', 'theta_i', 'goodness_of_fit'])
        df_boo = pd.DataFrame(data2, columns=['theta_c', 'theta_d', 'theta_x', 'theta_s', 'theta_i', 'goodness_of_fit(mean, SD)'])
    elif sys.argv[1] == '-m':
        df_loo = pd.DataFrame(data1, columns=['theta_c', 'theta_e', 'theta_f', 'goodness_of_fit'])
        df_boo = pd.DataFrame(data2, columns=['theta_c', 'theta_e', 'theta_f', 'goodness_of_fit(mean, SD)'])

    print(df_loo.to_latex(index=False))
    print(df_boo.to_latex(index=False))
