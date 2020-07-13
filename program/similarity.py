import math

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
    for i in experiment1:
        for j in experiment2:
            dot_product += i * j

    # Calculate the magnitude of the experiment
    magnitude1 = math.sqrt(sum([i**2 for i in experiment1]))
    magnitude2 = math.sqrt(sum([i**2 for i in experiment2]))

    cosine_similarity = dot_product / magnitude1 * magnitude2

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
    bias = - math.log(diff / 20)
    # Compute the sum between cosine similarity multiplying by 100 and bias
    # the higher cosine similarity the experiment has, the higher score it gets
    # the lower bias value the experiment has, the higher score it obtains
    score = cosine_similarity * 100 + bias
    return score


def Leave_one_experiment(observations):
    """
    This function compute sum of the similarities between one experiment and all
    others using the above introduced similarity. Then the function stores these
    scores in a dictionary and returns it.

    Parameters:
        observations(list): the list of observation
    """
    scores = {}
    for i, observation in enumerate(observations):
        score = 0
        leave_one = observations[i]
        others = observations[:i] + observations[i + 1:]
        for experiment in others:
            score += similarity(leave_one, experiment)
        scores[i] = score

    return scores
