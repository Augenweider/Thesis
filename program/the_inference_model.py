def inference_model(c, d, x, s, i):
    """
    Return the predicting result of an individual's selection on Wason selection
    task using the inference model.

    Parameters:
        c(float): The parameter c quantifies the probability of a conditional
                  interpretation. By contrast, the probability of a biconditional
                  one is (1 - c).
        d(float): The parameter d quantifies the probability with which the
                  the rule id seen as one warranting backward inferences rather
                  than a reverse one (1 - d).
        x(float): The parameter x quantifies the probability of the bidirectional
                  interpretation. The biconditional rule is conversely interpreted
                  as a case distinction with probability (1 - x).
        s(float): The parameter s quantifies the probability with which the
                  antecedent condition in the perceived direction is seen as
                  sufficient. It is seen as necessary with probability (1 - s).
        i(float): The parameter i quantifies the probability with which the avaliable
                  inferences are applied only to the visible sides of the cards.
                  Conversely, individual also consider the invisble side and apply
                  the avaliable infernece to the invisible sides.

    Return:
        selection(string): the individual's card-selection.
    """
    probabilities = {}
    # 0001
    p_0001 = c * (1 - d) * (1 - s) * i
    probabilities["0001"] = p_0001
    # 0010
    p_0010 = c * (1 - d) * s * i
    probabilities["0010"] = p_0010
    # 0011
    p_0011 = (1 - c) * (1 - x) * (1 - d) * i
    probabilities["0011"] = p_0011
    # 0100
    p_0100 = c * d * (1 - s) * i
    probabilities["0100"] = p_0100
    # 0101
    p_0101 = (1 - c) * x * (1 - s) * i
    probabilities["0101"] = p_0101
    # 0110
    p_0110 = c * d * (1 - s) * (1 - i)
    p_0110 += c * (1 - d) * s * (1 - i)
    probabilities["0110"] = p_0110
    # 1000
    p_1000 = c * d * s * i
    probabilities["1000"] = p_1000
    # 1001
    p_1001 = c * d * s * (1 - i)
    p_1001 += c * (1 - d) * (1 - s) * (1 - i)
    probabilities["1001"] = p_1001
    #1010
    p_1010 = (1 - c) * x * s * i
    probabilities["1010"] = p_1010
    # 1100
    p_1100 = (1 - c) * (1 - x) * d * i
    probabilities["1100"] = p_1100
    # 1111
    p_1111 = (1 - c) * x * s * (1 - i)
    p_1111 += (1 - c) * x * (1 - s) * (1 - i)
    p_1111 += (1 - c) * (1 - x) * d * (1 - i)
    p_1111 += (1 - c) * (1 - x) * (1 - d) * (1 - i)
    probabilities["1111"] = p_1111

    selection = sorted([(key, -value) for key, value in probabilities.items()],
                       key=lambda x: x[1])[0][0]
    return selection
