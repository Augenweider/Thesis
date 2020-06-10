def the_model_theory(c, e, f):
    """
    Return the predicting result of an individual's selection on Wason selection
    task using the model theory.

    Parameters:
        c(float): The parameter c quantifies the probability with which the scan
                  model is in converse direction too. The probabilities of scaning
                  model from p to q is (1 - c).
        e(float): The parameter e quantifies the probability of the partial insight.
                  In contrast, the no insight one is (1 - e).
        f(float): The parameter f quantifies the probability of the complete insight.
                  The probability without further insight is (1 - f).

    Return:
        selection(string): the individual's card-selection.
    """
    probabilities = {}
    # 1000
    p_1000 = (1 - c) * (1 - e)
    probabilities["1000"] = p_1000
    # 1010
    p_1010 = c * (1 - e)
    p_1010 += (1 - c) * e * (1 - f)
    probabilities["1010"] = p_1010
    # 1001
    p_1001 = (1 - c) * e * f
    p_1001 += c * e * f
    probabilities["1001"] = p_1001
    # 1011
    p_1011 = c * e * (1 - f)
    probabilities["1011"] = p_1011

    selection = sorted([(key, -value) for key, value in probabilities.items()],
                       key=lambda x: x[1])[0][0]

    return selection
