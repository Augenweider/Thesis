def the_inference_model(params):
    """
    Return the predicting result of an individual's selection on Wason selection
    task using the inference model.

    Parameters:
        params(list): the list of assumptions

    Return:
        selection(string): the individual's card-selection.
    """
    rules = {}
    result = []
    selection = []
    turn = []
    # Conditional rules can be understood as forward inference or backward inference.
    rules["conditional"] = ("forward", "backward")
    # Biconditional rules can be understood as conjunctions of conditional premises
    # in bidirectional or case distinction.
    rules["biconditional"] = ("bidirectional", "case_distinction")
    rules["bidirectional"] = ["forward", "backward"]
    rules["case_distinction"] = ["sufficient", "necessary"]
    # If the rule is understood as forward inference then the invited inference is MP or DA.
    rules["forward"] = ("p", "not p")
    # If the rule is understood as backward inference then the invited inference is AC or MT.
    rules["backward"] = ("q", "not q")
    # If the antecedent is seen as sufficient then the invited inference is MP or AC.
    rules["sufficient"] = ("p", "q")
    # If the antecedent is seen as necessary then the invited inference is DA or MT.
    rules["necessary"] = ("not p", "not q")
    # If the available inference is also applied to the invisible side then the card that
    # leads to a contradiction with the visible side will be select.
    rules[("reversible", "p")] = ["p", "not q"]
    rules[("reversible", "not p")] = ["not p", "q"]
    rules[("reversible", "q")] = ["q", "not p"]
    rules[("reversible", "not q")] = ["not q", "p"]

    if "reversible" in params:
        reversible = True
    elif "irreversible" in params:
        reversible = False

    for assumption in params:
        if assumption in rules:
            result.append(rules[assumption])

    for res in result:
        # If conditional(or biconditional) is seen as one of the given ways then delete
        # the element of conditional(or biconditional).
        if type(res) is tuple:
            for r in res:
                # Whether the element is one of the assumptions or not
                if r in params:
                    result.remove(res)

    for index, res in enumerate(result):
        # The available inference is applied to both conditional.
        if type(res) is list:
            for r in res:
                selection.append(list(set(rules[r]).intersection(set(result[index - 1])))[0])

    # The rule is seen as conditional rather then biconditional.
    if len(selection) == 0:
        selection.append(list(set(result[0]).intersection(set(result[1])))[0])

    if reversible:
        for select in selection:
            key = ("reversible", select)
            turn += rules[key]
    else:
        for select in selection:
            turn.append(select)

    return turn
