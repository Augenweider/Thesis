def insight(conditional, assumption, cards):
    """
        Return the predicting result of an individual's selection on Wason selection
        task using the model theory.

        Parameters:
            conditional(list): the list of conditional
            assumption(list): the level of the individual's insight
            cards(list): the list of available cards

        Return:
            selection(list): the individual's card-selection.
        """
    selection = []
    turn = []
    proposition = []
    # Processing the given list of conditional
    for c in conditional:
        if c != 'If' and c != 'then':
            proposition.append(c)
            proposition.append("not " + c)

    # Generate a mental model of the given hypothesis
    model = [[proposition[0], proposition[2]], ['...']]

    # Select different cards according to the scanning direction
    if "Converse" in assumption:
        selection = model[0]
    elif "No converse" in assumption:
        selection.append(model[0][0])

    if "Partial insight" in assumption:
        # The mental model fleshes out to the fully explicit model.
        model.remove(['...'])
        model.append([proposition[1], proposition[2]])
        model.append([proposition[1], proposition[3]])
        if proposition[2] in selection:
            selection.append(proposition[3])
        else:
            selection.append(proposition[2])

    # Generate the fully explicit model when the individual have complete insight on the counterexample from the outset.
    if "Fully insight" in assumption:
        # All conjunctions with possibility or impossibility
        all_conjunction = [[proposition[0], proposition[2]],
                           [proposition[1], proposition[2]],
                           [proposition[1], proposition[3]],
                           [proposition[0], proposition[3]]]

        for conjunction in all_conjunction:
            if conjunction not in model:
                selection = conjunction

    # If the selection is available on the given cards then turn it.
    for s in selection:
        if s in cards:
            turn.append(s)
    # Return the turned cards
    return turn
