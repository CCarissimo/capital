def production(multiplier, elasticity, capital, labour):
    return multiplier*capital**(elasticity)*labour**(1-elasticity)


def redistribution(Y, beta, capital_allocation, labour_allocation):
    to_capitalists = Y * beta
    to_labourers = Y * (1 - beta)

    # shortcut
    if capital_allocation == 0 or labour_allocation == 0:
        return [.0, .0]
    else:
        capital_return = to_capitalists / capital_allocation
        labour_return = to_labourers / labour_allocation
        return [labour_return, capital_return]

