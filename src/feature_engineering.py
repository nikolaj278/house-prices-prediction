def add_new_feat(X):
    """
    Receives a DataFrame and returns a copy 
    with additional engineered features.
    
    """

    X['Size_n_Quality'] = X['GrLivArea'] * X['OverallQual']
    X['BsmtBathByNeigh'] = X['BsmtFullBath'] * X['Neighborhood']
    X['TotalArea'] =  X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
    X['TotalBathrooms'] = (X['FullBath'] + 0.5 * X['HalfBath'] +
                        X['BsmtFullBath'] + 0.5 * X['BsmtHalfBath'])
    return X