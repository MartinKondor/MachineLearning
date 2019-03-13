
def accuracy(y, y_hat):
    good = 0
    
    for i in range(y_hat.shape[0]):
        if y[i] == 1 and y_hat[i] > .5:
            good+=1
            
    return 100*good / y_hat.shape[0]


# Sum all the columns
def sumcols(df):
    """
    :param df: pandas.DataFrame
    """
    X = df.loc[:, df.columns[0]].values

    for i in range(1, len(df.columns)):
        X += df.loc[:, df.columns[i]].values

    return X
    