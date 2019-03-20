from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae


def E(model, X_train, X_test, y_train, y_test):
    xtrainp = model.predict(X_train)
    xtestp = model.predict(X_test)
    
    print('RMSE on test:', mse(y_test, xtestp)**(1/2) )
    print('MSE on test:', mse(y_test, xtestp) )
    print('MAE on test:', mae(y_test, xtestp) )
    
    print('RMSE on train:', mse(y_train, xtrainp)**(1/2) )
    print('MSE on train:', mse(y_train, xtrainp) )
    print('MAE on train:', mae(y_train, xtrainp) )

