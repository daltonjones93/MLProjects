from data_analyzer import DataAnalzerTesla
from forecasting_library import ForecastingModel, ForecastingCompare
from models import nnModel
from forecast_visualizer import RegressionEvaluationPlotter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import torch
from torch import nn
import pandas as pd


#Clean and import data
ml_analyzer = DataAnalzerTesla("data.csv")
X,y = ml_analyzer.get_data()
X_predict = X[-24:,:]
times_predict = ml_analyzer.data['interval_start_time'].values[-24:]
X = X[:-24,:]
y = y[:-24]

X_train,X_test,y_train,y_test = train_test_split(X,y)

#Create Models
nn_model = nnModel(input_size=X.shape[1],activation = nn.LeakyReLU(),n_hidden_units=1100)



neural_network = ForecastingModel(nnModel(input_size=X.shape[1],activation = nn.LeakyReLU(),n_hidden_units=1100), name = "Neural Network")
rf_model = ForecastingModel(RandomForestRegressor(), name="Random Forest")

#Train Model
rf_model.train(X_train,y_train)
rf_model.evaluate_model(X_test,y_test)

#Evaluate Random Forest
y_pred = rf_model.predict(X_test)
rf_plotter = RegressionEvaluationPlotter(y_test,y_pred)


neural_network.train(X_train,y_train,X_test = X_test,y_test = y_test,
                     epochs = 5000,lr=.001,batch_size = 64)

#Evaluate Trained Neural Network'
neural_network.evaluate_model(X_test,y_test)
y_pred = neural_network.predict(X_test)
nn_plotter = RegressionEvaluationPlotter(y_test,y_pred)





# Compare Models Using Cross Validation
###This is too time consuming given the parameters of the assignment but would be valuable
###in practice, especially with access to a gpu.

# task = ForecastingCompare(models=[rf_model,neural_network])
# evaluations = task.compare_models_crossvalidation(X.values, y)
# for model_name, metrics in evaluations.items():
#     avg_metric = np.mean(metrics)
#     print(f"{model_name}: Average MSE - {avg_metric}")

#make predictions with highest performing model
y_predict = neural_network.predict(X_predict)

df = pd.DataFrame({'interval_start_time': pd.to_datetime(times_predict, utc=True, infer_datetime_format=True), 
                   'CAISO_system_load': y_predict})

# df.to_csv('TeslaPredictions.csv',index = False)

#output predictions as csv


