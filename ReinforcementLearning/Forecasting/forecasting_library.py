
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from time import time
from models import nnModel

class ForecastingModel:
    def __init__(self, model, name=None):
        self.model = model
        self.name = name or str(model)

    def train(self, X_train, y_train,verbose = False,X_test = None, y_test= None,
               epochs = 10000, lr = .005, batch_size = 256):
        time0 = time()
        # Assume X_train, y_train are numpy arrays
        if isinstance(self.model, nn.Module):
            self._train_pytorch(X_train, y_train,X_test= X_test,y_test= y_test,epochs = epochs,
                                lr = lr,batch_size = batch_size)
        else:
            self._train_sklearn(X_train, y_train)
        time1 = time()
        if verbose:
            print("Training Model Took {} seconds".format(str(time1-time0)))

    def _train_sklearn(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        

    def _train_pytorch(self, X_train, y_train, epochs, lr, batch_size, X_test= None, y_test = None):
        if isinstance(X_test,np.ndarray):
            X_test,y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
            
            if isinstance(X_test,torch.Tensor):
                if epoch % 10 == 0:
            
                    y_pred_test = self.model(X_test)
                    
                    mse_test = mean_squared_error(y_pred_test.detach().numpy().flatten(),y_test.detach().numpy().flatten())
                    
                    print("Epoch: " +str(epoch)+", Loss: "+str(loss.item()) + ", Test Loss: "+str(mse_test))

    def predict(self, X_test):
        # Assume X_test is a numpy array
        if isinstance(self.model, nn.Module):
            return self._predict_pytorch(X_test)
        else:
            return self._predict_sklearn(X_test)

    def _predict_sklearn(self, X_test):
        return self.model.predict(X_test)

    def _predict_pytorch(self, X_test):
        self.model.eval()
        with torch.inference_mode():
            output = self.model(torch.from_numpy(X_test).float())
        return output.numpy().squeeze()

    def clone(self):
        return ForecastingModel(clone(self.model), name=self.name)
    
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the out-of-sample performance of a machine learning model.

        Parameters:
        - model: The trained machine learning model
        - X_test: Test features
        - y_test: True labels for the test set

        Returns:
        - Dictionary containing evaluation metrics
        """
        # Make predictions on the test set
        y_pred = self.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print the evaluation metrics
        print(f'Mean Squared Error (MSE): {mse:.4f}')
        print(f'Mean Absolute Error (MAE): {mae:.4f}')
        print(f'R-squared (R2): {r2:.4f}')

        # Return evaluation metrics as a dictionary
        return {'MSE': mse, 'MAE': mae, 'R2': r2}



class ForecastingCompare:
    def __init__(self, models: List[ForecastingModel], evaluation_metric=mean_squared_error):
        self.models = models
        self.evaluation_metric = evaluation_metric

    def train_models(self, X_train, y_train):
        for model in self.models:
            model.train(X_train, y_train)

    def evaluate_models(self, X_test, y_test):
        evaluations = {}
        for model in self.models:
            y_pred = model.predict(X_test)
            evaluation = self.evaluation_metric(y_test, y_pred)
            evaluations[model.name] = evaluation
        return evaluations

    def compare_models_crossvalidation(self, X, y, cv_splits=5):
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        evaluations = {}

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.train_models(X_train, y_train)
            evaluation_fold = self.evaluate_models(X_test, y_test)

            for model_name, evaluation in evaluation_fold.items():
                evaluations.setdefault(model_name, []).append(evaluation)

        return evaluations

def example_usage():
    # Example usage of the forecasting library
    # Generate synthetic data for demonstration
    np.random.seed(42)
    X = pd.date_range(start="2022-01-01", periods=100, freq="D")
    y = 2 * np.arange(100) + np.random.normal(0, 5, size=100)

    # Feature engineering (you might want to customize this based on your data)
    X = pd.DataFrame({"date": X})
    X["day_of_week"] = X["date"].dt.dayofweek
    X["day_of_year"] = X["date"].dt.dayofyear
    X = X.set_index("date")

    # Split data into train and test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create forecasting models
    neural_network = ForecastingModel(nnModel(input_size = 2), name = "Neural Network")
    linear_model = ForecastingModel(LinearRegression(), name="Linear Regression")
    rf_model = ForecastingModel(RandomForestRegressor(), name="Random Forest")
    neural_network.train(X_train.values,y_train,X_test = X_test.values,y_test = y_test)
    

    linear_model.train(X_train, y_train)
    rf_model.train(X_train, y_train)
    neural_network.evaluate_model(X_test.values,y_test)
    linear_model.evaluate_model(X_test,y_test)
    rf_model.evaluate_model(X_test,y_test)

    # Create a forecasting task
    task = ForecastingCompare(models=[linear_model, rf_model,neural_network])

    # Backtest models
    evaluations = task.compare_models_crossvalidation(X.values, y)

    # Print average evaluation metrics
    for model_name, metrics in evaluations.items():
        avg_metric = np.mean(metrics)
        print(f"{model_name}: Average MSE - {avg_metric}")

if __name__ == "__main__":
    example_usage()