import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class RegressionEvaluationPlotter:
    def __init__(self, y_true, y_pred):
        """
        Initialize the RegressionEvaluationPlotter with true and predicted values.

        Parameters:
        - y_true (array-like): True values.
        - y_pred (array-like): Predicted values.
        """
        self.y_true = y_true
        self.y_pred = y_pred

        self.scatter_plot()
        self.residual_plot()
        self.distribution_plot()
    def scatter_plot(self):
        """
        Create a scatter plot of true vs predicted values.
        """
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.y_true, y=self.y_pred)
        plt.title('True vs Predicted Values')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.show()

    def residual_plot(self):
        """
        Create a residual plot.
        """
        residuals = self.y_true - self.y_pred
        plt.figure(figsize=(8, 6))
        sns.residplot(x=self.y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.show()

    def distribution_plot(self):
        """
        Create a distribution plot of residuals.
        """
        residuals = self.y_true - self.y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True)
        plt.title('Distribution of Residuals')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.show()

    def plot_actual_vs_predicted(self, num_points=100):
        """
        Plot actual vs predicted values for a sample of data points.

        Parameters:
        - num_points (int): Number of points to plot.
        """
        sample_indices = np.random.choice(len(self.y_true), num_points, replace=False)
        actual_values = self.y_true[sample_indices]
        predicted_values = self.y_pred[sample_indices]

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=actual_values, y=predicted_values)
        plt.title('Actual vs Predicted Values (Sample)')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()
