# Install necessary libraries if not already installed
# pip install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

class MLDataAnalyzer:
    def __init__(self,file_path,drop_col = [], target_col='target'):
        self.target_col = target_col
        self.data = None
        self.features = None
        self.target = None
        self.load_data(file_path,drop_col)

    def load_data(self, file_path,drop_col = []):
        # Load data from a CSV file
        self.data = pd.read_csv(file_path)
        self.target = self.data[self.target_col].values.reshape(-1,1)
        nonfeatures = drop_col + [self.target_col]
        self.features = self.data.drop(columns=nonfeatures)
        # self.data = self.data.drop(drop_col,axis = 1)

    def fill_missing_values(self, strategy='mean'):
        # Fill in missing values in the features using the specified strategy
        imputer = SimpleImputer(strategy=strategy)
        self.features = pd.DataFrame(imputer.fit_transform(self.features), columns=self.features.columns)
        self.target = imputer.fit_transform(self.target)

    def analyze_feature_importance(self, n_features=10, random_state=42):
        # Analyze feature importance using a Random Forest classifier
        rf_regressor = RandomForestRegressor(random_state=random_state)
        rf_regressor.fit(self.features, self.target.flatten())

        feature_importances = pd.Series(rf_regressor.feature_importances_, index=self.features.columns)
        top_features = feature_importances.nlargest(n_features)

        # Plot feature importance
        self.plot_feature_importance(top_features)

    def plot_seasonal_decomposition(self):
        self.data['interval_start_time'] = pd.to_datetime(self.data['interval_start_time'], utc=True, infer_datetime_format=True)
        res = sm.tsa.seasonal_decompose(pd.Series(self.target.flatten(), index = self.data['interval_start_time']), model='additive')
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 12))
        res.observed.plot(ax=ax1, title='Observed')
        res.trend.plot(ax=ax2, title='Trend')
        res.resid.plot(ax=ax3, title='Residual')
        res.seasonal.plot(ax=ax4, title='Seasonal')
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, feature_importance_series):
        # Plot feature importance
        feature_importance_series.plot(kind='barh', figsize=(10, 6))
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Top Important Features')
        plt.show()

    def plot_box_plot(self,feature_name):
        plt.figure()
        sns.boxplot(x=self.features[feature_name])
        plt.show()

    def trim_outliers(self,feature_name, max_val,min_val):
        #note, be sure to do this before filling in missing values
        self.features.loc[self.features[feature_name] > max_val,feature_name] = np.nan
        self.features.loc[self.features[feature_name] < min_val,feature_name] = np.nan

    def stationarity_test(self):
        # Visualize the time series
        time_series = self.target.flatten()
        plt.plot(time_series)
        plt.title("Time Series Data")
        plt.show()

        # Augmented Dickey-Fuller test
        result = adfuller(time_series)
        print("Augmented Dickey-Fuller Test:")
        print(f'Test Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print(f'Critical Values: {result[4]}')

        # Interpret the results
        if result[1] <= 0.05:
            print("The time series is stationary.")
        else:
            print("The time series is not stationary.")

    def plot_covariance_matrix(self):
        # Plot covariance matrix of the time series data
        cov_estimator = EmpiricalCovariance()
        cov_estimator.fit(self.features)

        cov_matrix = pd.DataFrame(cov_estimator.covariance_, columns=self.features.columns, index=self.features.columns)

        plt.figure(figsize=(10, 8))
        plt.imshow(cov_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Covariance Matrix')
        plt.colorbar()
        plt.xticks(range(len(cov_matrix)), cov_matrix.columns, rotation='vertical')
        plt.yticks(range(len(cov_matrix)), cov_matrix.columns)
        plt.show()

    def autocovariance_plot(self, max_lag=24*8):
        time_series = self.target.flatten()
        _, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6))
        plot_acf(time_series, lags=max_lag, ax=ax1)
        plot_pacf(time_series, lags=max_lag, ax=ax2)
        plt.tight_layout()
        plt.show()
    
    def scale_features(self):
        scaler = StandardScaler()
        self.features.iloc[:,:] = scaler.fit_transform(self.features)
        
    def get_data(self):
        return self.features.values, self.target.flatten()
    

class DataAnalzerTesla(MLDataAnalyzer):
    def __init__(self, file_name,drop_col = ["interval_start_time"], target_col='CAISO_system_load'):
        df = pd.read_csv(file_name)
        #add features
        self.add_features_tesla(df)
        super().__init__("dataTesla.csv",drop_col,target_col)
        self.scale_features()
        self.fill_missing_values()


    
    def add_features_tesla(self,df):
        df["interval_start_time"] = pd.to_datetime(df["interval_start_time"], utc=True, infer_datetime_format=True)

        system_load = df['CAISO_system_load'].values

        # df["prev_day_system_load"] = np.zeros(df.shape[0])
        # for j in range(24,df.shape[0]):
        #     df["prev_day_system_load"].iloc[j] = system_load[j-24] 
        for i in range(24):
            df["prev_day_system_load_hour_"+str(i)] = np.zeros(df.shape[0])
            for j in range(24,df.shape[0]):
                idx = 24 * (j//24 - 1) + i
                df["prev_day_system_load_hour_"+str(i)].iloc[j] = system_load[idx] 

        timex = []
        timey = []
        weekend = []
        for i in range(df.shape[0]):
            day = df['interval_start_time'][i].weekday()
            day = 1 if day in [5,6] else 0
            weekend.append(day)
            
            timex.append(np.cos((df['interval_start_time'][i].hour / 24.0)*2 * np.pi))
            timey.append(np.sin((df['interval_start_time'][i].hour / 24.0)*2 * np.pi))

        df['timex'] = timex
        df['timey'] = timey
        df['weekend'] = weekend
        df = df.iloc[24:,:]
        df.to_csv('dataTesla.csv', index = False)

        


# Example usage:
if __name__ == "__main__":
    
    #do some feature engineering within subclass
    ml_analyzer = DataAnalzerTesla("data.csv")



    ml_analyzer.stationarity_test()

    ml_analyzer.autocovariance_plot(max_lag = 24*8)

    # Analyze feature importance using a Random Forest regressor
    ml_analyzer.analyze_feature_importance(n_features=10)


    ml_analyzer.plot_seasonal_decomposition()

    # for name in ml_analyzer.features:
    #     ml_analyzer.plot_box_plot(name)

    # Plot the covariance matrix
    ml_analyzer.plot_covariance_matrix()
