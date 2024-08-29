import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load and prepare data
df = pd.read_csv("transformed_data.csv", sep='\t')

# Features and targets
X = df[['Latitude', 'Longitude']]
y = df[['time_sin', 'time_cos']]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the neural network
mlp = MLPRegressor(hidden_layer_sizes=(1000,), max_iter=300)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
print(f'Mean Squared Error MLP: {mse_mlp}')

# Save the trained model
joblib.dump(mlp, 'mlp_model.pkl')

# Convert predictions from sin and cos to time
def convert_sin_cos_to_time(sin_val, cos_val):
    angle = np.arctan2(sin_val, cos_val)
    angle = angle % (2 * np.pi)
    total_minutes = (angle / (2 * np.pi)) * (24 * 60)
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    return hours, minutes

    #*******************************************Must normalize lat long before predicting***************************************************



#
# # Create a MultiOutputRegressor with RandomForestRegressor as the base estimator
# rfr = RandomForestRegressor(n_estimators=120, max_depth=2)  # You can tune parameters like n_estimators and max_depth
# multi_target_rfr = MultiOutputRegressor(rfr)
# multi_target_rfr.fit(X_train, y_train)
# y_pred_rfr = multi_target_rfr.predict(X_test)
# mse_rfr = mean_squared_error(y_test, y_pred_rfr)
# print(f'Mean Squared Error RFR: {mse_rfr}')
#
#
# # Create and fit the model
# svr = SVR(kernel='linear', C=0.4, epsilon=0.8)  # You can specify hyperparameters if needed
# multi_target_svr = MultiOutputRegressor(svr)
# multi_target_svr.fit(X_train, y_train)
# y_pred_svr = multi_target_svr.predict(X_test)
# mse_svr = mean_squared_error(y_test, y_pred_svr)
# print(f'Mean Squared Error SVR: {mse_svr}')
#
#
#
# gbr = GradientBoostingRegressor(n_estimators=70, learning_rate=0.05, max_depth=2)  # You can tune parameters like n_estimators, learning_rate, and max_depth
# multi_target_gbr = MultiOutputRegressor(gbr)
# multi_target_gbr.fit(X_train, y_train)
# y_pred_gbr = multi_target_gbr.predict(X_test)
# mse_gbr = mean_squared_error(y_test, y_pred_gbr)
# print(f'Mean Squared Error GBR: {mse_gbr}')
