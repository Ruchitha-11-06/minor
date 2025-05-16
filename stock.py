import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load data
data = pd.read_csv('stock price.csv')

# Step 2: Preprocess and feature engineering
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

# Example: Use previous day's Close price as feature to predict today's Close
data['Prev_Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

X = data[['Prev_Close']]
y = data['Close']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Prediction and evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Plot actual vs predicted prices
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
