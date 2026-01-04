# House Price Prediction using Linear Regression
# Offline dataset using make_regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic housing data (offline)
X, y = make_regression(
    n_samples=500,
    n_features=5,
    noise=15,
    random_state=42
)

# Convert to DataFrame
columns = ['AvgRooms', 'CrimeRate', 'Distance', 'Area', 'Population']
X = pd.DataFrame(X, columns=columns)
y = pd.Series(y, name="HousePrice")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("House Price Prediction (Linear Regression)")
plt.show()
