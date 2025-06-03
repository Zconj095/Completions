import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulated historical data: days and number of game events (e.g., player logins)
days = np.array([i for i in range(1, 31)])  # Day 1 to 30
events = np.array([2*i + np.random.randint(-5, 5) for i in days])  # Linear trend with noise

# Reshape data for modeling
X = days.reshape(-1, 1)
y = events.reshape(-1, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Plot the results
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, predictions, color='blue', linewidth=3, label='Linear regression')
plt.xlabel('Day')
plt.ylabel('Number of Events')
plt.title('Game Event Forecasting')
plt.legend()
plt.show()
