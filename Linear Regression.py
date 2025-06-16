# 🎯 Real-World Linear Regression Example: Predict the student's marks based on hours studied
# -----------------------------------------------------------
# Input: Hours Studied (x)
# Output: Marks (y)
# -----------------------------------------------------------

# 🔍 Linear Regression Background:
# -----------------------------------------------------------
# 1. Linear Regression models the relationship as: y = wx + b
# 2. It predicts a continuous number (not probability)
# 3. Cost Function = Mean Squared Error (MSE): (y - ŷ)^2
# 4. Model uses gradient descent to minimize total MSE
# -----------------------------------------------------------

# Step 1: Linear Regression Dataset
X_reg = np.array([[2], [3], [4], [5], [6]])      # Hours studied
y_reg = np.array([50, 60, 65, 70, 80])           # Marks received

# Step 2: Train Linear Regression Model
lin_model = LinearRegression()
lin_model.fit(X_reg, y_reg)

# Step 3: Predict using Linear Regression
x_input = 5
predicted_marks = lin_model.predict([[x_input]])[0]

# Step 4: Calculate MSE manually for x = 5
actual_marks = 70
mse = (actual_marks - predicted_marks) ** 2

print("\n--- Linear Regression Manual Calculation ---")
print(f"Input x = {x_input} hours")
print(f"Predicted marks = {predicted_marks:.2f}")
print(f"Actual marks = {actual_marks}")
print(f"Mean Squared Error (for one point) = {mse:.4f}")

# Step 5: Linear Regression Line Visualization
plt.scatter(X_reg, y_reg, color='blue', label='Actual Data')
plt.plot(X_reg, lin_model.predict(X_reg), color='red', label='Regression Line')
plt.scatter(x_input, predicted_marks, color='green', label='Predicted Point')
plt.title('Linear Regression - Hours Studied vs Marks')
plt.xlabel('Hours Studied')
plt.ylabel('Marks')
plt.legend()
plt.grid(True)
plt.show()


#===================================================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Sample real-world-style data (Sq.ft vs Price in $1000s)
square_feet = np.array([800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000]).reshape(-1, 1)
prices = np.array([150, 180, 200, 240, 280, 310, 330, 360, 400, 420])

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(square_feet, prices, test_size=0.2, random_state=1)

# Step 3: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict on Test Data
y_pred = model.predict(X_test)

# Step 5: Evaluate with MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Step 6: Print Model Parameters
print(f"Model Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Model Intercept: {model.intercept_:.2f}")

# Step 7: Predict for a new house (e.g., 2600 sq.ft)
new_sqft = np.array([[2600]])
predicted_price = model.predict(new_sqft)
print(f"Predicted price for 2600 sq.ft house: ${predicted_price[0]*1000:.2f}")

# Step 8: Plot the Results
plt.scatter(square_feet, prices, color='blue', label='Actual Prices')
plt.plot(square_feet, model.predict(square_feet), color='red', label='Regression Line')
plt.xlabel('Square Footage')
plt.ylabel('House Price ($1000s)')
plt.title('Linear Regression: House Price Prediction')
plt.legend()
plt.grid(True)
plt.show()