# ✅ Full Logistic Regression Example with Step-by-Step Explanation

import numpy as np
import math
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 🎯 Real-World Example: Predict if a student will pass based on study hours
# -----------------------------------------------------------
# Input: Hours Studied (x)
# Label: Pass (1) / Fail (0)
# -----------------------------------------------------------

# 🔍 Background on How Logistic Regression Works:
# -----------------------------------------------------------
# 1. It uses a linear equation: z = w * x + b
#    where w = weight, x = input, b = bias
# 2. The output z is passed through a sigmoid function to convert it to a probability:
#    sigmoid(z) = 1 / (1 + e^-z)
# 3. The output of sigmoid is a value between 0 and 1 (e.g., 0.88)
# 4. We use log loss (cost function) to measure how good the prediction is:
#    Cost = -[y * log(ŷ) + (1 - y) * log(1 - ŷ)]
#    - If prediction is close to actual, cost is low.
#    - If prediction is wrong and confident, cost is high.
# 5. During training, the model adjusts w and b using gradient descent to minimize the cost.
# -----------------------------------------------------------

# Step 1: Sample Data (Very small example for simplicity)
X = np.array([[2], [3], [4], [5], [6]])  # Hours studied
y = np.array([0, 0, 0, 1, 1])           # 0 = Fail, 1 = Pass

# Step 2: Create and Train Logistic Regression Model
model = LogisticRegression()
model.fit(X, y)

# Step 3: Predict for a new student who studied 5 hours
x_input = 5
z = model.coef_[0][0] * x_input + model.intercept_[0]
y_hat = 1 / (1 + math.exp(-z))  # Sigmoid manually

print("\n--- Manual Calculation ---")
print(f"Input x = {x_input} hours")
print(f"z = {model.coef_[0][0]:.4f} * {x_input} + {model.intercept_[0]:.4f} = {z:.4f}")
print(f"e^-z = math.exp(-z) = {math.exp(-z):.4f}")
print(f"Predicted probability (sigmoid) = {y_hat:.4f}")

# Step 4: Compute Cost (Log Loss)
y_actual = 1  # Assume the student actually passed
cost = -(y_actual * math.log(y_hat) + (1 - y_actual) * math.log(1 - y_hat))
print(f"Log Loss (Cost) = {cost:.4f}")

# Step 5: Final Prediction
prediction = 1 if y_hat >= 0.5 else 0
print(f"Predicted class = {prediction} => {'Pass' if prediction == 1 else 'Fail'}")

# Step 6: Optional: Visualize the Sigmoid Curve
z_vals = np.linspace(-6, 6, 100)
sigmoid_vals = 1 / (1 + np.exp(-z_vals))

plt.plot(z_vals, sigmoid_vals)
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('Sigmoid(z)')
plt.grid(True)
plt.axhline(0.5, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.show()


#================================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Simulated real-world dataset
hours_studied = np.array([1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
pass_fail = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(hours_studied, pass_fail, test_size=0.2, random_state=0)

# Step 3: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Predict and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 5: Predict for a new student who studied 4.2 and 6.5 hours
new_data = np.array([[4.2], [6.5]])
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)

for i, hours in enumerate(new_data):
    print(f"\nIf studied {hours[0]} hours:")
    print(f"Prediction: {'Pass' if predictions[i] == 1 else 'Fail'}")
    print(f"Probability of Pass: {probabilities[i][1]:.2f}")

# Step 6: Visualize the logistic curve
X_range = np.linspace(0, 11, 300).reshape(-1, 1)
probs = model.predict_proba(X_range)[:, 1]

plt.scatter(hours_studied, pass_fail, color='blue', label='Actual Data')
plt.plot(X_range, probs, color='red', label='Logistic Curve')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression: Student Pass Prediction')
plt.legend()
plt.grid(True)
plt.show()
