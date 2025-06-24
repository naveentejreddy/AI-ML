from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# 1. Prepare the data
X = np.array([
    [1000, 2],
    [850, 2],
    [1200, 3],
    [950, 1],
    [1100, 2]
])  # Features: Size and Bedrooms

y = np.array([2000, 1850, 2500, 1700, 2200])  # Rent ($)

# 2. Define the model
model = KNeighborsRegressor(n_neighbors=3)

# 3. Train the model
model.fit(X, y)

# 4. Predict rent for a new house
new_house = np.array([[1050, 2]])
predicted_rent = model.predict(new_house)

print(f"🏡 Predicted Rent: ${predicted_rent[0]:.2f}")