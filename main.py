from sklearn.linear_model import LinearRegression
import numpy as np

hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
scores = np.array([20, 30, 40, 50, 60, 70, 80, 90])

model = LinearRegression()
model.fit(hours, scores)

study_hours = float(input("Enter study hours: "))

prediction = model.predict([[study_hours]])

print("Predicted Score:", round(prediction[0], 2))