# student_performance_predictor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset (you can expand this later)
data = pd.DataFrame({
    'study_hours': [1, 2, 3, 4, 5, 6],
    'attendance': [60, 70, 80, 85, 90, 95],
    'previous_score': [40, 50, 60, 65, 70, 80],
    'final_score': [45, 55, 65, 70, 75, 85]
})

X = data[['study_hours', 'attendance', 'previous_score']]
y = data['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))