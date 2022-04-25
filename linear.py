from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

db_connection_str = 'mysql+pymysql://root: @localhost:3306/education'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM education.primary', con=db_connection)
df.to_dict('list')

df = df.dropna()
print(df)


x = df['Female'].to_numpy().reshape((-1, 1))
y = df['Male'].to_numpy()

# Create Model
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
rSquared = model.score(x, y)
print('\nLinear Regression: Female vs Male Result')
print('=======================================')
print('R Squared / Coefficient of Determination: ', rSquared)
print('Intercept: ', model.intercept_)
print('Slope: ', model.coef_)

# Predictions
yPrediction = model.predict(x)  # make predictions

# Display Scatterplot
plt.scatter(x, y)
plt.plot(x, yPrediction, color='red')
plt.title('Female vs Male')
plt.xlabel('Female')
plt.ylabel('Male')
plt.show()
