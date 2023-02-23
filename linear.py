# So here we do import necessary modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
# Reading data from the course
df = pd.read_csv('Ecommerce Customers')
df.head()
df.info()

# Importing seaborn and setting style
import seaborn as sns
sns.set_style('whitegrid')
sns.jointplot(data = df,x = 'Time on Website',y='Yearly Amount Spent')
sns.jointplot(data = df,x = 'Time on App',y='Yearly Amount Spent')
sns.pairplot(df)

# splitting data intp train and test data
from sklearn.model_selection import train_test_split
df.columns
X = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state=101)

# Using preapred model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.coef_)

results = lm.predict(X_test)
plt.scatter(y_test,results)

#Checking how our models performs
from sklearn import metrics
MAE = metrics.mean_absolute_error(y_test,results)
MSE = metrics.mean_squared_error(y_test,results)
RMSE = np.sqrt(metrics.mean_squared_error(y_test,results))
plt.hist(y_test - results)
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# Calculating MAE by hand
hook = 0
suma_cala = 0
correct = list(y_test)
for result in results:
    suma = (result -correct[hook] )
    suma = abs(suma)
    suma_cala += suma
    hook += 1