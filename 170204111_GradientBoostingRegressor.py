import pandas as pd
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('car-data.csv')

print(dataset.head())
X = dataset[['Present_Price','Kms_Driven','Seating_Capacity','Car_Engine_cc','Top_speed_kmh','Age']]
Y = dataset[['Selling_Price']]


x = X.values
y = Y.values


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.66, random_state=42)


from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=4)

gbr.fit(x_train, y_train)

predicted=gbr.predict(x_test)


print("Predict value " + str(predicted))
print("Real value " + str(y_test))
print('r2_score: ',skl.metrics.r2_score(y_test,predicted)*100)
print('explained_variance_score: ',skl.metrics.explained_variance_score(y_test,predicted)*100)
print('neg_mean_gamma_deviance: ',skl.metrics.mean_gamma_deviance(y_test,predicted)*100)
print('neg_mean_absolute_percentage_error',skl.metrics.mean_absolute_percentage_error(y_test,predicted)*100)

