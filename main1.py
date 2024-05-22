import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, LinearRegression
import lightgbm as lgbm
import xgboost as xg
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

data = pd.read_csv("Train (1).csv")
test_data = pd.read_csv("Test (1).csv")


if 'Measures' in data.columns:
    for i in range(data.shape[0]):
        tuple_string = data.loc[i, 'Measures']
        values_str = tuple_string.strip("()").split(", ")
        float1, float2, float3 = map(float, values_str)

        data.at[i, 'Measures'] = float1
        data.at[i, 'Measures1'] = float1
        data.at[i, 'Measures2'] = float2
        data.at[i, 'Measures3'] = float3


if 'Weights' in data.columns:
    for i in range(data.shape[0]):
        tuple_string = data.loc[i, 'Weights']
        values_str = tuple_string.strip("()").split(", ")
        float1, float2, float3 = map(float, values_str)

        data.at[i, 'Weights'] = float1
        data.at[i, 'Weights1'] = float1
        data.at[i, 'Weights2'] = float2
        data.at[i, 'Weights3'] = float3


if 'Measures' in test_data.columns:
    for i in range(test_data.shape[0]):
        tuple_string = test_data.loc[i, 'Measures']
        values_str = tuple_string.strip("()").split(", ")
        float1, float2, float3 = map(float, values_str)


        test_data.at[i, 'Measures'] = float1
        test_data.at[i, 'Measures1'] = float1
        test_data.at[i, 'Measures2'] = float2
        test_data.at[i, 'Measures3'] = float3

if 'Weights' in test_data.columns:
    for i in range(test_data.shape[0]):
        tuple_string = test_data.loc[i, 'Weights']
        values_str = tuple_string.strip("()").split(", ")
        float1, float2, float3 = map(float, values_str)


        test_data.at[i, 'Weights'] = float1
        test_data.at[i, 'Weights1'] = float1
        test_data.at[i, 'Weights2'] = float2
        test_data.at[i, 'Weights3'] = float3


column_to_drop = 'Measures'
data = data.drop(column_to_drop, axis=1)

column_to_drop = 'Weights'
data = data.drop(column_to_drop, axis=1)

data = pd.get_dummies(data, columns=['Gender'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Gender'], drop_first=True)

for col in data.columns:
    fig, ax = plt.subplots(figsize=(len(data.columns), 1))
    sns.boxplot(x=data[col], ax=ax)
    ax.set(ylabel=col)  # Set y-label for the subplot

data.drop(data[data["Measures3"] >= 0.6].index, inplace=True)

ss = StandardScaler()

rs = np.random.RandomState(0)
corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')
print(corr)

x = data.drop(columns='Age', axis=1)
y = data["Age"]
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=30)
x_train = ss.fit_transform(x_train.values)
x_test = ss.fit_transform(x_test.values)

def compare_models(x):
    x.fit(x_train, y_train)
    yhat = x.predict(x_test)
    y_known = x.predict(x_train)
    algoname = x.__class__.__name__
    return algoname, round(np.sqrt(mean_squared_error(y_test, yhat)),2)

algo=[GradientBoostingRegressor(), lgbm.LGBMRegressor(), xg.XGBRFRegressor(), xg.XGBRegressor(), SGDRegressor(), LinearRegression(),RandomForestRegressor()]
score=[]

for a in algo:
    score.append(compare_models(a))

print(pd.DataFrame(score, columns=['Model', 'MSE']))

gb_regressor = GradientBoostingRegressor( n_estimators=154,learning_rate=0.1, max_depth=3,  min_samples_split=2,  min_samples_leaf=2,  subsample=1.0)
gb_regressor.fit(x_train, y_train)

y_pred_test = gb_regressor.predict(ss.transform(test_data.drop(columns=['Measures', 'Weights']).values))

print(type(y_pred_test))
test_data = test_data.drop(columns=["Measures","Weights"])
print(type(test_data))

test_data['ID'] = range(0, len(test_data))
results_test = pd.DataFrame({
    'ID': test_data['ID'],
    'y_pred_test': y_pred_test
})
results_test.to_csv('result_predictions.csv', index=False)
print("Result predictions saved to 'result_predictions.csv'")