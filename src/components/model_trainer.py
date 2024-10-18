import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


df = pd.read_csv("src/artifacts/data.csv")


# Select X , y
X = df.drop(columns=["math_score"],axis=1)
y = df[["math_score"]]


# Compose the Preprocess
stand_object = StandardScaler()
onehot_object = OneHotEncoder()

o_features = X.select_dtypes(include="object").columns
num_features = X.select_dtypes(exclude="object").columns

preprocess = ColumnTransformer(
    [
        ("OneHotEncoder", onehot_object, o_features),
        ("StandardScaler", stand_object, num_features),
    ]
)

X = preprocess.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(true, predicted):

    mae = mean_squared_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)

    return mae, rmse, r2_square

models = {"Linear Regression": LinearRegression(),
          "Lasso": Lasso(),
          "Ridge": Ridge(),
          "K-Neighbors Regressor": KNeighborsRegressor(),
          "Decision Tree": DecisionTreeRegressor(),
          "Random Forest Regressor": RandomForestRegressor(),
          "XGBRegressor": XGBRegressor(),
          "CatBoostin Regressor": CatBoostRegressor(verbose=False),
          "AdaBoost Regressor": AdaBoostRegressor()}

model_list = []
r2_list = []

for i in range(len(list(models))):
    # Create model
    model = list(models.values())[i]
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics to check
    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)


    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])

    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')

    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    r2_list.append(model_test_r2)

    print('=' * 35)
    print('\n')



df_results = pd.DataFrame(list(zip(model_list, r2_list)), columns=["Model_Name", "R2_Score"])


#   Analyse Linear Model
lin_model = LinearRegression(fit_intercept=True)
lin_model = lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
score = r2_score(y_test, y_pred)*100
print(" Accuracy of the model is %.2f" %score)


matplotlib.use("TkAgg")
plt.ion()
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

sns.regplot(x=y_test, y=y_pred,ci=None,color="red")


y_test_1d = y_test.ravel() if hasattr(y_test, 'ravel') else y_test
y_pred_1d = y_pred.ravel() if hasattr(y_pred, 'ravel') else y_pred

# Criar o DataFrame
pred_df = pd.DataFrame({
    'Actual Value': y_test_1d,
    'Predicted Value': y_pred_1d,
    'Difference': y_test_1d - y_pred_1d
})
