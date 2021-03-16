# Regression

## Step 1: collect dataset and select X

    from sklearn import datasets
    boston = datasets.load_boston()     # 返回一个类似于字典的类
    X = boston.data
    y = boston.target
    features = boston.feature_names
    boston_data = pd.DataFrame(X,columns=features)
    boston_data["Price"] = y

## Step 2: select evaluation metrics
- MSE均方误差：measure the average of the squares of the errors

    $\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.$

    assign more weight to the bigger errors

- RMSE均方根误差: measure the square root of the squares of the errors

    $\text{RMSE}(y, \hat{y}) = \sqrt{\frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.}$

- MAE平均绝对误差: measure the average of the absolute errors

    $\text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|$

    a linear score which means that all the individual differences are weighted equally in the average

- $R^2$决定系数：

    $R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$

    range between [0,1]

- 解释方差得分: measure the proportion to which a model accounts for the variation of a given data set
    
    $explained\_{}variance(y, \hat{y}) = 1 - \frac{Var\{ y - \hat{y}\}}{Var\{y\}}$

### sklearn的调用
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import explained_variance_score

    mean_squared_error(y,y_predict)
    mean_squared_error(y,y_predict,squared = False)
    mean_absolute_error(y,y_predict)
    r2_score(y,y_predict)
    explained_variance_score(y, y_predict)

API-evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
