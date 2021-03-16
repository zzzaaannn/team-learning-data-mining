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
- MSE均方误差：$\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.$
- MAE平均绝对误差:$\text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|$
- $R^2$决定系数：$R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$
- 解释方差得分:$explained\_{}variance(y, \hat{y}) = 1 - \frac{Var\{ y - \hat{y}\}}{Var\{y\}}$
