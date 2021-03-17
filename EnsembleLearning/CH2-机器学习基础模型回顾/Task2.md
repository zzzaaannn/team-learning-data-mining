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

    ![equation](https://latex.codecogs.com/gif.latex?\text{MSE}(y,&space;\hat{y})&space;=&space;\frac{1}{n_\text{samples}}&space;\sum_{i=0}^{n_\text{samples}&space;-&space;1}&space;(y_i&space;-&space;\hat{y}_i)^2.)

    assign more weight to the bigger errors

- RMSE均方根误差: measure the square root of the squares of the errors

    ![equation](https://latex.codecogs.com/gif.latex?\text{RMSE}(y,&space;\hat{y})&space;=&space;\sqrt{\frac{1}{n_\text{samples}}&space;\sum_{i=0}^{n_\text{samples}&space;-&space;1}&space;(y_i&space;-&space;\hat{y}_i)^2.})

- MAE平均绝对误差: measure the average of the absolute errors

    ![equation](https://latex.codecogs.com/gif.latex?\text{MAE}(y,&space;\hat{y})&space;=&space;\frac{1}{n_{\text{samples}}}&space;\sum_{i=0}^{n_{\text{samples}}-1}&space;\left|&space;y_i&space;-&space;\hat{y}_i&space;\right|.)

    a linear score which means that all the individual differences are weighted equally in the average

- R^2决定系数：

    ![equation](https://latex.codecogs.com/gif.latex?R^2(y,&space;\hat{y})&space;=&space;1&space;-&space;\frac{\sum_{i=1}^{n}&space;(y_i&space;-&space;\hat{y}_i)^2}{\sum_{i=1}^{n}&space;(y_i&space;-&space;\bar{y})^2}.)

    range between [0,1]

- 解释方差得分: measure the proportion to which a model accounts for the variation of a given data set
    
    ![equation](https://latex.codecogs.com/gif.latex?explained\:variance(y,&space;\hat{y})&space;=&space;1&space;-&space;\frac{Var(&space;y&space;-&space;\hat{y})}{Var(y)})


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

## Step 3: select the model

### Linear Regression

    - 回归最初的概念：趋于“平均” 

    - 一种预测性的建模技术，研究的是因变量（目标）和自变量（特征）之间的关系

    - 通常用于预测分析，时间序列模型以及发现变量之间的因果关系

    - 使用曲线/线来拟合数据点，目标是使曲线到数据点的距离差异最小

#### 推算方法

1. Least Squares

![equation](https://latex.codecogs.com/gif.latex?L(w)&space;=&space;\sum\limits_{i=1}^{N}||w^Tx_i-y_i||_2^2=\sum\limits_{i=1}^{N}(w^Tx_i-y_i)^2&space;=&space;(w^TX^T-Y^T)(w^TX^T-Y^T)^T&space;=&space;w^TX^TXw&space;-&space;2w^TX^TY&plus;YY^T)  

solve it by taking the minimum
![equation](https://latex.codecogs.com/gif.latex?\hat{w}&space;=&space;argmin\;L(w))  


take the derivative
![equation](https://latex.codecogs.com/gif.latex?\frac{\partial&space;L(w)}{\partial&space;w}&space;=&space;2X^TXw-2X^TY&space;=&space;0)  


![equation](https://latex.codecogs.com/gif.latex?\hat{w}&space;=&space;(X^TX)^{-1}X^TY)  


