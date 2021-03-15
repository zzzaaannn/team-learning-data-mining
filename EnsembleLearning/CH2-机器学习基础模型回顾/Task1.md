# Machine Learning

## Supervised Learning(有监督学习)
given y and x
e.g. 我们使用**房间面积**，**房屋所在地区**，**环境等级**等特征去预测某个地区的**房价**

## Unsupervised Learning(无监督学习)
only given x 
e.g. 我们给定某电商用户的基本信息和消费记录，通过观察数据中的哪些类型的用户彼此间的行为和属性类似，形成一个客群。注意，我们本身并不知道哪个用户属于哪个客群

# Regression(回归)
y is continuous

    from sklearn import datasets    # sklearn自带数据集
    boston = datasets.load_boston()     # 返回一个类似于字典的类
    dir(boston) #返回此数据集中所带的对象
    X = boston.data
    y = boston.target
    features = boston.feature_names
    boston_data = pd.DataFrame(X,columns=features)
    boston_data["Price"] = y
    
    sns.scatterplot(boston_data['NOX'],
                    boston_data['Price'],
                    color="r",alpha=0.6)  #x和y变量的关系图
    plt.title("Price~NOX")
    plt.show()



# Classification(分类)
y is discrete