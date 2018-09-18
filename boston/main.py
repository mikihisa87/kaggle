import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from scipy.stats import zscore
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
%matplotlib inline

boston=load_boston()
boston.feature_names
X=pd.DataFrame(boston.data,columns=boston.feature_names)
x=X['LSTAT']
y=boston.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

lin_1d = LinearRegression()
lin_1d.fit(x[:,None],y)

score_1d = lin_1d.score(x[:,None],y)
print('一次式における"LSTAT"の住宅価格への決定係数は%.3f'%(score_1d))

y_plot=[]
dim=[2,3,4]

for i in dim:
    degree = PolynomialFeatures(degree=i)
    x_P=degree.fit_transform(x[:,None])
    line=LinearRegression()
    line.fit(x_P,y)
    L = line.score(x_P,y)
    print('{}次関数における"LSTAT"の住宅価格への決定関数は%.5f'.format(i)%(L))
    print('2乗和誤差は%.5f'%mean_squared_error(y,line.predict(x_P)))
    
    n=np.linspace(np.min(x),np.max(x),506)
    y_predict = line.predict(degree.fit_transform(n[:,np.newaxis]))
    y_plot.append(y_predict)
    print(x.ndim)
    
mean_squared_error(y,y_predict)
