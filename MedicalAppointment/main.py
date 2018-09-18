from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df =pd.read_csv('KaggleV2-May-2016 5.csv')
a =df['No-show'].replace('0','No')
y =a.replace('1','Yes')
feature_dframe['No-show'] = y.apply(lambda g:1 if g =='Yes' else 0)
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
x =df['AppointmentDay'] 

n =feature_dframe['No-show'].replace('0','No')
yn =a.replace('1','Yes')
yn.value_counts()

yes =yn[yn.str.contains('Yes')]
no =yn[yn.str.contains('No')]

feature_dframe['number'] = y.apply(lambda g:1 if g =='Yes' else 0)
df2 =pd.concat([df['AppointmentDay'], yn,feature_dframe['number']], axis=1)
xxx =df2.groupby(['AppointmentDay', 'No-show']).count()

ap_no =pd.concat([x,no],axis = 1)
ap_no_gby = ap_no.groupby(['AppointmentDay']).count()
no_show_no =ap_no_gby.rename(columns ={'No-show': 'No-show-No'})
no_show_no

ap_yes =pd.concat([x,yes],axis =1)
ap_yes_gby = ap_yes.groupby(['AppointmentDay']).count()
no_show_yes =ap_yes_gby.rename(columns={'No-show':'No-show-Yes'})

plt.figure(figsize=(10,6))
plt.plot(no_show_no, label='No-show No')
plt.plot(no_show_yes, label='No-show Yes')

plt.title('Medical Appointment')
plt.xlabel('AppointDay')
plt.ylabel('Number')

plt.legend()
plt.show()

feature_dframe = pd.DataFrame()
feature_dframe['Age'] = df['Age']
feature_dframe['Patientld'] = df['PatientId']
feature_dframe['AppointmentID'] = df['AppointmentID']
feature_dframe['Gender'] = df['Gender'].apply(lambda g:1 if g == 'M' else 0)
feature_dframe['SMS_received'] = df['SMS_received']
feature_dframe['Handcap'] = df['Handcap']
feature_dframe['Alcoholism'] = df['Alcoholism']
feature_dframe['Diabetes'] = df['Diabetes']
feature_dframe['Hipertension'] = df['Hipertension']
feature_dframe['Scholarship'] = df['Scholarship']
feature_dframe['No-show'] = df['No-show'].apply(lambda g:1 if g == 'Yes' else 0)

plt.figure(figsize=(10,10))
sns.heatmap(feature_dframe.corr(),annot=True)

y =feature_dframe['No-show']
xx =feature_dframe['Scholarship'] .replace('VA;PALESTINA','0')
#xxx =xx.rename('Scholarship2')
feature_dframe['Scholarship'] = xx

#feature_dframe['Hipertension'] = df['Hipertension'].apply(xx)
X =np.array(feature_dframe.drop('No-show', axis=1))
#X =feature_dframe.drop('No-show', axis=1)

#サンプルを最初にシャッフルしてから、訓練、テストデータに分割
#random_stateを明示的にすることで、結果の再現性を制御する
ss =ShuffleSplit(n_splits=1,
                 train_size=0.7,
                test_size=0.3,
                random_state=0)

#next():呼出の度に、元の要素を順番通りに返す
#split:数字、アルファベット、記号などが入り混じった文字列を、ある規則に従って分割
train_index, test_index = next(ss.split(X,y))

X_train, X_test =X[train_index],X[test_index]
y_train,y_test =y[train_index],y[test_index]

clf = RandomForestClassifier(max_depth=15,criterion='entropy', random_state=0)
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

tree_clf =tree.DecisionTreeClassifier(max_depth=15,criterion='entropy',random_state=0)
tree_clf.fit(X_train,y_train)

print(tree_clf.score(X_train,y_train))
print(tree_clf.score(X2_test,y_test))
