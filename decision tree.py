import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
#from sklearn.ensemble import RandomForestClassifier


df=pandas.read_csv("wine_quality_skylab.csv")
df['quality (target)']=df['quality (target)'].astype(float) #işlem yapılabilmesi için veri tiplerinin aynı olması gerkiyormuş.
features=['alcohol','colorIntensity',]
target=['quality (target)']
x=df[features]                        #seçtiğim özellikler
y=df[target]         #tahmin edilecek olan özellik

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.80,test_size=0.20,random_state=0) #datayı train ve test olarak ayırdım

scaler = MinMaxScaler() #seçtiğim özelliklerin değerleri yakın olmadığından scale yaptım
x_train = scaler.fit_transform(x_train) #train de ü işlem yapıldığı için fit,testte yapılmadığından transform
x_test = scaler.transform(x_test)

#seçtiğim y özelliği sebebiyle classification yapmak daha mantıklı.Decision Tree yapmaya karar verdim.
dt = DecisionTreeClassifier(max_depth = 4, random_state = 0) #çok karışık olamsını istemedim hem de yüksek accuracy istediğimden depth i 4 yaptım
dt= dt.fit(x_train,y_train)

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')


y_pred = dt.predict(x_test)
score = round(metrics.accuracy_score(y_test, y_pred), 3)
print("Accuracy:",score)   #doğruluk oranı


tree.plot_tree(dt);
fn=['alcohol','colorIntensity',]
cn=['1','2','3']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(dt,feature_names = fn,class_names=cn,filled = True);
fig.savefig('dt.png')
