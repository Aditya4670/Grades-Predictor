import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student-mat.csv")

categorical = ['school',\
               'sex',\
               'address',\
               'famsize',\
               'Pstatus',\
               'Mjob',\
               'Fjob',\
               'reason',\
               'guardian',\
               'schoolsup',\
               'famsup',\
               'paid',\
               'activities',\
               'nursery',\
               'higher',\
               'internet',\
               'romantic',
                ]

le = LabelEncoder()
df[categorical] = df[categorical].apply(lambda col: le.fit_transform(col))

X = df.drop("G3", axis=1)
Y = df["G3"]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=20)
model = LinearRegression()
model.fit(Xtrain, Ytrain)
print(model.score(Xtest, Ytest) * 100)
