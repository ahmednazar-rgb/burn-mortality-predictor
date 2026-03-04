import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

data = [
[100,18,'F',0],
[35,44,'F',0],
[90,60,'F',0],
[30,31,'M',1],
[75,18,'M',0],
[13,6,'F',1],
[60,4,'M',0],
[20,36,'F',1],
[10,5,'M',1],
[8,0.75,'M',1],
[20,29,'F',1],
[100,32,'M',0],
[15,1.5,'M',1],
[15,17,'M',1],
[10,4,'M',1],
[15,25,'F',1],
[100,15,'F',0]
]

df = pd.DataFrame(data, columns=["Burn","Age","Sex","Outcome"])

X = df[['Burn','Age','Sex']]
y = df['Outcome']

preprocess = ColumnTransformer([
("num", StandardScaler(), ["Burn","Age"]),
("cat", OneHotEncoder(), ["Sex"])
])

model = Pipeline([
("prep", preprocess),
("clf", LogisticRegression())
])

model.fit(X,y)

st.title("Burn Mortality Predictor")

age = st.number_input("Age",0.0,100.0)
burn = st.slider("Burn %",0,100)
sex = st.selectbox("Sex",["M","F"])

if st.button("Predict"):
    input_df = pd.DataFrame([[burn,age,sex]],columns=["Burn","Age","Sex"])
    prob = model.predict_proba(input_df)[0][0]
    st.write("Mortality risk:",round(prob*100,2),"%")
