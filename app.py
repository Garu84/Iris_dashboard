import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# PAGE CONFIGURATION
st.set_page_config(page_title="Iris Species Classification",layout="wide")

# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv("Iris.csv")

df = load_data()

# TITLE
st.title("Iris Species Classification Dashboard")
st.write("Machine Learning dashboard for predicting Iris flower species.")

# PREPARE DATA
X = df.drop(["Species", "Id"], axis=1)
y = df["Species"]

# SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# TRAIN MODEL
model = RandomForestClassifier()
model.fit(X_train, y_train)

# PREDICTIONS
y_pred = model.predict(X_test)

# METRICS
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test,y_pred,average="weighted")
recall = recall_score(y_test,y_pred,average="weighted")

f1 = f1_score(y_test,y_pred,average="weighted")

# METRIC CARDS
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", round(accuracy, 2))
col2.metric("Precision", round(precision, 2))
col3.metric("Recall", round(recall, 2))
col4.metric("F1 Score", round(f1, 2))

# SIDEBAR INPUTS
st.sidebar.header("Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length",4.0,8.0,5.1)
sepal_width = st.sidebar.slider("Sepal Width",2.0,4.5,3.5)
petal_length = st.sidebar.slider("Petal Length",1.0,7.0,1.4)
petal_width = st.sidebar.slider("Petal Width",0.1,2.5,0.2)

# USER INPUT DATA
input_data = pd.DataFrame({
    "SepalLengthCm": [sepal_length],
    "SepalWidthCm": [sepal_width],
    "PetalLengthCm": [petal_length],
    "PetalWidthCm": [petal_width]})

# PREDICTION
prediction = model.predict(input_data)

# SHOW PREDICTION
st.subheader("Predicted Species")
st.success(prediction[0])

# SPECIES DISTRIBUTION
st.subheader("Species Distribution")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.countplot(data=df,x="Species",ax=ax1)
ax1.set_title("Species Distribution")
st.pyplot(fig1)

# SCATTERPLOT
st.subheader("Petal Length vs Petal Width")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df,x="PetalLengthCm",y="PetalWidthCm",hue="Species",ax=ax2)
ax2.set_title("Petal Length vs Petal Width")
st.pyplot(fig2)

# 3D SCATTER PLOT
st.subheader("3D Iris Visualization")
fig3d = px.scatter_3d(df,x="SepalLengthCm",y="SepalWidthCm",z="PetalLengthCm",color="Species")
st.plotly_chart(fig3d)

# DATASET PREVIEW
st.subheader("Dataset Preview")
st.dataframe(df)