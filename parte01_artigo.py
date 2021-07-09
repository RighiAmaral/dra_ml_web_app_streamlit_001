import streamlit as st
import pandas as pd
from sklearn import datasets
import plotly_express as px


@st.cache
def carrega_dados():
    iris_ds = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris_ds.data,
                           columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    iris_df["classe"] = iris_ds.target
    iris_df['especie'] = iris_df['classe'].apply(lambda x: iris_ds.target_names[x])
    return (iris_ds, iris_df)

st.sidebar.title('Exibir EDA')
exibir_eda = st.sidebar.checkbox("EDA")

if exibir_eda:
    st.title("EDA")
    st.write("Informações sobre o Iris dataset: [link](https://en.wikipedia.org/wiki/Iris_flower_data_set)")
    st.header("Exibir Dataframe com os dados e a descrição destes dados")
    eda_expander = st.beta_expander("Dataframe e Descrição")
    with eda_expander:
        st.title("Iris Dataframe")
        df = carrega_dados()[1]
        st.dataframe(df)
        st.title("Descrição do Dataframe")
        tipo_flor = st.radio("Selecione o tipo de Flor",['Setosa','Versicolor','Virginica'])
        st.table(df[df['especie'] == str.lower(tipo_flor)].describe())
