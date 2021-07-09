import streamlit as st
import pandas as pd
from sklearn import datasets
import plotly_express as px

st.set_page_config(
         page_title="Minha primeira Web APP", #1
         page_icon="🏛", #2
         layout="wide", #3
         initial_sidebar_state="expanded", #4
     )


@st.cache
def carrega_dados():
    iris_ds = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris_ds.data,
                           columns=["sepal_length", "sepal_width",
                                    "petal_length", "petal_width"])
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

    box_sepal_l = px.box(carrega_dados()[1], x='especie', y='sepal_length')
    box_sepal_w = px.box(carrega_dados()[1], x='especie', y='sepal_width')
    box_petal_l = px.box(carrega_dados()[1], x='especie', y='petal_length')
    box_petal_w = px.box(carrega_dados()[1], x='especie', y='petal_width')
    st.header("Exibir Boxplots")
    eda_box_plots = st.beta_expander("Box Plots")

    with eda_box_plots:
        bp1, bp2 = st.beta_columns(2)
        bp3, bp4 = st.beta_columns(2)

        bp1.header('sepal_length')
        bp1.plotly_chart(box_sepal_l)
        bp2.header('sepal_width')
        bp2.plotly_chart(box_sepal_w)
        bp3.header('petal_length')
        bp3.plotly_chart(box_sepal_l)
        bp4.header('petal_width')
        bp4.plotly_chart(box_sepal_w)

    hist_sepal_l = px.histogram(carrega_dados()[1], x='sepal_length', facet_col='especie')
    hist_sepal_w = px.histogram(carrega_dados()[1], x='sepal_width', facet_col='especie')
    hist_petal_l = px.histogram(carrega_dados()[1], x='petal_length', facet_col='especie')
    hist_petal_w = px.histogram(carrega_dados()[1], x='petal_width', facet_col='especie')

    st.header("Exibir Histogramas")
    eda_hist = st.beta_expander("Histogramas")

    with eda_hist:
        h1, h2 = st.beta_columns(2)
        h3, h4 = st.beta_columns(2)

        h1.header('sepal_length')
        h1.plotly_chart(hist_sepal_l)
        h2.header('sepal_width')
        h2.plotly_chart(hist_sepal_w)
        h3.header('petal_length')
        h3.plotly_chart(hist_petal_l)
        h4.header('petal_width')
        h4.plotly_chart(hist_petal_w)






