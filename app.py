import streamlit as st
import pandas as pd
from sklearn import datasets
import plotly_express as px

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#Configurações de página
st.set_page_config(
         page_title="ML Web App",
         page_icon="🏛",
         layout="wide", #Outras opções: centered
         initial_sidebar_state="expanded", #Outras opções [auto, collapsed]
     )

#carregando os dados e transformando-os em um pandas datframe
@st.cache
def carrega_dados():
    iris_ds = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris_ds.data,
                           columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    iris_df["classe"] = iris_ds.target
    iris_df['especie'] = iris_df['classe'].apply(lambda x: iris_ds.target_names[x])
    return (iris_ds, iris_df)

st.sidebar.title('Exibir EDA')
exibir_eda = st.sidebar.checkbox("EDA", value=False)

if exibir_eda:
    st.title("EDA")
    st.write("Informações sobre o Iris dataset: [link](https://en.wikipedia.org/wiki/Iris_flower_data_set)")
    st.header("Exibir Dataframe com os dados e a descrição destes dados")
    eda_expander = st.beta_expander("Dataframe e Descrição")
    with eda_expander:
        df = carrega_dados()[1]
        st.header("Iris Dataframe")
        st.dataframe(df)
        st.header("Descrição do Dataframe")
        tipo_flor = st.radio("Selecione o tipo de Flor",['Setosa','Versicolor','Virginica'])
        st.table(df[df['especie'] == str.lower(tipo_flor)].describe())


    box_sepal_l = px.box(carrega_dados()[1], x = 'especie', y = 'sepal_length')
    box_sepal_w = px.box(carrega_dados()[1], x = 'especie', y = 'sepal_width')
    box_petal_l = px.box(carrega_dados()[1], x = 'especie', y = 'petal_length')
    box_petal_w = px.box(carrega_dados()[1], x = 'especie', y = 'petal_width')

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

    st.header("Exibir Histogramas")
    hist_sepal_l = px.histogram(carrega_dados()[1], x = 'sepal_length', facet_col='especie')
    hist_sepal_w = px.histogram(carrega_dados()[1], x = 'sepal_width', facet_col='especie')
    hist_petal_l = px.histogram(carrega_dados()[1], x = 'petal_length', facet_col='especie')
    hist_petal_w = px.histogram(carrega_dados()[1], x = 'petal_width', facet_col='especie')

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

#Modelos de ML--------------------------------------------------------------------------------------------------------

X = carrega_dados()[0].data
y= carrega_dados()[0].target

st.sidebar.title("Treinamento")
form_modelo = st.sidebar.form(key = 'modelos')
modelos_sel = form_modelo.multiselect("Modelos a serem treinados", ['RandomForest', 'Regressão Logística', 'SVM', 'KNN'])
nr_quebras = form_modelo.slider("Nr de folds",2, 10, 5, step=1)
percentual_validacao = form_modelo.slider("Percentual de validação", 0.1, 0.5, 0.2, step = 0.05)
X_treino, X_validacao, y_treino, y_validacao = train_test_split(X, y, test_size=percentual_validacao)
treinar = form_modelo.form_submit_button("Treinar modelos!")

lista_modelos = {'RandomForest': RandomForestClassifier(),
                 'Regressão Logística': LogisticRegression(),
                 'SVM': SVC(),
                 'KNN': KNeighborsClassifier()}

@st.cache
def treinamento(modelos_sel, lista_modelos, nr_quebras, X_treino, y_treino):

    resultados = {}
    for modelo in modelos_sel:
        fold = StratifiedKFold(n_splits=nr_quebras, shuffle=True)
        resultado_cv = cross_val_score(lista_modelos[modelo], X_treino, y_treino, cv=fold, scoring='accuracy')
        resultados[modelo] = [resultado_cv.mean(), resultado_cv.std()]

    return pd.DataFrame.from_dict(data=resultados, orient='index', columns=['Média', 'Desvio_Padrão'])

if treinar:
    #treinamento(modelos_sel, lista_modelos, nr_quebras, X_treino, y_treino)
    st.title("Resultados do Treinamento - Acurácia dos Modelos")
    st.table(treinamento(modelos_sel, lista_modelos, nr_quebras, X_treino, y_treino))

st.sidebar.title("Validação")
form_valida = st.sidebar.form(key = 'valida')
modelos_validar = form_valida.multiselect("Modelos a serem validados",['RandomForest', 'Regressão Logística', 'SVM', 'KNN'])
validar = form_valida.form_submit_button("Validar Modelos!")

@st.cache
def validacao(modelos_validar):
    resultados_v = []
    for nome_modelo_v in modelos_validar:
        modelo_v = lista_modelos[nome_modelo_v]
        modelo_v.fit(X_treino,y_treino)
        predicoes = modelo_v.predict(X_validacao)
        acuracia_validacao = accuracy_score(y_validacao, predicoes)
        precisao_validacao = precision_score(y_validacao, predicoes, average=None)
        recall_validacao = recall_score(y_validacao, predicoes, average=None)
        matriz_conf_validacao = confusion_matrix(y_validacao, predicoes)
        resultados_v.append([acuracia_validacao, precisao_validacao,recall_validacao, matriz_conf_validacao,nome_modelo_v])

    return resultados_v

if validar:
    #validacao(modelos_validar)
    st.title("Resultados do Treinamento - Acurácia dos Modelos")
    st.table(treinamento(modelos_sel, lista_modelos, nr_quebras, X_treino, y_treino))
    rv = validacao(modelos_validar)
    for i in rv:
        st.title(i[4])
        acr, prc, rcl, mcf = st.beta_columns(4)
        acr.header("Acurácia")
        acr.write(i[0])
        prc.header("Precisão")
        prc.write(i[1])
        rcl.header("Recall")
        rcl.write(i[2])
        mcf.header("Matriz de Confusão")
        mcf.write(i[3])
        st.subheader('Class 0 = Setosa   /   Class 1 = Versicolor   /   Class 2 = Virginica')






























