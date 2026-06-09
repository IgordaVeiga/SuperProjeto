import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Impacto da IA na Vida do Estudante", layout="wide")


# =========================
# Leitura e preparação dos dados
# =========================
df = pd.read_csv("AI_Student_Life_india.csv")

# IMPORTANTE: Criamos um ID sequencial estático para corrigir o problema de contagem
df_original = df.copy()
df["Student_ID"] = range(1, len(df) + 1)

aba_dados, aba_classificacao = st.tabs([
"Análise de dados",
"Classificação probabilística",
])

# Verificações de qualidade dos dados
valores_ausentes = df.isnull().sum()
duplicatas_linhas = df.duplicated().sum()
duplicatas_id = df_original["Student_ID"].duplicated().sum() if "Student_ID" in df_original.columns else 0


# Tratamento de dados
df = df.drop_duplicates()


# =========================
# Cabeçalho
# =========================

with aba_dados:
    st.title("📊 Impacto da IA na Vida do Estudante")
    st.markdown(
        """
    Com a evolução constante da Inteligência Artificial, seu uso no ambiente estudantil tem crescido significativamente.
    Este dashboard busca entender **como**, **por que** e **com que frequência** os estudantes utilizam essas ferramentas.
    """
    )


    st.markdown("## ⚖️ Questões éticas")
    st.markdown(
        """
    Embora a IA possa auxiliar no aprendizado, seu uso exige responsabilidade.
    Quando utilizada como apoio, ela pode melhorar a produtividade e ampliar o acesso à informação.
    Por outro lado, quando substitui o esforço do estudante, pode prejudicar o aprendizado e a autonomia acadêmica.
    """
    )


    # =========================
    # Tratamento e qualidade dos dados
    # =========================
    st.markdown("## 🧹 Tratamento e qualidade dos dados")


    col1, col2, col3 = st.columns(3)
    col1.metric("Valores ausentes", int(valores_ausentes.sum()))
    col2.metric("Linhas totalmente duplicadas", int(duplicatas_linhas))
    col3.metric("IDs redundantes tratados", int(duplicatas_id))


    st.write("### Verificação de valores ausentes por coluna")
    st.dataframe(valores_ausentes.reset_index().rename(columns={"index": "Coluna", 0: "Valores ausentes"}), use_container_width=True)


    if valores_ausentes.sum() == 0 and duplicatas_linhas == 0:
        st.success(
            f"🛠️ **Justificativa Técnica:** O conjunto de dados original não apresentou valores ausentes. "
            f"No entanto, detectamos {duplicatas_id} IDs repetidos na chave primária. "
            f"Tratamos essa inconsistência gerando um índice sequencial limpo (1 a {len(df)}) para garantir a integridade exigida no projeto."
        )
    else:
        st.warning(
            "Foram encontrados problemas de qualidade nos dados. Neste caso, aplicamos remoção de duplicatas "
            "e o tratamento necessário antes das análises."
        )


    # =========================
    # Filtro principal
    # =========================
    atributos = [
        "Estudantes",
        "Education_Level",
        "City",
        "AI_Tool_Used",
        "Impact_on_Grades",
        "Satisfaction_Level",
    ]


    atributos_selecionados = st.multiselect(
        "Escolha o attribute para visualizar:",
        atributos,
        default=["City"]
    )


    # =========================
    # Educação
    # =========================
    if "Education_Level" in atributos_selecionados:
        st.header("🎓 Níveis de educação")


        st.subheader("Quantidade de estudantes por nível educacional")
        contador_ensino = df["Education_Level"].value_counts()
        st.bar_chart(contador_ensino)
        nivel_comum = contador_ensino.idxmax()
        st.markdown(
            f"O nível de escolaridade com maior presença no dataset é **{nivel_comum}**. "
            f"Isso mostra que esse grupo representa a maior parcela dos estudantes analisados."
        )


        st.subheader("Média de horas diárias de uso de IA por nível")
        media_estudantes_ia = df.groupby("Education_Level")["Daily_Usage_Hours"].mean().sort_values(ascending=False)
        grupo_que_mais_usa = media_estudantes_ia.idxmax()
        st.bar_chart(media_estudantes_ia)
        st.markdown(
            f"O grupo que utiliza IA por mais tempo é **{grupo_que_mais_usa}**, com média de **{media_estudantes_ia.max():.1f} horas por dia**. "
            f"Isso sugere maior integração da IA à rotina acadêmica desse nível de ensino."
        )


        st.subheader("Ferramenta de IA mais usada em cada nível educacional")
        most_used_ai_by_education = (
            df.groupby("Education_Level")["AI_Tool_Used"]
            .value_counts()
            .reset_index(name="Count")
        )
        most_used_ai_by_education = most_used_ai_by_education.loc[
            most_used_ai_by_education.groupby("Education_Level")["Count"].idxmax()
        ]
        st.bar_chart(most_used_ai_by_education, x="Education_Level", y="Count", color="AI_Tool_Used")


    # =========================
    # Ferramentas de IA
    # =========================
    if "AI_Tool_Used" in atributos_selecionados:
        st.header("🤖 Ferramentas utilizadas")


        st.subheader("Distribuição de propósito por ferramenta")
        analise_ferramenta = df.groupby(["AI_Tool_Used", "Purpose"]).size().reset_index(name="Quantidade")
        st.bar_chart(analise_ferramenta, x="AI_Tool_Used", y="Quantidade", color="Purpose")


        st.subheader("Total de horas diárias somadas por ferramenta")
        horas_diarias = df.groupby("AI_Tool_Used")["Daily_Usage_Hours"].sum().sort_values(ascending=False)
        horas_diarias_nome = horas_diarias.idxmax()
        st.bar_chart(horas_diarias)
        st.markdown(
            f"A ferramenta com maior volume total de uso é **{horas_diarias_nome}**, somando **{horas_diarias.max():.1f} horas**. "
            f"Isso reforça sua relevância entre os estudantes analisados."
        )


    # =========================
    # Cidades
    # =========================
    if "City" in atributos_selecionados:
        st.header("🌆 Aspectos das cidades")


        st.subheader("Qual cidade concentra mais usuários de IA?")
        contador_cidade = df.groupby("City")["AI_Tool_Used"].count().sort_values(ascending=False)
        st.bar_chart(contador_cidade)
        st.markdown(
            f"A cidade com maior número de estudantes utilizando IA é **{contador_cidade.idxmax()}**, com **{contador_cidade.max()} usuários registrados**. "
            f"Isso pode indicar maior acesso à tecnologia ou maior adesão a ferramentas digitais nessa região."
        )


        st.subheader("Ferramenta de IA mais usada em cada cidade")
        ai_cada_cidade = df.groupby("City")["AI_Tool_Used"].value_counts().reset_index(name="Count")
        ai_cada_cidade = ai_cada_cidade.loc[ai_cada_cidade.groupby("City")["Count"].idxmax()]
        st.bar_chart(ai_cada_cidade, x="City", y="Count", color="AI_Tool_Used")


        universitarios_cidade = df[df["Education_Level"] == "University"].groupby("City")["Student_ID"].count()
        escola_cidade = df[df["Education_Level"] == "School"].groupby("City")["Student_ID"].count()
        colegiais_cidade = df[df["Education_Level"] == "College"].groupby("City")["Student_ID"].count()
        media_cidade = df.groupby("City").size().mean()


        st.markdown(
            f"- A cidade com mais universitários é **{universitarios_cidade.idxmax()}**, com **{universitarios_cidade.max()} estudantes**.\n"
            f"- A cidade com mais estudantes de nível College é **{colegiais_cidade.idxmax()}**, com **{colegiais_cidade.max()} estudantes**.\n"
            f"- A cidade com mais alunos de School é **{escola_cidade.idxmax()}**, com **{escola_cidade.max()} estudantes**.\n"
            f"- A média de estudantes que utilizam IA por cidade é de **{media_cidade:.0f}**."
        )


    # =========================
    # Estudantes
    # =========================
    if "Estudantes" in atributos_selecionados:
        st.header("👩‍🎓 Perfil dos estudantes")


        st.subheader("Qual gênero mais utiliza IA para estudos?")
        contagem_genero = df["Gender"].value_counts()
        st.bar_chart(contagem_genero)
        st.markdown(
            f"O gênero com maior presença entre os usuários analisados é **{contagem_genero.idxmax()}**."
        )


        st.subheader("Qual idade aparece com mais frequência entre os usuários?")
        contagem_idade = df["Age"].value_counts().sort_index()
        idade_mais_comum = df["Age"].mode().iloc[0]
        st.bar_chart(contagem_idade)
        st.markdown(
            f"A idade que mais aparece entre os estudantes do dataset é **{idade_mais_comum} anos**."
        )


    # =========================
    # Impacto nas notas
    # =========================
    if "Impact_on_Grades" in atributos_selecionados:
        st.header("📈 Impacto da IA nas notas")


        impacto_notas = df["Impact_on_Grades"].value_counts()
        st.bar_chart(impacto_notas)
        principal_impacto = impacto_notas.idxmax()
        st.markdown(
            f"O impacto mais frequente registrado foi **{principal_impacto}**. "
            f"Isso sugere que o uso da IA pode estar associado a mudanças perceptíveis no desempenho acadêmico."
        )


    # =========================
    # Satisfação
    # =========================
    if "Satisfaction_Level" in atributos_selecionados:
        st.header("😊 Nível de satisfação dos estudantes")


        satisfacao = df["Satisfaction_Level"].value_counts()
        st.bar_chart(satisfacao)
        satisfacao_mais_comum = satisfacao.idxmax()
        st.markdown(
            f"O nível de satisfação mais frequente é **{satisfacao_mais_comum}**. "
            f"Isso ajuda a entender como os estudantes percebem a utilidade da IA em sua rotina."
        )


    # =========================
    # Conclusão
    # =========================
    st.markdown("---")
    st.markdown(
        """
    ### ✅ Conclusão
    A Inteligência Artificial já faz parte da rotina dos estudantes e tende a crescer ainda mais.
    Os dados mostram padrões relevantes de uso, diferenças entre perfis e possíveis impactos acadêmicos.
    Além disso, a análise reforça a importância de um uso consciente, ético e equilibrado dessas ferramentas.


    ### 🙌 Obrigado!
    """
    )

# ==========================================
# SEGUNDA ABA DESENVOLVIDA SEGUINDO A LAUDA
# ==========================================
with aba_classificacao:
    st.title("🧮 Classificação Probabilística e Modelos")
    
    # 1. Interface de Entrada de Valores
    st.header("1. Defina o Perfil do Estudante para Predição")
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        f_gender = st.selectbox("Gênero do Aluno:", df["Gender"].unique())
    with col_in2:
        f_edu = st.selectbox("Nível de Escolaridade:", df["Education_Level"].unique())
    with col_in3:
        f_tool = st.selectbox("Ferramenta de IA:", df["AI_Tool_Used"].unique())

    # 2. Cálculo Manual do Teorema de Bayes
    st.header("2. Resolução do Teorema de Bayes Manual")
    st.latex(r"P(Classe|Perfil) = \frac{P(Perfil|Classe) \cdot P(Classe)}{P(Perfil)}")
    
    classes_alvo = df["Impact_on_Grades"].unique()
    total_registros = len(df)
    resultados_bayes = {}
    
    for c in classes_alvo:
        df_c = df[df["Impact_on_Grades"] == c]
        p_c = len(df_c) / total_registros  # Probabilidade a priori
        
        # Verossimilhanças com Suavização de Laplace (+1) para não quebrar
        p_gender_c = (len(df_c[df_c["Gender"] == f_gender]) + 1) / (len(df_c) + df["Gender"].nunique())
        p_edu_c = (len(df_c[df_c["Education_Level"] == f_edu]) + 1) / (len(df_c) + df["Education_Level"].nunique())
        p_tool_c = (len(df_c[df_c["AI_Tool_Used"] == f_tool]) + 1) / (len(df_c) + df["AI_Tool_Used"].nunique())
        
        resultados_bayes[c] = (p_gender_c * p_edu_c * p_tool_c) * p_c

    soma_numeradores = sum(resultados_bayes.values())
    df_bayes_final = pd.DataFrame([
        {"Classe (Impacto)": k, "Probabilidade Posteriori (%)": round((v / soma_numeradores) * 100, 2)}
        for k, v in resultados_bayes.items()
    ])
    
    st.write("### Probabilidades Calculadas por Bayes para cada tipo de Impacto:")
    st.dataframe(df_bayes_final, use_container_width=True)

    # 3. Modelos Extras (Árvore de Decisão e KNN) via scikit-learn
    features_cols = ["Gender", "Education_Level", "AI_Tool_Used"]
    df_ml = df[features_cols + ["Impact_on_Grades"]].dropna().copy()
    
    encoders = {}
    for col in df_ml.columns:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        encoders[col] = le
        
    X = df_ml[features_cols]
    y = df_ml["Impact_on_Grades"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Árvore de Decisão
    clf_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf_tree.fit(X_train, y_train)
    acc_tree = accuracy_score(y_test, clf_tree.predict(X_test))
    
    # KNN
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(X_train, y_train)
    acc_knn = accuracy_score(y_test, clf_knn.predict(X_test))
    
    # Processando a entrada do usuário para os modelos
    g_enc = encoders["Gender"].transform([f_gender])[0]
    e_enc = encoders["Education_Level"].transform([f_edu])[0]
    t_enc = encoders["AI_Tool_Used"].transform([f_tool])[0]
    
    pred_tree_text = encoders["Impact_on_Grades"].inverse_transform([clf_tree.predict([[g_enc, e_enc, t_enc]])[0]])[0]
    pred_knn_text = encoders["Impact_on_Grades"].inverse_transform([clf_knn.predict([[g_enc, e_enc, t_enc]])[0]])[0]

    # 4. Comparação Visual Exigida pela Lauda
    st.header("3. Painel de Comparação entre os Modelos")
    
    classe_bayes_provavel = df_bayes_final.loc[df_bayes_final["Probabilidade Posteriori (%)"].idxmax(), "Classe (Impacto)"]
    porcentagem_bayes = df_bayes_final.loc[df_bayes_final["Probabilidade Posteriori (%)"].idxmax(), "Probabilidade Posteriori (%)"]
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Veredito Bayes (Manual)", classe_bayes_provavel, f"{porcentagem_bayes}% de certeza")
    with col_m2:
        st.metric("Predição Árvore de Decisão", pred_tree_text, f"Acurácia de Teste: {acc_tree*100:.1f}%")
    with col_m3:
        st.metric("Predição KNN (3 Vizinhos)", pred_knn_text, f"Acurácia de Teste: {acc_knn*100:.1f}%")

    st.info(
        "💡 **Dica de Defesa:** O Teorema de Bayes calcula as probabilidades utilizando a base inteira de forma direta. "
        "Já os modelos de Machine Learning dividem os dados em Treino e Teste (75/25) para aprender padrões, o que valida a capacidade deles de prever perfis que o sistema ainda não conhece."
    )