import streamlit as st
import pandas as pd
import pycaret.regression as pr


@st.cache(persist=False,
          allow_output_mutation=True,
          suppress_st_warning=True,
          show_spinner=True)
def load_csv():
    df_input = pd.DataFrame()
    df_input = pd.read_csv(input, sep=None, engine='python', encoding='utf-8',
                           parse_dates=True,
                           infer_datetime_format=True)
    return df_input


if __name__ == "__main__":
    st.set_page_config(page_title="Python Key Influencers", page_icon="🔍", layout="wide")

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title(
        "Python Key Influencers"
    )
    st.subheader(
        """
         A simple XAI (eXplainable Artificial Intelligence) machine learning app inspired by Key Influencers module from Power BI. 
        """
    )
    with st.expander("What's going on?"):
        st.write(
            """
                We are using [*SHAP (SHapley Additive exPlanation)*](https://github.com/slundberg/shap) values, a new fancy game theoretical approach to analise machine learning algorithms such as Gradient Boosted Tree Models like [*LightGBM*](https://lightgbm.readthedocs.io/en/latest/).
                
                It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.
                
            """
        )

    st.markdown('-----')

    st.sidebar.title("Menu")

    st.sidebar.subheader("1. Upload your .csv")

    input = st.sidebar.file_uploader(label="Note: only .csv")

    if not input:
        st.sidebar.subheader("Or you can directly load a Complete Example")
        with st.sidebar.expander("Examples"):

            check1=st.checkbox("Ex 1. Price column in Automobile")
            check2=st.checkbox("Ex 2. Price in Diamond")
            check3=st.checkbox("Ex 3. Strength in Concrete")
            check4=st.checkbox("Ex 4. CNT in Bike")
            if check1+check2+check3+check4>1:
                st.sidebar.warning("Check only one example at time")


        if check1:
            from pycaret.datasets import get_data
            df = get_data('automobile')
            with st.expander('Explore data'):
                st.dataframe(df.head(10))
            with st.spinner("Analyzing..."):
                reg = pr.setup(
                    df,
                    target='price',
                    use_gpu=True,
                    silent=True,
                    feature_selection=True
                )

                lgbm = pr.create_model('lightgbm')
                fig = pr.interpret_model(lgbm)
                st.text('SHAP analysis of your Data')
                st.pyplot(fig)
                st.sidebar.success("Succesful Analysis")
        if check2:
            from pycaret.datasets import get_data

            df = get_data('diamond')
            with st.expander('Explore data'):
                st.dataframe(df.head(10))
            with st.spinner("Analyzing..."):
                reg = pr.setup(
                    df,
                    target='Price',
                    use_gpu=True,
                    silent=True,
                    feature_selection=True
                )

                lgbm = pr.create_model('lightgbm')
                fig = pr.interpret_model(lgbm)
                st.text('SHAP analysis of your Data')
                st.pyplot(fig)
                st.sidebar.success("Succesful Analysis")
        if check3:
            from pycaret.datasets import get_data

            df = get_data('concrete')
            with st.expander('Explore data'):
                st.dataframe(df.head(10))
            with st.spinner("Analyzing..."):
                reg = pr.setup(
                    df,
                    target='strength',
                    use_gpu=True,
                    silent=True,
                    feature_selection=True
                )

                lgbm = pr.create_model('lightgbm')
                fig = pr.interpret_model(lgbm)
                st.text('SHAP analysis of your Data')
                st.pyplot(fig)
                st.sidebar.success("Succesful Analysis")
        if check4:
            from pycaret.datasets import get_data

            df = get_data('bike')
            with st.expander('Explore data'):
                st.dataframe(df.head(10))
            with st.spinner("Analyzing..."):
                reg = pr.setup(
                    df,
                    target='cnt',
                    use_gpu=True,
                    silent=True,
                    feature_selection=True
                )

                lgbm = pr.create_model('lightgbm')
                fig = pr.interpret_model(lgbm)
                st.text('SHAP analysis of your Data')
                st.pyplot(fig)
                st.sidebar.success("Succesful Analysis")
    if input:
        df = load_csv()
        with st.expander('Explore data'):
            st.dataframe(df.head(10))
        columns = list(df.columns)
        st.sidebar.subheader("2. Select objective column")
        y_column_name = st.sidebar.selectbox("Note: only numerical", index=0, options=sorted(columns),
                                             key="Columna Objetivo")
        st.sidebar.subheader("3. Begin Analysis")
        if st.sidebar.button('Run'):
            with st.spinner("Analyzing..."):
                reg = pr.setup(
                    df,
                    target=y_column_name,
                    use_gpu=True,
                    silent=True,
                    feature_selection=True
                )

                lgbm = pr.create_model('lightgbm')
                fig = pr.interpret_model(lgbm)
                st.text('SHAP analysis of your Data')
                st.pyplot(fig)
                st.sidebar.success("Succesful Analysis")

    st.sidebar.header('About')
    st.sidebar.success(
        """
           Python Key Influencers app is maintained by 
           **Roger Pou López** while working as Data Scientist at [**Grupo Vall Companys IT department**](https://www.vallcompanys.es/). No property intended. If you like this app please star its
           [**GitHub**](https://github.com/rogerpou/Key-Influencers-App)
           repo, share it and feel free to open an issue if you find a bug 
           or if you want some additional features. 
        """
    )
