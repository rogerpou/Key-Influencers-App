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
    st.set_page_config(page_title="Key Influencers", page_icon="🔍", layout="wide")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title(
        "Python Key Influencers"
    )
    st.sidebar.title("Menu")
    st.sidebar.subheader("1. Upload your .csv")
    input = st.sidebar.file_uploader(label="Note: only .csv")
    if not input:
        st.write("Or you can Load Example")
        if st.sidebar.checkbox("Example 1. Price column in Automobile Dataset"):
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
                st.pyplot(fig)
                st.sidebar.success("Succesful Analysis")
            if st.sidebar.checkbox("Example 2. Price in Diamond Dataset"):
                from pycaret.datasets import get_data

                df = get_data('diamond')
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
                    st.pyplot(fig)
                    st.sidebar.success("Succesful Analysis")
                if st.sidebar.checkbox("Example 3. Strength in Concrete Dataset"):
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
                        st.pyplot(fig)
                        st.sidebar.success("Succesful Analysis")
                if st.sidebar.checkbox("Example 4. cnt in bike Dataset"):
                    from pycaret.datasets import get_data

                    df = get_data('concrete')
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
                st.pyplot(fig)
                st.sidebar.success("Succesful Analysis")
    st.sidebar.header('About')
    st.sidebar.warning(
        """
           Python Key Influencer  app is created and maintained by 
           **Roger Pou López**. If you like this app please star its
           [**GitHub**](https://github.com/rogerpou/Key-Influencers-App)
           repo, share it and feel free to open an issue if you find a bug 
           or if you want some additional features. Feel free to check my [**LinkedIn**](https://www.linkedin.com/in/roger-pou/)
        """
    )
