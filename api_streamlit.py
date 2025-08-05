# Data Explorer Packages
import pandas as pd
import streamlit as st
import wbdata
import importlib
import requests
import matplotlib.pyplot as plt
import io
import urllib.parse
import time

# Text Summarization Packages
import subprocess
import os
from tempfile import NamedTemporaryFile
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import fitz  # PyMuPDF
import markdown

# 🔐 Password protection
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["PASSWORD"]:
            st.session_state["authenticated"] = True
            st.session_state["password"] = ""
        else:
            st.session_state["authenticated"] = False
            st.error("❌ Incorrect password")

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔒 Internal Access Only")
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        st.stop()

check_password()

# Initialize session state variables for api tool
if 'tool' not in st.session_state:
    st.session_state.tool = None
if 'dfs' not in st.session_state:
    st.session_state.dfs = []
if 'reset' not in st.session_state:
    st.session_state.reset = False
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'filter_mode' not in st.session_state:
    st.session_state.filter_mode = None
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = None
if 'current_md' not in st.session_state:
    st.session_state.current_md = None

# Initialize clustering session state variables
if 'cluster_step' not in st.session_state:
    st.session_state.cluster_step = None
if 'cluster_dataset' not in st.session_state:
    st.session_state.cluster_dataset = None
if 'cluster_cols' not in st.session_state:
    st.session_state.cluster_cols = None
if 'numerical_data' not in st.session_state:
    st.session_state.numerical_data = None
if 'scaled_data' not in st.session_state:
    st.session_state.scaled_data = None
if 'kmeans_fitted' not in st.session_state:
    st.session_state.kmeans_fitted = None
if 'cat_cols_to_add_back' not in st.session_state:
    st.session_state.cat_cols_to_add_back = []
if 'cat_data' not in st.session_state:
    st.session_state.cat_data = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 0
if 'cluster_confirmed' not in st.session_state:
    st.session_state.cluster_confirmed = False
    
def load_google_sheet(file_id: str, sheet_name: str) -> pd.DataFrame:
    encoded_sheet_name = urllib.parse.quote(sheet_name)
    url = f"https://docs.google.com/spreadsheets/d/{file_id}/gviz/tq?tqx=out:csv&sheet={encoded_sheet_name}"
    df = pd.read_csv(url)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

# Metadata files for World Bank and WHO
# wb_metadata = '/Users/rome/Documents/Medikana/Database Project/DataBase Index - WB Dataset Index.csv'
# who_metadata = None
GOOGLE_SHEET_ID = st.secrets["GOOGLESHEETID"]
METADATA_FILES = {
    'World Bank': 'WB Dataset Index',
    # 'World Health Organization': 'WHO Dataset Index'
}

# Country Codes for Analysis
COUNTRIES = ['ARG', 'BOL', 'BRA', 'CHL', 'COL', 'ECU', 'GUY', 
             'PRY', 'PER', 'SUR', 'URY', 'VEN', 'BLZ', 'CRI', 
             'SLV', 'GTM', 'HND', 'NIC', 'PAN', 'MEX', 'CUB', 
             'DOM', 'USA', 'CAN']

# Utility to cache metadata
@st.cache_data
def load_metadata(source_key: str) -> pd.DataFrame:
    sheet_name = METADATA_FILES[source_key]
    return load_google_sheet(GOOGLE_SHEET_ID, sheet_name)

# Helper to query the World Bank API using wbdata
def fetch_wb(indicator_code: str, countries: list) -> pd.Series:
    """Fetch data from WB using wbdata library for specified countries."""
    try:
        importlib.reload(wbdata)  # Avoid SQLite threading issue

        data_series = wbdata.get_series(indicator_code, country=countries)
        if data_series is None or data_series.empty:
            return pd.DataFrame(columns=["date", "country", "value"])

        df = data_series.reset_index()

        # Ensure 'date' column exists
        if "date" not in df.columns and "year" in df.columns:
            df.rename(columns={"year": "date"}, inplace=True)

        # Identify value column dynamically
        value_col = [col for col in df.columns if col not in ["date", "country"]]
        if value_col:
            df.rename(columns={value_col[0]: "value"}, inplace=True)
        else:
            df["value"] = None  # No data

        return df[["date", "country", "value"]]

    except Exception as e:
        st.error(f"Error fetching data for {indicator_code}: {e}")
        return pd.DataFrame(columns=["date", "country", "value"])

# Placeholder for WHO API
def fetch_who(indicator_code: str, **kwargs):
    st.warning('WHO API fetching not yet implemented')
    return pd.Series()

# helper to return copy of base metadata when dealing with st.session_state.current_md updates
def get_base_metadata(api_choice):
    return load_metadata(api_choice).copy()

# Fetchers dictionary to map data sources to their respective fetch functions
FETCHERS = {
    'World Bank': fetch_wb,
    'World Health Organization': fetch_who
}

# PDF generator from AI marksdown summary helper
def generate_pdf_from_markdown(md_text):
    html_text = markdown_to_html(md_text) # convert md to html
    doc = fitz.open()
    page = doc.new_page()
    
    # Define a rectangle for text placement
    rect = fitz.Rect(50, 50, 550, 800)
    page.insert_htmlbox(rect, html_text)

    # Save to bytes
    pdf_bytes = io.BytesIO()
    doc.save(pdf_bytes)
    pdf_bytes.seek(0)
    return pdf_bytes.getvalue()
    
def markdown_to_html(md_text):
    return markdown.markdown(md_text)

################################################   STREAMLIT APP   ###################################################
if not st.session_state.tool:
    st.title('Welcome to the Medikana Research Tool!')
    st.subheader('Please choose a tool to use:')
    tool_options = ['Select a tool...', 'Data Explorer', 'Text Summarizer', 'Segmentation/Clustering']
    tool_choice = st.selectbox('Select a tool:', tool_options)

    if tool_choice == 'Select a tool...':
        st.info("Please select a tool from the dropdown menu above to get started.")
    else:
        st.session_state.tool = tool_choice
        st.rerun()

else:
    if st.session_state.tool == 'Data Explorer':
        # success_message = st.empty()
        # success_message.success("You have selected the Data Explorer tool. Use the sidebar to explore data by indicator names or filters.")
        # time.sleep(1)
        # success_message.empty()
        if st.button("⬅ Back to Home"):
            st.session_state.tool = None
            st.rerun()

    ############################### Data Explorer Tool #####################################

        st.title('Medikana Health API Data Explorer (Beta V.2.0)')

        # ---- SIDEBAR UI ---- #
        with st.sidebar:
            if st.button("Reset Data Explorer"):
                st.session_state.dfs = []
                # st.session_state.dataset_names = []
                st.session_state.reset = True
                st.session_state.mode = None
                st.session_state.filter_mode = None
                st.session_state.filters_applied = None
                st.session_state.current_md = None
                st.rerun()

            st.markdown("Choose Method of Exploration")
            if st.button("Explore by Indicator Names"):
                st.session_state.mode = 'by_names'

            elif st.button("Explore by Filters"):
                st.session_state.mode = 'filters'

            if st.session_state.mode == 'by_names':
                st.markdown("### Search by Indicator Names")
                # search bar that searches both indicator codes and names in WHO and WB metadata
                search_query = st.text_input("Enter Indicator Name", key='search_query')
                if search_query:
                    results = {}
                    for source, path in METADATA_FILES.items():
                        md = load_metadata(source)
                        # Search in both 'code' and 'name' columns and return if matches in either
                        matches = md[md['name'].str.contains(search_query, case=False, na=False)]
                        # get tuple of name, code pairs in matches df
                        for _, row in matches.iterrows():
                            results[row['name']] = (row['code'], source)
                    if results:
                        # first create single select box with all the indicator names to select from then whichever chosen are df to fetch and concat
                        selected_indicator = st.selectbox("Select Indicators", ["Select..."] + list(results.keys()))
                        if selected_indicator != "Select...":
                            # go into fetch function to return data then filter by selected indicator
                            fetcher = FETCHERS[results[selected_indicator][1]]
                            code = results[selected_indicator][0]
                            series_data = fetcher(code, COUNTRIES)
                            st.session_state.dfs = [(selected_indicator, series_data.reset_index())]
                            # st.session_state.dataset_names = [selected_indicator]

                                    

            if st.session_state.mode == 'filters':                
                st.markdown("### Explore by Filters")
                api_choice = st.selectbox('Choose Data Source/API', list(METADATA_FILES.keys()))
                md = load_metadata(api_choice)
                st.session_state.current_md = md  # store current metadata for further use

                st.markdown("Choose Filter to Explore")
                if st.button('Gender'):
                    st.session_state.filter_mode = 'gender'
                elif st.button('Age Group'):
                    st.session_state.filter_mode = 'age_group'
                elif st.button('Rural / Urban'):    
                    st.session_state.filter_mode = 'rural_urban'
                elif st.button('Quarter'):  
                    st.session_state.filter_mode = 'quarter'
                elif st.button('Segment'):
                    st.session_state.filter_mode = 'segment'

                # Filter selections
                if st.session_state.filter_mode == 'gender':
                    base_md = get_base_metadata(api_choice)
                    gender_filter = st.selectbox('Gender (optional)', ['KEEP ALL'] + sorted(base_md['gender'].dropna().unique().tolist()))
                    if gender_filter != 'KEEP ALL':
                        st.session_state.current_md = base_md[base_md['gender'] == gender_filter]
                        st.session_state.filter_mode = 'segment'
                        st.session_state.filters_applied = 'Gender'
                    else:
                        st.session_state.current_md = base_md
                    
                elif st.session_state.filter_mode == 'age_group':
                    base_md = get_base_metadata(api_choice)
                    age_group_filter = st.selectbox('Age group (optional)', ['KEEP ALL'] + sorted(base_md['agegroup'].dropna().unique().tolist()))
                    if age_group_filter != 'KEEP ALL':
                        st.session_state.current_md = base_md[base_md['agegroup'] == age_group_filter]
                        st.session_state.filter_mode = 'segment'  # go into segment filter with modified st.session_state.current_md to be for age group
                        st.session_state.filters_applied = 'Age Group'
                    else:
                        st.session_state.current_md = base_md
            
                elif st.session_state.filter_mode == 'rural_urban':
                    base_md = get_base_metadata(api_choice)
                    rural_urban_filter = st.selectbox('Rural / Urban (optional)', ['KEEP BOTH'] + sorted(base_md['rural/urban'].dropna().unique().tolist()))
                    if rural_urban_filter != 'KEEP BOTH':
                        st.session_state.current_md = base_md[base_md['rural/urban'] == rural_urban_filter]
                        st.session_state.filter_mode = 'segment'  # go into segment filter with modified st.session_state.current_md to be for rural/urban
                        st.session_state.filters_applied = 'Rural / Urban'
                    else:
                        st.session_state.current_md = base_md
                    
                elif st.session_state.filter_mode == 'quarter':
                    base_md = get_base_metadata(api_choice)
                    quarter_filter = st.selectbox('Quarter (optional)', ['KEEP ALL YEARS/QUARTERS'] + sorted(base_md['quarter'].dropna().unique().tolist()))
                    if quarter_filter != 'KEEP ALL YEARS/QUARTERS':
                        st.session_state.current_md = base_md[base_md['quarter'] == quarter_filter]
                        st.session_state.filter_mode = 'segment'
                        st.session_state.filters_applied = 'Quarter/Yearly'
                    else:
                        st.session_state.current_md = base_md
                
                if st.session_state.filter_mode == 'segment':
                    if st.session_state.current_md is None:
                        st.session_state.current_md = get_base_metadata(api_choice)
                    segment = st.selectbox('Segment', ['Select...'] + sorted(st.session_state.current_md['segment'].dropna().unique().tolist()))
                    if segment != 'Select...':
                        st.write(f'Filters Applied: {st.session_state.filters_applied}')
                        # st.session_state.current_st.session_state.current_md = st.session_state.current_md  # store current metadata for further use
                        st.session_state.current_md = st.session_state.current_md[st.session_state.current_md['segment'] == segment]
                        md_seg = st.session_state.current_md
                        category = st.selectbox('Category', ['Select...'] + sorted(md_seg['category'].dropna().unique().tolist()))

                        if category != 'Select...':
                            md_cat = md_seg[md_seg['category'] == category]

                            if md_cat['subcategory'].isnull().all():
                                subcategory = ''
                                md_sub = md_cat
                            else:
                                subcat_options = md_cat['subcategory'].dropna().unique().tolist()
                                subcategory = st.selectbox('Sub-category', ['Select...'] + sorted(subcat_options))
                                md_sub = md_cat if subcategory == 'Select...' else md_cat[md_cat['subcategory'] == subcategory]

                            metric = st.selectbox('Metric', ['Select...'] + sorted(md_sub['metric'].dropna().unique().tolist()))

                            if metric != 'Select...':
                                md_metric = md_sub[md_sub['metric'] == metric]

                                if md_metric['submetric'].isnull().all():
                                    submetric = ''
                                    md_submetric = md_metric
                                else:
                                    submetric_options = md_metric['submetric'].dropna().unique().tolist()
                                    submetric = st.selectbox('Submetric (optional)', ['Select...'] + sorted(submetric_options))
                                    md_submetric = md_metric if submetric == 'Select...' else md_metric[md_metric['submetric'] == submetric]

                                fetch_clicked = st.button('Fetch data')

                                if fetch_clicked:
                                    query_df = md_submetric.copy()

                                    if query_df.empty:
                                        st.error('No indicator codes found for the selected filters.')
                                    else:
                                        st.success(f'Found {len(query_df)} indicator code(s). Fetching data...')
                                        fetcher = FETCHERS[api_choice]
                                        data_frames = []
                                        for _, row in query_df.iterrows():
                                            code = row['code']
                                            name = row['name']
                                            series_data = fetcher(code, COUNTRIES)
                                            if not series_data.empty:
                                                data_frames.append((name, series_data))

                                        if data_frames:
                                            st.session_state.dfs = data_frames

        if st.session_state.dfs is not None:
            df_all = st.session_state.dfs

            for name, df in df_all:
                if "date" not in df.columns:
                    st.error(f"No 'date' column found in dataset for {name}.")
                    continue
                
                st.markdown(name)
                # show the pct and num of missing data points in df
                total_data_points = len(df)
                num_missing = df.isna().sum().sum()
                pct_missing = (num_missing / total_data_points) * 100
                st.write("Data Summary:")
                st.write(f"Total Data Points: {total_data_points}")
                st.write(f"Missing Data Points: {num_missing} ({pct_missing:.2f}%)")

                # filter by year logic
                year_options = sorted(df['date'].unique(), reverse=True)
                select_all_option = 'Select All Years'
                selected = st.multiselect('Select Years to Display', [select_all_option] + year_options)
                if select_all_option in selected:
                    selected_years = year_options  # all years
                else:
                    selected_years = selected
                df_filtered = df[df['date'].isin(selected_years)]

                # show final df
                pivot_df = df_filtered.pivot(index='country', columns='date', values='value')
                pivot_df = pivot_df[sorted(pivot_df.columns, reverse=True)]  # Sort years descending
                st.dataframe(pivot_df, use_container_width=True, height=600)

                if st.button(f"Show Scatter Plot for: {name}"):
                    fig, ax = plt.subplots(figsize=(12, 6))
                    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'P', '*', 'X', 'h']
                    for i, country in enumerate(df_filtered['country'].unique()):
                        subset = df_filtered[df_filtered['country'] == country]
                        ax.plot(subset['date'], subset['value'], marker=markers[i % len(markers)], label=country)
                    ax.set_title(name, fontsize=12)
                    ax.set_xlabel("Year", fontsize=10)
                    ax.set_ylabel("Value", fontsize=10)
                    ax.tick_params(axis='x', labelrotation=45, labelsize=8)
                    ax.tick_params(axis='y', labelsize=8)
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
                    st.pyplot(fig)

                    # Add download button for plot
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png', bbox_inches='tight')
                    st.download_button("Download Plot as PNG", data=img_buffer.getvalue(), file_name=f"{name}_plot.png", mime="image/png")

    ########################### End of Data Explorer Tool Logic ##############################


    ############################### Text Summarizer Tool #####################################
    elif st.session_state.tool == 'Text Summarizer':
        # success_message = st.empty()
        # success_message.success("You have selected the Text Summarizer tool. Use the sidebar to summarize text documents.")
        # time.sleep(1)
        # success_message.empty()
        if st.button("⬅ Back to Home"):
            st.session_state.tool = None
            st.rerun()

        # Placeholder for text summarization logic
        st.set_page_config(layout='wide', page_title='Medikana Text Summarizer Tool')
        st.write("This tool will allow you to summarize text documents using AI models.")

        # Add your text summarization logic here
        MODEL = 'gpt-4o-mini'
        
        def move_file_to_downloads(pdf_file_path):
            downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
            destination_path = os.path.join(downloads_path, os.path.basename(pdf_file_path))
            shutil.move(pdf_file_path, destination_path)
            return destination_path

        def load_and_summarize(file, prompt_template):
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                file_path = tmp.name
            
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                # prompt_template = """
                # Write a business report from the following earnings call transcript:
                # {text}

                # Use the following Markdown format:
                # # Insert Descriptive Report Title

                # ## Earnings Call Summary
                # Use 3 to 7 numbered bullet points

                # ## Important Financials:
                # Describe the most important financials discussed during the call. Use 3 to 5 numbered bullet points.

                # ## Key Business Risks
                # Describe any key business risks discussed on the call. Use 3 to 5 numbered bullets.

                # ## Conclusions
                # Conclude with any overarching business actions that the company is pursuing that may have positive or negative implications and what those implications are. 
                # """
                
                prompt = PromptTemplate.from_template(prompt_template)
                model = ChatOpenAI(
                    model=MODEL,
                    temperature=0,
                    api_key=st.secrets["OPENAI"],
                )

                llm_chain = LLMChain(llm=model, prompt=prompt)
                stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
                response = stuff_chain.invoke(docs)
                
            finally:
                os.remove(file_path)

            return response['output_text']

        # Streamlit Interface
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Upload a PDF document:')
            uploaded_file = st.file_uploader("Choose a file", type="pdf", key="file_uploader")
            prompt_template = st.text_area('Please enter your Open AI Prompt Template for summarization:')
            if uploaded_file and prompt_template:
                summarize_flag = st.button('Summarize Document', key="summarize_button")
            else:
                st.warning("Please upload a PDF file and enter a prompt template to summarize the document.")

        if uploaded_file and prompt_template and summarize_flag:
            with col2:
                with st.spinner('Summarizing...'):
                    summary = load_and_summarize(uploaded_file, prompt_template)
                    st.subheader('Summarization Result:')
                    st.markdown(summary)

                    pdf_data = generate_pdf_from_markdown(summary)
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_data,
                        file_name="summary.pdf",
                        mime="application/pdf"
                    )

        else:
            with col2:
                st.write("No file uploaded. Please upload a PDF file to proceed.")


    ############################### End of Text Summarizer Tool #####################################


    ################################## Segmentation/Clustering Tool #####################################
    elif st.session_state.tool == 'Segmentation/Clustering':

        if st.button("⬅ Back to Home"):
            st.session_state.tool = None
            st.session_state.cluster_step = None
            st.session_state.cluster_dataset = None
            st.session_state.cluster_cols = None
            st.session_state.numerical_data = None
            st.session_state.scaled_data = None
            st.session_state.kmeans_fitted = None
            st.session_state.cat_cols_to_add_back = []
            st.session_state.cat_data = None
            st.session_state.scaler = None
            st.session_state.n_clusters = None

            st.rerun()

        # Placeholder for segmentation/clustering logic
        # st.set_page_config(layout='wide', page_title='Medikana Segmentation/Clustering Tool')
        st.write("This tool will allow you to segment or cluster data based on various criteria.")

        # Add your segmentation/clustering logic here
        MODEL = 'gpt-4o-mini'

        # First we need to load the dataset that user wants to segment/cluster
        if st.session_state.cluster_dataset is None:
            st.session_state.cluster_step = 'load_dataset'
            st.subheader('Load Dataset for Segmentation/Clustering')
            # File uploader for dataset
            if st.session_state.cluster_step == 'load_dataset':
                dataset = st.file_uploader("Choose a file", type="csv", key="file_uploader")
                if dataset is not None:
                    try:
                        st.session_state.cluster_dataset = pd.read_csv(dataset, header=0) # ensure header is there
                        st.success("Dataset loaded successfully!")
                        st.session_state.cluster_step = 'view_dataset'
                    except Exception as e:
                        st.error(f"Error loading dataset: {e}")
                else:
                    st.warning("Please upload a CSV file to proceed.")

        if st.session_state.cluster_step == 'view_dataset':
            # Now that dataset is loaded let us give the user the option to view the dataset
            if st.session_state.cluster_dataset is not None:
                if st.button("View Dataset"):
                    st.subheader('Dataset Preview')
                    st.dataframe(st.session_state.cluster_dataset.head(10), use_container_width=True)

                    st.write("### Data Types:")
                    st.write(st.session_state.cluster_dataset.dtypes)

                    st.session_state.cluster_step = 'select_columns'
                
        if st.session_state.cluster_step == 'select_columns':
            # let us give the user the option to choose the columns they want to choose for segmentation/clustering
            st.markdown("### Select Columns for Segmentation/Clustering")
            st.warning("Please ensure the columns you want to use for segmentation/clustering are numeric (i.e not categorical or text). The model will not work with non-numeric columns.")
            columns = st.multiselect("Select columns to use for segmentation/clustering",
                                        options=st.session_state.cluster_dataset.columns.tolist(),
                                        default=[], # change to empty
                                        key="cluster_columns_select")
            cat_cols_to_add_back = st.multiselect("Select categorical columns to add back to the dataset after segmentation/clustering (usually names or IDs)",
                                                    options=st.session_state.cluster_dataset.columns.tolist(),
                                                    default=[],
                                                    key="cluster_cats_select")
            
            if columns:
                st.session_state.cluster_cols = columns
                st.session_state.cat_cols_to_add_back = cat_cols_to_add_back
                st.success(f"Selected Numerical Columns: {', '.join(columns)}")
                st.success(f"Selected Categorical Columns to Add Back After Clustering: {', '.join(cat_cols_to_add_back)}")

                # Show dataset summary including total rows and columns, number of missing values for each column, and data types
                st.markdown("### Dataset Summary")
                st.write(f"Total Datapoints: {len(st.session_state.cluster_dataset)}")
                st.write("### Missing Values:")
                st.write(st.session_state.cluster_dataset.isnull().sum())
            
                # Show a warning if there are any non-numeric columns in the selected columns
                st.warning("Please ensure all selected columns for segmentation are numeric. Before clearing null values, ensure you are only deleting data due to missing important rows")
                
                if st.button("Proceed to Clear Null Values"):
                    st.session_state.cluster_step = 'clear_null_values'
                    st.rerun()
    
        if st.session_state.cluster_step == 'clear_null_values':
            # Show button to clear null values
            st.warning("Please ensure you are only deleting data due to missing important rows. This will clear all null values from the selected columns and retain categorical data.")
            st.write('Number of Datapoints with null values in the following columns:')
            st.write(st.session_state.cluster_dataset[st.session_state.cluster_cols].isnull().sum())

            if st.button("Clear Null Values"):
                st.session_state.numerical_data = st.session_state.cluster_dataset.copy()
                st.session_state.numerical_data = st.session_state.numerical_data[st.session_state.cluster_cols]
                st.session_state.numerical_data = st.session_state.numerical_data.dropna()

                # only remove data from columns in cluster_cols ONLY because then we want to keep only those categorical features after na filtering
                st.session_state.cat_data = st.session_state.cluster_dataset.copy()
                st.session_state.cat_data = st.session_state.cat_data[st.session_state.cluster_cols + st.session_state.cat_cols_to_add_back].dropna(subset=st.session_state.cluster_cols, how='any')
                st.session_state.cat_data = st.session_state.cat_data[st.session_state.cat_cols_to_add_back]

                st.success("Null values cleared and Categorical data retained successfully!")
                # Show how many rows and columns are left after clearing null values
                st.write(f"Total Rows after clearing null values: {len(st.session_state.numerical_data)}")

                # First give the user the summary statistics of the dataset
                st.markdown("### Final Cluster Dataset Summary Statistics") 
                st.write(st.session_state.numerical_data.describe())       

            if st.button("Proceed to Standardization"):
                st.session_state.cluster_step = 'proceed_to_standardization'
                st.rerun()

        if st.session_state.cluster_step == 'proceed_to_standardization':
            # Show a button to proceed to segmentation/clustering but show warning if there are any non-numeric columns
            
            if st.session_state.numerical_data.select_dtypes(include=['object', 'category']).shape[1] != 0:
                st.warning("Please ensure all selected columns are numeric.")
                if st.button("Select Columns Again"):
                    # Reset the session state to allow user to select columns again
                    st.session_state.cluster_step = 'select_columns'
                    st.session_state.numerical_data = None
                    st.session_state.cat_data = None
                    st.session_state.cat_cols_to_add_back = []
                    st.session_state.cluster_cols = None
                    st.rerun()
            else:
                st.success("Proceeding to segmentation/clustering...")

            # Now give user option to choose the type of sttandardization they want to apply
            st.markdown("### Choose Standardization Method")
            standardization_method = st.selectbox("Select Standardization Method",
                                                ["Min-Max Scaling", "Standard Scaling", "Robust Scaling"])
            # give quick description of each method
            if standardization_method == "Min-Max Scaling":
                st.write("Min-Max Scaling scales the data to a fixed range, usually [0, 1]. It is sensitive to outliers.")
            elif standardization_method == "Standard Scaling":
                st.write("Standard Scaling standardizes the data to have a mean of 0 and a standard deviation of 1. It is less sensitive to outliers.")
            elif standardization_method == "Robust Scaling":
                st.write("Robust Scaling scales the data using statistics that are robust to outliers, such as the median and interquartile range.")

            # Show button to apply standardization
            if st.button("Apply Standardization"):
                if standardization_method == "Min-Max Scaling":
                    from sklearn.preprocessing import MinMaxScaler
                    st.session_state.scaler = MinMaxScaler()
                elif standardization_method == "Standard Scaling":
                    from sklearn.preprocessing import StandardScaler
                    st.session_state.scaler = StandardScaler()
                elif standardization_method == "Robust Scaling":
                    from sklearn.preprocessing import RobustScaler
                    st.session_state.scaler = RobustScaler()

                st.session_state.scaled_data = st.session_state.scaler.fit_transform(st.session_state.numerical_data)
                st.success("Standardization applied successfully!")
            if st.button("Analyze Potential Clusters"):
                st.session_state.cluster_step = 'analyze_clusters'
                st.rerun()

        if st.session_state.cluster_step == 'analyze_clusters':
            
            # Now we will analyze potential clusters from 1 to 20 clusters
            if st.session_state.scaled_data is not None:
                st.markdown("### Analyze Potential Clusters")
                from sklearn.cluster import KMeans
                import matplotlib.pyplot as plt

                # Calculate inertia for each number of clusters
                inertia = []
                for n_clusters in range(1, 21):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    kmeans.fit(st.session_state.scaled_data)
                    inertia.append(kmeans.inertia_)

                # Plot the elbow curve
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(range(1, 21), inertia, marker='o')
                ax.set_title('Elbow Method for Optimal Clusters')
                ax.set_xlabel('Number of Clusters')
                ax.set_ylabel('Inertia')
                st.pyplot(fig)

                st.success("Elbow curve plotted successfully! Please choose the optimal number of clusters based on the elbow point.")

                # Now give user option to choose the number of clusters
                n_clusters = st.selectbox("Select Number of Clusters", options=range(1, 21), key="num_clusters_input")
                st.session_state.n_clusters = n_clusters
                # Confirm button
                if st.button("Confirm Number of Clusters"):
                    if 1 <= n_clusters <= 20:
                        st.session_state.cluster_confirmed = True
                        st.success(f"Number of clusters set to {n_clusters}")
                        st.rerun()  # Jump to next render instantly
                    else:
                        st.error("Please select a number of clusters between 1 and 20.")

                # Proceed button appears only if confirmed
                if st.session_state.cluster_confirmed:
                    if st.button("Proceed to Run Clustering"):
                        st.session_state.cluster_step = 'run_clustering'
                        st.rerun()

        if st.session_state.cluster_step == 'run_clustering':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=st.session_state.n_clusters, random_state=42)
            st.session_state.kmeans_fitted = kmeans.fit(st.session_state.scaled_data)
            st.session_state.numerical_data['Cluster'] = st.session_state.kmeans_fitted.labels_
            st.session_state.numerical_data[st.session_state.cat_cols_to_add_back] = (
                st.session_state.cat_data[st.session_state.cat_cols_to_add_back].values
            )
            st.success("Clustering completed successfully!")

            # Give option to view clustered data
            if st.button("View Clustered Data"):
                st.subheader('Clustered Data')
                st.dataframe(st.session_state.numerical_data, use_container_width=True)

                # Show summary of clusters
                st.markdown("### Cluster Summary")
                # Get numeric columns for averaging
                numeric_cols = st.session_state.numerical_data.select_dtypes(include='number').columns
                
                # Mean values per cluster
                mean_summary = (
                    st.session_state.numerical_data
                    .groupby('Cluster')[numeric_cols]
                    .mean()
                )
                
                # Count of rows per cluster
                count_summary = (
                    st.session_state.numerical_data
                    .groupby('Cluster')
                    .size()
                    .rename("Count")
                )
                
                # Combine them into one DataFrame
                cluster_summary = mean_summary.join(count_summary)
                st.dataframe(cluster_summary, use_container_width=True)

                # Show download button for clustered data
                csv = st.session_state.numerical_data.to_csv(index=False)
                st.download_button("Download Clustered Data as CSV", csv, "clustered_data.csv", "text/csv")

            if st.button("Proceed to Generate Cluster Summary"):
                st.session_state.cluster_step = 'generate_cluster_summary'
                st.rerun()

        if st.session_state.cluster_step == 'generate_cluster_summary':
            # Give option to give an Open AI text summary of the clusters in markdown format
            
            if not st.session_state.kmeans_fitted:
                st.error("Please run clustering first.")
            else:
                prompt_template = """
                Write a business report summarizing the following clusters:
                {clusters}

                Use the following Markdown format:
                # Insert Descriptive Report Title

                ## Cluster Summary
                Give each cluster a descriptive name based on its characteristics. Use 3 to 5 numbered bullet points.
                Please summarize the key characteristics of each cluster based on the pandas dataframes .describe() for each cluster.

                ## Strategic Implications
                Describe any strategic implications of the clusters. Use 3 to 5 numbered bullet points.

                ## Conclusions
                Conclude with any overarching business actions that the company is pursuing that may have positive or negative implications and what those implications are.
                """

                # Convert your DataFrames to a single string
                clusters_text = "\n\n".join(
                    f"Cluster {i}:\n{df.describe().to_string()}"
                    for i, df in enumerate(
                        st.session_state.numerical_data[st.session_state.numerical_data['Cluster'] == j]
                        for j in range(st.session_state.numerical_data['Cluster'].nunique())
                    )
                )

                # Build the LLMChain directly
                prompt = PromptTemplate.from_template(prompt_template)
                model = ChatOpenAI(
                    model=MODEL,
                    temperature=0,
                    api_key=st.secrets['OPENAI'],
                )
                llm_chain = LLMChain(llm=model, prompt=prompt)

                # Run GPT with your cluster data
                with st.spinner("Generating AI cluster summary..."):
                    response = llm_chain.invoke({"clusters": clusters_text})

                # Some versions of LangChain return 'text', others 'output_text'
                summary = (
                    response.get("text")
                    if isinstance(response, dict) and "text" in response
                    else response
                )

                # Display the result
                st.subheader("Open AI Summary of Clusters")
                st.markdown(summary)

                pdf_data = generate_pdf_from_markdown(summary)
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_data,
                    file_name="summary.pdf",
                    mime="application/pdf"
                )


                            

