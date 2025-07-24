# Data Explorer Packages
import pandas as pd
import streamlit as st
import wbdata
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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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

# Initialize session state variables
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
@st.cache_data
def fetch_wb(indicator_code: str, countries: list) -> pd.Series:
    """Fetch data from WB using wbdata library for specified countries."""
    try:
        data_series = wbdata.get_series(indicator_code, country=countries)
        data_series = data_series.reset_index()
        return data_series
    except Exception as e:
        st.error(f"Error fetching data for {indicator_code}: {e}")
        return pd.Series()

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

################################################   STREAMLIT APP   ###################################################
if not st.session_state.tool:
    st.title('Welcome to the Medikana Research Tool!')
    st.subheader('Please choose a tool to use:')
    tool_options = ['Select a tool...', 'Data Explorer', 'Text Summarizer']
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

        def generate_pdf_with_reportlab(text: str) -> bytes:
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
        
            y_position = height - 50
            for line in text.split("\n"):
                c.drawString(50, y_position, line)
                y_position -= 15
                if y_position < 50:  # Start new page
                    c.showPage()
                    y_position = height - 50
            c.save()
            buffer.seek(0)
            return buffer.getvalue()
        
        # def sanitize_text(text):
        #     # Normalize Unicode characters to simple ASCII where possible
        #     text = unicodedata.normalize('NFKD', text)
        #     text = text.encode('ascii', 'ignore').decode('utf-8')  # Remove non-ASCII chars
        #     text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-")
        #     return text
        
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

                    # summary = sanitize_text(summary)
                    pdf_bytes = generate_pdf_with_reportlab(summary)
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_bytes,
                        file_name="summary.pdf",
                        mime="application/pdf"
                    )

        else:
            with col2:
                st.write("No file uploaded. Please upload a PDF file to proceed.")
