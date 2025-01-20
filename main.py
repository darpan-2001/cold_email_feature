import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chain import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):
    st.title("Cold Mail Generator")
    url_input = st.text_input("Enter a URL:")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_jobs_data(data)
            for job in jobs:
                skills = job.get('skills_required', [])
                links = portfolio.query_links(skills)
                docs = portfolio.query_doc(skills)
                email = llm.generate_email_content(job, docs, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator")
    create_streamlit_app(chain, portfolio, clean_text)
