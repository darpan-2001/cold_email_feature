import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    def extract_jobs_data(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            **SCRAPPED DATA FROM WEBSITE: **
            {page_data}
            **SCRAPPED DATA ENDS HERE**
            ** INSTRUCTIONS : ** The scrapped data is from a company's career website, your task is to extract the job postings data and return that into a JSON object containing the following keys: 'role', 'skills_required', 'experience', 'qualification'.
            Only return the valid JSON object.
            **NO PREAMBLE, ONLY VALID JSON OBJECT**
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Unable to parse data.")
        return res if isinstance(res, list) else [res]

    def generate_email_content(self, job, doc,links):
        prompt_email = PromptTemplate.from_template(
            """
                ### JOB DESCRIPTION:
                {job_description}

                ### INSTRUCTION:
                You are a Software Engineer, working in the AI and Data Science domain. You are now looking for a new job.
                With your skills and experience, you have empowered your current company's business by building and testing deep learning models, which has heightened overall efficiency of AI models being pushed to production.
                Your job is to write a cold email to the hiring manager, explaining how your skillset and experience can be helpful for their organization.
                Also, include the most relevant portfolio link of yours with the document name in the following format:

                Take a look at my work here:
                {doc_name}: {link_list}

                Ensure that each link is listed only once, the name of the portfolio document is provided, and it is presented in a clean, readable format.
                Do not repeat any link, even if you are referencing them multiple times within the email and also include the name of documnet only before prinitng it's link.
                Remember you are a software engineer working in AI domain.
                Do not provide a preamble.

                ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job),"doc_name": doc, "link_list": links})
        return res.content