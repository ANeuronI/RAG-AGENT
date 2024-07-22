from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
import os

# summeriser
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Initialize ChatGroq for summarization
summarizer_llm = ChatGroq(
    temperature=0.7,
    model="llama3-8b-8192",
    api_key=GROQ_API_KEY,
    streaming=True,
    verbose=True
)

# Define a prompt template for summarization
summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following content: {text}"
)

# Create the summarization chain
summarization_chain = LLMChain(
    llm=summarizer_llm,
    prompt=summarization_prompt
)

# Define the summarizer tool
def summarize_content_tool(text: str) -> str:
    return summarization_chain.run(text=text)

summarizer_tool = Tool(
    name="summarizer",
    description="Summarizes content using a language model.",
    func=summarize_content_tool
)


