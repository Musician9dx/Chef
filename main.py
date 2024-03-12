import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import Cohere
from langchain.embeddings import CohereEmbeddings
import os
import PyPDF2
from tqdm import tqdm

os.environ["cohere_api_key"]="---"
filepath="book"
def convert_pdf_to_string(file_path):

  with open(file_path, 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)

    text = ""
    for page_num in tqdm(range(num_pages)):
      page = pdf_reader.pages[page_num]
      text += page.extract_text()  # Accumulate text from each page

  return text



text=convert_pdf_to_string(file_path=filepath)


llm=Cohere(temperature=0.6)

text_splitter=CharacterTextSplitter(

    separator="\n",
    chunk_size=3500,
    chunk_overlap=1000,
    length_function=len

)

texts=text_splitter.split_text(text)

embedding=CohereEmbeddings()

vectorStore=FAISS.from_texts(texts,embedding)

st.header("Chef")
st.subheader("Musician 9DX")

text_input=st.text_input("Enter the Recipe")
btn=st.button("Search")
ragChain=load_qa_chain(llm=llm)

if btn:

    st.info("Entered And Running")
    doc=vectorStore.similarity_search(text_input)
    st.info("Got the Vector")
    answer=ragChain.run(input_documents=doc,question=text_input)
    st.write(answer)



