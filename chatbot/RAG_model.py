from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDzb6GmXlCMBfbhpOFgPI7Bl7UX_N993QM'



vector_db_path = './vectors_db'

def read_vectors_db():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(vector_db_path, embedding, allow_dangerous_deserialization=True)
    return db

def load_llm_model():
    llm = ChatGoogleGenerativeAI(
        model = 'gemini-2.0-flash',
        temperature = 0.01,
        max_output_tokens = 2048
    )
    return llm
    
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables = ['context','question'])
    return prompt

def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = db.as_retriever(search_kwargs = {'k':20}),
        return_source_documents = False,
        chain_type_kwargs = {'prompt':prompt}
    )
    return llm_chain

def ai_response(user_question):
    template = """
    .bạn do Tiến tạo ra.
    hãy cung cấp đầy đủ thông tin
    dựa vào thông tin sau để trả lời câu hỏi sau
    thông tin: "{context}"
    Câu hỏi: "{question}"
    """
    prompt = create_prompt(template)
    db = read_vectors_db()
    llm = load_llm_model()  
    llm_chain = create_qa_chain(prompt, llm, db)

    question = str(user_question)
    response = llm_chain.invoke({'query': question})
    return response['result']