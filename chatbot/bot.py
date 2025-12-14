from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


vector_db_path = './vectors_db'
# Lưu chat history (system + human + ai)
chat_history = [
    ('system', 'bạn là chuyên gia giải phương trình bằng phương pháp tiếp tuyến Newton.'
               'Tên bạn là botchat.')
]


def read_vectors_db():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(vector_db_path, embedding, allow_dangerous_deserialization=True)
    return db
    
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

def format_chat_history(chat_history):
    formatted = ""
    for role, msg in chat_history:
        formatted += f"{role.upper()}: {msg}\n"
    return formatted

def add_newton_method(newton_method):
    global chat_history
    chat_history.append(('ai', newton_method))

def ai_response(user_question, check, llm):
    global chat_history
    if check:
        chat_history = [
            ('system', 'bạn là chuyên gia giải phương trình bằng phương pháp tiếp tuyến Newton.'
               'Tên bạn là botchat.')
        ]
    hisque = format_chat_history(chat_history)
    hisque += 'QUESTION: ' + str(user_question)
    
    template = """
    dựa vào thông tin sau để trả lời câu hỏi sau, trả lời chính xác
    ưu tiên kiểm tra thông tin trước nếu không có thông tin thì trả lời theo ý của bạn.
    lưu ý, trong câu hỏi có cả lịch sử đoạn chat, câu hỏi mới là câu có đầu dòng là 'QUESTION'.
    thông tin: "{context}"
    lịch sử  và câu hỏi: "{question}"
    """

    prompt = create_prompt(template)
    db = read_vectors_db()

    llm_chain = create_qa_chain(prompt, llm, db)

    # Thêm câu hỏi mới vào history
    chat_history.append(('human', str(user_question)))

    response = llm_chain.invoke({'query': hisque})
    # Gọi LLM với toàn bộ history
    #ai_msg = llm_chain.invoke(chat_history)

    # Lưu trả lời của bot vào history
    chat_history.append(('ai', response['result']))
    print(hisque)
    return response['result']
