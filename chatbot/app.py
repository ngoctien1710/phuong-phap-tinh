from UI import start_ui, get_user_question, send_bot_answer, get_expression_and_epsilon
from bot import ai_response, add_newton_method
import threading
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
sys.path.append('/home/tien/my_project/phuong-phap-tinh/algorithm')  
from newton_method_final import NewtonSolver, format_output_string 

import os

os.environ['GOOGLE_API_KEY'] = 'AIzaSyB9lwxGn0BFdGk2L80-2sZyN3GOrfywuLE'

def load_llm_model():
    llm = ChatGoogleGenerativeAI(
        model = 'gemini-2.0-flash',
        temperature = 0,
        max_output_tokens = 2048
    )
    return llm
llm = load_llm_model()

def bot_loop():
    while True:
        q, check = get_user_question()
        if q:
            send_bot_answer(ai_response(q, check, llm))
        if get_expression_and_epsilon(False)[0] is not None and get_expression_and_epsilon(False)[1] is not None:
            c = format_output_string(NewtonSolver().solve(str(get_expression_and_epsilon(False)[0]), get_expression_and_epsilon(False)[1]))
            add_newton_method(c)
            if c != 'Lỗi: Không tìm thấy khoảng phân ly nào trong phạm vi Interval(-1000, 1000).':
                send_bot_answer(c)
            print(get_expression_and_epsilon(True))

if __name__ == "__main__":  

    threading.Thread(target=bot_loop, daemon=True).start()
    start_ui()
