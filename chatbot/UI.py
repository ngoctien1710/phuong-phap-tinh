import gradio as gr
import sympy as sp
import re

# ======================
# Biến trạng thái chat & biểu thức
# ======================
_chat_history = []
_user_question = None
_expression_result = None
_epsilon_result = None


def get_user_question():
    global _user_question
    q = _user_question
    _user_question = None
    return q, check_clear_chat_history()


def send_bot_answer(answer: str):
    global _chat_history
    _chat_history.append({"role": "assistant", "content": answer})
    return _chat_history


def on_user_submit(message, chat_state):
    global _user_question, _chat_history
    _user_question = message
    _chat_history.append({"role": "user", "content": message})
    return "", _chat_history


def refresh_chat():
    global _chat_history
    return _chat_history

def preprocess_math_expr(expr_str):
    expr_str = expr_str.replace("^", "**")
    expr_str = re.sub(r'\bln\((.*?)\)', r'log(\1)', expr_str)
    expr_str = re.sub(r'\blg\((.*?)\)', r'log(\1,10)', expr_str)
    expr_str = re.sub(r'\blog2\((.*?)\)', r'log(\1,2)', expr_str)
    expr_str = re.sub(r'\be\^(\([^\)]+\)|[a-zA-Z0-9_]+)', r'exp(\1)', expr_str)
    return expr_str

def validate_expression(expr_str):
    """Kiểm tra hợp lệ, bật/tắt nút 'Tiếp tục'"""
    if not expr_str.strip():
        return gr.update(interactive=False)
    try:
        expr_str = preprocess_math_expr(expr_str)
        expr = sp.sympify(expr_str)

        variables = list(expr.free_symbols)
        if len(variables) != 1:
            return gr.update(interactive=False)

        if "=" in expr_str or ";" in expr_str or "," in expr_str:
            return gr.update(interactive=False)

        return gr.update(interactive=True)
    except Exception:
        return gr.update(interactive=False)


def validate_epsilon(eps_str):
    """Kiểm tra epsilon hợp lệ, bật/tắt nút 'Hoàn tất'"""
    if not eps_str.strip():
        return gr.update(interactive=False)
    eps_str = eps_str.strip().replace(',', '.')
    try:
        float(eps_str)
        return gr.update(interactive=True)
    except ValueError:
        return gr.update(interactive=False)


def parse_expression(expr_str):
    global _expression_result
    try:
        expr_str = preprocess_math_expr(expr_str)
        expr = sp.sympify(expr_str)
        _expression_result = expr
        return expr
    except Exception as e:
        _expression_result = None
        return f"Lỗi: {e}"


# ======================
# UI logic
# ======================
def show_expr_input():
    return gr.update(visible=True, value=""), gr.update(visible=True, interactive=False), gr.update(visible=False)


def show_epsilon_input(expr_str):
    parse_expression(expr_str)
    return (
        gr.update(visible=False, value=""),  # expr_input
        gr.update(visible=False),            # expr_next_btn
        gr.update(visible=True, value=""),   # eps_input
        gr.update(visible=True, interactive=False)  # eps_done_btn
    )


def store_epsilon(eps_str):
    """Lưu epsilon (hợp lệ mới lưu)"""
    global _epsilon_result
    if eps_str:
        eps_str = eps_str.strip().replace(',', '.')
        try:
            _epsilon_result = float(eps_str)
        except ValueError:
            _epsilon_result = None
    else:
        _epsilon_result = None
    return (
        gr.update(visible=False, value=""),
        gr.update(visible=False),
        gr.update(visible=True)
    )


# ======================
# Getter
# ======================
def get_expression_and_epsilon(reset=False):
    global _expression_result, _epsilon_result
    expr, eps = _expression_result, _epsilon_result
    if reset:
        _expression_result, _epsilon_result = None, None
    return expr, eps

def on_clear_chat():
    """Xử lý khi người dùng bấm biểu tượng sọt rác"""
    global _chat_history, _user_question, _expression_result, _epsilon_result
    _chat_history = []
    _user_question = None
    _expression_result = None
    _epsilon_result = None
    return _chat_history

def check_clear_chat_history():
    global _chat_history
    return _chat_history is None

# ======================
# Giao diện
# ======================
def start_ui():
    css = """
    .gradio-container {
        max-width: 1500px;
        margin: auto;
    }
    .chatbot {
        width: 1500px !important;
        height: 700px !important;
        overflow-y: auto;
    }
    """

    with gr.Blocks(css=css) as demo:
        gr.Markdown("## 💬 Make Newton live again")

        # Chatbot
        with gr.Row():
            chatbot = gr.Chatbot(label="Cuộc hội thoại", type="messages", elem_classes=["chatbot"])
            chatbot.clear(on_clear_chat, None, chatbot)


        # Input chat
        with gr.Row():
            user_input = gr.Textbox(placeholder="Nhập tin nhắn...", show_label=False, scale=4)
            send_btn = gr.Button("Gửi", scale=1)

        chat_state = gr.State([])
        send_btn.click(on_user_submit, [user_input, chat_state], [user_input, chatbot])
        user_input.submit(on_user_submit, [user_input, chat_state], [user_input, chatbot])

        # Input biểu thức + epsilon
        with gr.Row():
            expr_button = gr.Button("Nhập biểu thức")
            expr_input = gr.Textbox(
                placeholder="Nhập biểu thức toán học...",
                value="", visible=False, show_label=False
            )
            expr_next_btn = gr.Button("Tiếp tục", visible=False, interactive=False)

            eps_input = gr.Textbox(
                placeholder="Nhập sai số (Tối đa 1e-7)...",
                value="", visible=False, show_label=False
            )
            eps_done_btn = gr.Button("Hoàn tất", visible=False, interactive=False)

        # --- Sự kiện ---
        expr_button.click(fn=show_expr_input,
                          inputs=[], outputs=[expr_input, expr_next_btn, expr_button])

        # kiểm tra hợp lệ liên tục khi nhập biểu thức
        expr_input.change(validate_expression, inputs=[expr_input], outputs=[expr_next_btn])

        expr_next_btn.click(fn=show_epsilon_input,
                            inputs=[expr_input],
                            outputs=[expr_input, expr_next_btn, eps_input, eps_done_btn])

        # kiểm tra hợp lệ liên tục khi nhập epsilon
        eps_input.change(validate_epsilon, inputs=[eps_input], outputs=[eps_done_btn])

        eps_done_btn.click(fn=store_epsilon,
                           inputs=[eps_input],
                           outputs=[eps_input, eps_done_btn, expr_button])

        # Refresh chatbot mỗi 0.5s
        timer = gr.Timer(0.5)
        timer.tick(refresh_chat, None, chatbot)

    demo.launch(share = True)

