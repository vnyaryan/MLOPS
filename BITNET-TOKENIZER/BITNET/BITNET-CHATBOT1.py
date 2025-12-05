import gradio as gr
import requests
import logging
import os
import time
import json

# ----------------------------------------------------------------------------
# PATH SETUP
# ----------------------------------------------------------------------------
BASE_PATH = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ML\MLOPS-7\BITNET"
tokenizer_config_path = os.path.join(BASE_PATH, "tokenizer_config.json")
tokenizer_json_path = os.path.join(BASE_PATH, "tokenizer.json")
LOG_PATH = os.path.join(BASE_PATH, "logs", "bitnet_chatbot_validated.log")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ----------------------------------------------------------------------------
# LOGGING SETUP
# ----------------------------------------------------------------------------
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.debug("Gradio-based BitNet chatbot with validation started.")

# ----------------------------------------------------------------------------
# API SETUP
# ----------------------------------------------------------------------------
API_URL = "http://127.0.0.1:9090/completion"

# ----------------------------------------------------------------------------
# GLOBAL CHAT HISTORY
# ----------------------------------------------------------------------------
chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

# ----------------------------------------------------------------------------
# PROMPT VALIDATION LOGIC
# ----------------------------------------------------------------------------
def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load {path}: {e}")
        raise

def build_ruleset(config, tokenizer):
    bos = config.get("bos_token", "")
    eos = config.get("eos_token", "")
    chat_template = config.get("chat_template", None)
    tokenizer_class = config.get("tokenizer_class", "")

    special_token_map = {
        t["content"]: t["id"]
        for t in tokenizer.get("added_tokens", [])
        if t.get("special", False)
    }

    return {
        "bos_token": bos,
        "eos_token": eos,
        "chat_template": chat_template,
        "tokenizer_class": tokenizer_class,
        "valid_roles": ["User", "Assistant"],
        "must_start_with_bos": bool(bos),
        "must_end_messages_with_eos": bool(eos),
        "special_token_map": special_token_map,
        "enforce_spacing": {
            "lstrip": any(t.get("lstrip", False) for t in tokenizer.get("added_tokens", [])),
            "rstrip": any(t.get("rstrip", False) for t in tokenizer.get("added_tokens", [])),
        },
        "requires_normalization": any(t.get("normalized", False) for t in tokenizer.get("added_tokens", []))
    }

def validate_prompt(messages, ruleset):
    validated_lines = []
    invalid_lines = []
    last_role = None

    if ruleset["must_start_with_bos"]:
        validated_lines.append(ruleset["bos_token"])

    for line in messages:
        if not any(line.startswith(f"{role}:") for role in ruleset["valid_roles"]):
            logging.warning(f"Missing valid role: {line}")
            invalid_lines.append(line)
            continue

        try:
            role, content = line.split(":", 1)
            role = role.strip()
            content = content.strip()
        except ValueError:
            logging.warning(f"Improper format: {line}")
            invalid_lines.append(line)
            continue

        if not content:
            logging.warning(f"Empty message: {line}")
            invalid_lines.append(line)
            continue

        if ruleset["must_end_messages_with_eos"]:
            content += ruleset["eos_token"]

        validated_lines.append(f"{role}: {content}")
        last_role = role

    if last_role == "User":
        validated_lines.append("Assistant:")

    return validated_lines, invalid_lines

# ----------------------------------------------------------------------------
# CHAT HANDLER
# ----------------------------------------------------------------------------
def bitnet_chat(user_input, history):
    logging.debug(f"User input: {user_input}")
    chat_history.append({"role": "user", "content": user_input})

    # Prepare raw message list
    raw_lines = []
    for turn in chat_history:
        if turn["role"] == "user":
            raw_lines.append(f"User: {turn['content']}")
        elif turn["role"] == "assistant":
            raw_lines.append(f"Assistant: {turn['content']}")

    # Validate prompt
    try:
        tokenizer_config = load_json(tokenizer_config_path)
        tokenizer_data = load_json(tokenizer_json_path)
        ruleset = build_ruleset(tokenizer_config, tokenizer_data)

        valid_lines, invalid_lines = validate_prompt(raw_lines, ruleset)
        formatted_prompt = "\n".join(valid_lines)

        if invalid_lines:
            logging.warning(f"Invalid lines detected during validation: {invalid_lines}")

        logging.debug(f"Validated and formatted prompt:\n{formatted_prompt}")

    except Exception as e:
        error_msg = f"Prompt validation failed: {str(e)}"
        logging.error(error_msg)
        history.append((user_input, error_msg))
        return history, history

    # Call API
    try:
        start_time = time.time()
        response = requests.post(API_URL, json={
            "prompt": formatted_prompt,
            "n_predict": 200,
            "stop": ["<|eot_id|>"]
        }, timeout=300)
        response.raise_for_status()
        end_time = time.time()

        duration = round(end_time - start_time, 2)
        answer = response.json()["content"].strip()
        chat_history.append({"role": "assistant", "content": answer})

        formatted_response = f"{answer}\n\n(Response time: {duration} seconds)"
        history.append((user_input, formatted_response))
        return history, history

    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        logging.error(error_msg)
        history.append((user_input, error_msg))
        return history, history

# ----------------------------------------------------------------------------
# GRADIO UI
# ----------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 BitNet Chatbot (with Prompt Validation)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message and press Enter")
    state = gr.State([])

    def user_submit(user_input, history):
        return bitnet_chat(user_input, history)

    msg.submit(user_submit, [msg, state], [chatbot, state])
    msg.submit(lambda: "", None, msg)  # Clear input box

# ----------------------------------------------------------------------------
# RUN APP
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()
