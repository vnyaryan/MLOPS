import json
import os
import logging

# ----------------------------------------------------------------------------
# PATHS (edit as needed)
# ----------------------------------------------------------------------------
BASE_PATH = r"C:\Users\ARYAVIN\Documents\GitHub\AI\ML\MLOPS-7\BITNET"
tokenizer_config_path = os.path.join(BASE_PATH, "tokenizer_config.json")
tokenizer_json_path = os.path.join(BASE_PATH, "tokenizer.json")
message_txt_path = os.path.join(BASE_PATH, "message.txt")
output_txt_path = os.path.join(BASE_PATH, "final_message.txt")
log_path = os.path.join(BASE_PATH, "logs", "prompt_validator.log")

# ----------------------------------------------------------------------------
# LOGGING SETUP
# ----------------------------------------------------------------------------
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_path,
    filemode="w"
)

# ----------------------------------------------------------------------------
# LOAD JSON
# ----------------------------------------------------------------------------
def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load {path}: {e}")
        raise

# ----------------------------------------------------------------------------
# BUILD RULESET FROM BOTH JSON FILES
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# VALIDATE & FORMAT PROMPT
# ----------------------------------------------------------------------------
def validate_prompt(messages, ruleset):
    validated_lines = []
    invalid_lines = []
    last_role = None

    # Add BOS token if required
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

        # Append EOS token
        if ruleset["must_end_messages_with_eos"]:
            content += ruleset["eos_token"]

        validated_lines.append(f"{role}: {content}")
        last_role = role

    # If last message is from User, add generation cue
    if last_role == "User":
        validated_lines.append("Assistant:")

    return validated_lines, invalid_lines

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        tokenizer_config = load_json(tokenizer_config_path)
        tokenizer_data = load_json(tokenizer_json_path)
        with open(message_txt_path, 'r', encoding='utf-8') as f:
            messages = [line.strip() for line in f if line.strip()]

        ruleset = build_ruleset(tokenizer_config, tokenizer_data)
        valid_lines, invalid_lines = validate_prompt(messages, ruleset)

        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(valid_lines))

        print(f"✅ Prompt validated and saved to: {output_txt_path}")
        if invalid_lines:
            print(f"⚠️ Found {len(invalid_lines)} invalid lines. Check the log at: {log_path}")

    except Exception as e:
        print(f"❌ Script failed: {e}")
        logging.error(f"Script failed: {e}")
