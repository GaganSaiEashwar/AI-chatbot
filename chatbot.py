from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

chat_history_ids = None
print("🤖 Chatbot: Hello! Type 'exit' to stop.")

for step in range(5):
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"🤖 Chatbot: {response}")
