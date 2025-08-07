import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Streamlit app UI
st.title("🤖 Local Chatbot without OpenAI API")
st.write("Powered by DialoGPT from Hugging Face")

# Maintain chat history
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
    st.session_state.past_inputs = []
    st.session_state.generated_responses = []

# User input
user_input = st.text_input("You:", key="input")

if user_input:
    # Encode user input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append to chat history
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids

    # Generate a response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Save history
    st.session_state.chat_history_ids = chat_history_ids
    st.session_state.past_inputs.append(user_input)
    st.session_state.generated_responses.append(response)

# Show conversation
for user_msg, bot_msg in zip(st.session_state.past_inputs, st.session_state.generated_responses):
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Bot:** {bot_msg}")
