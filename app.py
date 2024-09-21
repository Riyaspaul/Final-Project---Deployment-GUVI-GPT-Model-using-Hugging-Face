import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model_name_or_path = "./fine_tuned_model"
#model_name_or_path = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the text generation function
def generate_text(model, tokenizer, seed_text, max_length=100, temperature=1.0, num_return_sequences=1):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
    generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    return generated_texts

# Streamlit app
st.title("Text Generation with GPT-2")
st.write("This app generates text using a fine-tuned GPT-2 model. Enter a prompt and the model will generate a continuation.")

seed_text = st.text_input("Enter your prompt:", "Google is known for")
max_length = st.slider("Max Length:", min_value=50, max_value=500, value=100)
temperature = st.slider("Temperature:", min_value=0.1, max_value=2.0, value=1.0)

if st.button("Generate"):
    with st.spinner("Generating text..."):
        generated_texts = generate_text(model, tokenizer, seed_text, max_length, temperature)
        for i, generated_text in enumerate(generated_texts):
            st.subheader(f"Generated Text {i + 1}")
            st.write(generated_text)
            st.warning("This GPT can make mistakes so for further reference please refer https://www.guvi.in")
