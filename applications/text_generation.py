## Conversational Agent using Self-Hosted LLaMA Model

# Install Required Libraries
# !pip install gradio transformers torch
# !python -m pip install protobuf 

import gradio as gr
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# Load Pre-trained LLM Model Locally
model_name = "openlm-research/open_llama_3b"
print("Loading model and tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda:0")
model = torch.compile(model)  # Compile for faster execution
print("Model loaded successfully.")

# Ensure the tokenizer has a padding token
tokenizer.pad_token = tokenizer.eos_token

def chat_with_agent(user_input, history=[]):
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, pad_token=tokenizer.pad_token).to(model.device)
    
    with torch.inference_mode():  # Disable gradient computation
        chat_history_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=32,  # Reduce for efficiency
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1,
            num_beams=1,  # Reduce for speed
            do_sample=True,
            temperature=0.7  # Sampling-based decoding
        )
    
    response = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
    return response

# Create Gradio ChatGPT-style Interface
def gradio_interface():
    return gr.ChatInterface(
        fn=chat_with_agent,
        title="AI Chatbot (Self-Hosted LLM)",
        description="A conversational AI powered running locally on your machine.",
    )

# Launch Gradio App
if __name__ == "__main__":
    print("Launching Gradio chat agent with self-hosted ...")
    app = gradio_interface()
    app.launch()
