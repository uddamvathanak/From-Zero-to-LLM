## Code Generation Tutorial with Gradio

# Install Required Libraries
# !python -m pip install gradio transformers torch

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Pre-trained Code Generation Model
model_name = "Salesforce/codegen-350M-mono"  # Example model for code generation
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded successfully.")

# Define Code Generation Function
def generate_code(prompt, max_length=100, temperature=0.7):
    print(f"Received prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    print("Tokenized input:", inputs)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated code:", generated_code)
    return generated_code

# Create Gradio Interface
def gradio_interface():
    interface = gr.Interface(
        fn=generate_code,
        inputs=[
            gr.Textbox(lines=5, placeholder="Enter your prompt for code generation here...", label="Code Prompt"),
            gr.Slider(minimum=50, maximum=512, step=10, value=100, label="Max Length"),
            gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperature")
        ],
        outputs=gr.Textbox(label="Generated Code"),
        title="Code Generation with Transformers",
        description="Provide a prompt to generate code using a pre-trained model. Adjust the max length and temperature for different outputs.",
    )
    return interface

# Launch Gradio App
if __name__ == "__main__":
    print("Launching Gradio app...")
    app = gradio_interface()
    app.launch()
