import gradio as gr
from transformers import pipeline

# Load model
generator = pipeline("text-generation", model="gpt2")

def ai_agent(prompt):
    output = generator(prompt, max_length=100, num_return_sequences=1)
    return output[0]['generated_text']

iface = gr.Interface(fn=ai_agent, 
                     inputs=gr.Textbox(lines=2, placeholder="Ask me anything..."), 
                     outputs="text",
                     title="🧠 Simple AI Agent",
                     description="Built on Hugging Face with GPT-2")
iface.launch()
