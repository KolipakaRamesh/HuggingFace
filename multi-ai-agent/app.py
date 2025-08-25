import gradio as gr
from transformers import pipeline

# Load two models for different agents
qa_agent = pipeline("text2text-generation", model="google/flan-t5-base")
creative_agent = pipeline("text-generation", model="gpt2")

def multi_agent_system(user_input):
    # Agent 1: QA style answer
    qa_response = qa_agent(user_input, max_length=100)[0]['generated_text']
    
    # Agent 2: Creative text generation
    creative_response = creative_agent(user_input, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    return qa_response, creative_response

# Build Gradio UI
iface = gr.Interface(
    fn=multi_agent_system,
    inputs=gr.Textbox(lines=2, placeholder="Ask something..."),
    outputs=[gr.Textbox(label="Agent 1 (QA)"), gr.Textbox(label="Agent 2 (Creative Writer)")],
    title="🤖 Multi-Agent AI System",
    description="Ask a question and get answers from two different AI agents!"
)

iface.launch()
