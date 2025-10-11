import gradio as gr
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Load model and processor
model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
processor = DonutProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Inference function
def answer_question(image: Image.Image, question: str):
    if image is None or question.strip() == "":
        return "⚠️ Please upload an image and enter a question."

    # Format prompt
    prompt = f"<s_docvqa><question>{question}</question><image>"

    try:
        image = image.convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=512,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )

        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio Interface
demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Image(type="pil", label="Upload Document Image"),
        gr.Textbox(label="Enter Your Question", placeholder="e.g., What is the invoice number?")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="🧾 Donut DocVQA - Ask Questions About Your Documents",
    description="Upload a document (like an invoice or form), type a question, and get an answer using the Donut DocVQA model.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
