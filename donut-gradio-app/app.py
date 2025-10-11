import gradio as gr
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
import json
import re

model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"
processor = DonutProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def safe_parse_json(text):
    text = text.strip()
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return json.loads(text)

def extract_document_json(image: Image.Image):
    if image is None:
        return json.dumps({"error": "Please upload a document image."}, indent=2)

    prompt = "<s_cord-v2>"

    try:
        image = image.convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=1024,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        try:
            parsed_json = safe_parse_json(decoded)
            return json.dumps(parsed_json, indent=2)
        except Exception as e:
            return json.dumps({
                "error": "Failed to parse model output as JSON.",
                "raw_output": decoded,
                "exception": str(e)
            }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

demo = gr.Interface(
    fn=extract_document_json,
    inputs=gr.Image(type="pil", label="Upload Document Image"),
    outputs=gr.Code(label="Extracted JSON", language="json"),
    title="📄 Donut Document Extractor (CORD-v2)",
    description="Upload a receipt or document and get structured data as formatted JSON using Donut.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
