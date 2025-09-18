import gradio as gr
from PIL import Image
from models.vision_captioner import generate_caption
from utils.nutrition_api import get_nutrition
from utils.recipe_generator import generate_recipes

def foodsnap_workflow(image):
    if image is None:
        return "Upload an image to get started.", {}, []
    
    # Generate caption
    caption = generate_caption(image)
    
    # Get nutrition info
    nutrition = get_nutrition(caption)
    
    # Generate recipes
    recipes = generate_recipes(caption, n=2)
    
    # Format nutrition for display
    if "error" in nutrition:
        nutrition_text = nutrition["error"]
    else:
        nutrition_text = "\n".join([f"{k}: {v}" for k, v in nutrition.items()])
    
    return caption, nutrition_text, recipes

title = "🍽️ FoodSnap – Snap a Meal, Get Nutrition & Recipes"
description = "Upload a photo of your food, get AI-powered description, nutrition info, and recipes."

iface = gr.Interface(
    fn=foodsnap_workflow,
    inputs=gr.Image(type="pil", label="Upload Food Image"),
    outputs=[
        gr.Textbox(label="📝 Food Description"),
        gr.Textbox(label="🧮 Nutrition Estimate"),
        gr.Textbox(label="📖 Recipes", lines=6)
    ],
    title=title,
    description=description,
    allow_flagging="never",
    theme="compact"
)

if __name__ == "__main__":
    iface.launch()
