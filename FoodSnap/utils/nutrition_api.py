import requests
import os

API_KEY = os.getenv("SPOONACULAR_API_KEY")

def get_nutrition(food_desc):
    if not API_KEY:
        return {"error": "Missing SPOONACULAR_API_KEY environment variable."}
    
    url = "https://api.spoonacular.com/recipes/guessNutrition"
    params = {"title": food_desc, "apiKey": API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}
