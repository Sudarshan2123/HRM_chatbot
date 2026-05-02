import os
import requests
from langchain_core.tools import tool

import logging

@tool("Current_Date_weather")
def get_weather(cityname: str) -> str:
    """Useful to get the current weather for a specific city. 
    The input should be a city name (e.g., 'Mumbai')."""
    
    API_KEY = os.getenv("API_key_Weather")
    if not API_KEY:
        return "Error: API Key not found in environment variables."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={cityname}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Check for HTTP errors
        data = response.json()
        
        # Check if the API returned an error message in the JSON
        if data.get("cod") != 200:
            return f"Error: {data.get('message', 'City not found')}"

        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        return f"The current temperature in {cityname} is {temp}°C with {desc}."
    
    except Exception as e:
        return f"System error: {str(e)}"


@tool("Get_Top_News")
def get_top_news(category: str = "general") -> str:
    """Essential for getting current news. Use 'general' for top stories, 
    or specific topics like 'business', 'science', 'sports', or 'technology'."""
    
    API_KEY = os.getenv("NEWS_API_KEY")
    
    # 1. First, try the 'Top Headlines' endpoint (fastest)
    top_url = f"https://newsapi.org/v2/top-headlines?country=in&category={category}&apiKey={API_KEY}"
    
    try:
        response = requests.get(top_url).json()
        articles = response.get("articles", [])

        # 2. If empty, try the 'Everything' endpoint searching for that keyword in India
        if not articles:
            # We search for the category name as a keyword for better results
            search_url = f"https://newsapi.org/v2/everything?q={category}+India&sortBy=publishedAt&language=en&apiKey={API_KEY}"
            response = requests.get(search_url).json()
            articles = response.get("articles", [])

        if not articles:
            return f"I couldn't find any recent {category} news. Perhaps try searching for a more general topic?"

        # 3. Format the top 3
        news_list = []
        for a in articles[:5]:
            title = a.get('title', 'No Title')
            source = a.get('source', {}).get('name', 'Unknown')
            news_list.append(f"- {title} ({source})")
            
        return f"Latest {category} updates for India:\n" + "\n".join(news_list)

    except Exception as e:
        return f"Error connecting to news service: {str(e)}"

tools = [get_weather,get_top_news]