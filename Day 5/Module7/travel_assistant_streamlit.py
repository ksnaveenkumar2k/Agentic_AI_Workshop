import os
import requests
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Intelligent Travel Assistant", page_icon="✈️", layout="wide"
)

# Initialize cache directory
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# Cache function for API responses
def get_cached_response(cache_key, expiry_hours=24):
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_data = json.load(f)

        # Check if cache is still valid
        cached_time = datetime.fromisoformat(cached_data["timestamp"])
        if datetime.now() - cached_time < timedelta(hours=expiry_hours):
            return cached_data["data"]

    return None


def save_to_cache(cache_key, data):
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

    with open(cache_file, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f)


# Initialize LLM (Gemini 1.5 Flash)
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)


# Custom Weather Tool with caching
@tool
def get_weather(destination: str) -> str:
    """Fetch the current weather forecast for a given destination."""
    # Check cache first
    cache_key = f"weather_{destination.lower().replace(' ', '_')}"
    cached_result = get_cached_response(
        cache_key, expiry_hours=1
    )  # Weather cache expires after 1 hour

    if cached_result:
        return cached_result

    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "Error: WEATHER_API_KEY not found in .env file"

    base_url = "http://api.weatherapi.com/v1/current.json"
    params = {"key": api_key, "q": destination}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        location = data["location"]["name"]
        region = data["location"]["region"]
        country = data["location"]["country"]
        temp_c = data["current"]["temp_c"]
        temp_f = data["current"]["temp_f"]
        condition = data["current"]["condition"]["text"]
        humidity = data["current"]["humidity"]
        wind_kph = data["current"]["wind_kph"]

        result = f"""Weather in {location}, {region}, {country}:
- Temperature: {temp_c}°C ({temp_f}°F)
- Condition: {condition}
- Humidity: {humidity}%
- Wind: {wind_kph} km/h"""

        # Cache the result
        save_to_cache(cache_key, result)

        return result
    except requests.RequestException as e:
        return f"Error fetching weather for {destination}: {str(e)}"


# Enhanced DuckDuckGo Search Tool with caching
class CachedDuckDuckGoSearchRun(DuckDuckGoSearchRun):
    def __call__(self, query: str) -> str:
        # Check cache first
        cache_key = f"search_{query.lower().replace(' ', '_')[:50]}"  # Limit key length
        cached_result = get_cached_response(
            cache_key, expiry_hours=6
        )  # Reduced cache time for more real-time results

        if cached_result:
            return cached_result

        # If not in cache, perform the search
        result = super().__call__(query)

        # Cache the result
        save_to_cache(cache_key, result)

        return result


# Initialize enhanced search tool
search_tool = CachedDuckDuckGoSearchRun()

# Define tools list
tools = [get_weather, search_tool]

# Define prompt for the agent with improved instructions for real-time content
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an intelligent Travel Assistant AI. Your task is to:
1. Fetch the current weather forecast for the user's destination using the get_weather tool.
2. Search for top tourist attractions in that destination using the search tool.
3. Search for the best time to visit the destination using the search tool.
4. Search for travel tips for the destination using the search tool.
5. Summarize the results in a structured, concise, and clear format.

Use the provided tools to gather information. Follow these steps:
1. First, use the get_weather tool to get current weather information.
2. Then, use the search tool with query "top tourist attractions in [destination]".
3. Next, use the search tool with query "best time to visit [destination] season climate".
4. Finally, use the search tool with query "[destination] travel tips local customs safety".

Format your response in a user-friendly way with clear sections:
* Current Weather
* Top Attractions (list 5-7 attractions with a brief description for each)
* Best Time to Visit (include seasonal information and climate details)
* Travel Tips (include local customs, safety advice, transportation tips)

IMPORTANT: For each section, provide ONLY information you found in the search results. 
DO NOT use placeholder text like "information not available" or "further research needed".
Instead, include whatever relevant information you found, even if limited.
If you genuinely found no information for a section, simply provide the most relevant information 
you did find that might help a traveler, and note that it's based on limited search results.

Be concise but informative. Focus on providing practical, real-time information that would be useful for a traveler.""",
        ),
        ("human", "Provide weather and top attractions for {destination}"),
        ("placeholder", "{agent_scratchpad}"),  # Required for tool-calling agent
    ]
)


# Create the tool-calling agent
@st.cache_resource
def get_agent():
    llm = get_llm()
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# Streamlit frontend
def main():
    st.title("✈️ Intelligent Travel Assistant")
    st.markdown(
        """
    Get real-time weather forecasts and discover top attractions, best times to visit, and travel tips for your next destination!
    
    This assistant uses:
    - WeatherAPI.com for real-time weather data
    - DuckDuckGo search for finding up-to-date travel information
    - Gemini 1.5 Flash AI for intelligent processing
    """
    )

    # Input field for destination with default value
    destination = st.text_input(
        "Where would you like to travel?", placeholder="e.g., Paris, Tokyo, New York"
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        # Button to trigger the assistant
        search_button = st.button(
            "🔍 Get Travel Info", type="primary", use_container_width=True
        )

    with col2:
        # Clear cache button
        if st.button("🔄 Clear Cache", use_container_width=True):
            for file in os.listdir(CACHE_DIR):
                os.remove(os.path.join(CACHE_DIR, file))
            st.success("Cache cleared successfully!")

    with col3:
        # Add a checkbox to force real-time data
        force_realtime = st.checkbox("Force real-time data (ignore cache)", value=False)

    # Process the request
    if search_button and destination:
        # If force real-time is checked, clear relevant cache entries
        if force_realtime:
            cache_patterns = [
                f"weather_{destination.lower().replace(' ', '_')}",
                f"search_{destination.lower().replace(' ', '_')}",
                f"search_top_tourist_attractions_in_{destination.lower().replace(' ', '_')[:30]}",
                f"search_best_time_to_visit_{destination.lower().replace(' ', '_')[:30]}",
                f"search_{destination.lower().replace(' ', '_')[:30]}_travel_tips",
            ]

            for file in os.listdir(CACHE_DIR):
                file_path = os.path.join(CACHE_DIR, file)
                file_name = os.path.splitext(file)[0]
                for pattern in cache_patterns:
                    if pattern in file_name:
                        os.remove(file_path)

            st.info("Using real-time data for this search.")

        with st.spinner("Researching your destination..."):
            try:
                agent_executor = get_agent()
                result = agent_executor.invoke({"destination": destination})

                # Display results in a nice format
                st.success(f"✅ Travel information for {destination}")

                # Create tabs for different sections
                weather_tab, attractions_tab, best_time_tab, tips_tab = st.tabs(
                    ["Weather", "Attractions", "Best Time to Visit", "Travel Tips"]
                )

                # Parse the output to extract different sections
                output = result["output"]

                with weather_tab:
                    if "Weather" in output:
                        weather_section = output.split("Top Attractions")[0]
                        st.markdown(f"### Current Weather\n{weather_section}")
                    else:
                        st.markdown(f"### Weather Information\n{output}")

                with attractions_tab:
                    if "Top Attractions" in output:
                        attractions_section = output.split("Top Attractions")[1]
                        if "Best Time to Visit" in attractions_section:
                            attractions_section = attractions_section.split(
                                "Best Time to Visit"
                            )[0]
                        elif "Travel Tips" in attractions_section:
                            attractions_section = attractions_section.split(
                                "Travel Tips"
                            )[0]
                        st.markdown(f"### Top Attractions\n{attractions_section}")
                    else:
                        st.markdown(
                            "No specific attractions information found in the search results."
                        )

                with best_time_tab:
                    if "Best Time to Visit" in output:
                        best_time = output.split("Best Time to Visit")[1]
                        if "Travel Tips" in best_time:
                            best_time = best_time.split("Travel Tips")[0]
                        st.markdown(f"### Best Time to Visit\n{best_time}")
                    else:
                        st.markdown(
                            "No specific 'best time to visit' information found in the search results."
                        )

                with tips_tab:
                    if "Travel Tips" in output:
                        travel_tips = output.split("Travel Tips")[1]
                        st.markdown(f"### Travel Tips\n{travel_tips}")
                    else:
                        st.markdown(
                            "No specific travel tips found in the search results."
                        )

                # Show the full response in an expandable section
                with st.expander("Show full AI response"):
                    st.markdown(output)

                # Show timestamp of when this information was retrieved
                st.caption(
                    f"Information retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info(
                    "Please check your API keys and internet connection, then try again."
                )
    elif search_button:
        st.warning("Please enter a destination.")


if __name__ == "__main__":
    main()
