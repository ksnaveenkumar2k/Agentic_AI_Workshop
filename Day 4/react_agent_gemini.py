# import os
# import json
# from google.generativeai import configure, GenerativeModel
# from tavily import TavilyClient
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# # Configure Gemini API
# configure(api_key=GEMINI_API_KEY)

# class ReActAgent:
#     def __init__(self, topic):
#         """Initialize the agent with a topic."""
#         self.topic = topic
#         self.questions = []
#         self.results = {}
#         self.gemini_model = GenerativeModel("gemini-1.5-flash")  # Free-tier model (adjust if needed)
#         self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

#     def generate_research_questions(self):
#         """Use Gemini API to generate 5-6 research questions."""
#         prompt = f"""
#         You are a research assistant. Given the topic '{self.topic}', generate a list of 5-6 well-structured research questions that cover different aspects of the topic. Return the questions in a JSON list.
#         Example:
#         ```json
#         [
#             "What are the main causes of {self.topic}?",
#             "How has {self.topic} evolved over time?"
#         ]
#         """
#         try:
#             response = self.gemini_model.generate_content(prompt)
#             questions_json = response.text.strip()
#             # Ensure the response is valid JSON
#             if questions_json.startswith("```json") and questions_json.endswith("```"):
#                 questions_json = questions_json[7:-3].strip()
#             self.questions = json.loads(questions_json)
#             return self.questions
#         except Exception as e:
#             print(f"Error generating questions: {e}")
#             return []

#     def search_web(self, question):
#         """Search the web for a given question using Tavily API."""
#         try:
#             response = self.tavily_client.search(
#                 query=question,
#                 search_depth="basic",
#                 max_results=5
#             )
#             # Extract title and content from search results
#             results = [
#                 {"title": result["title"], "content": result["content"]}
#                 for result in response["results"]
#             ]
#             return results
#         except Exception as e:
#             print(f"Error searching for '{question}': {e}")
#             return []

#     def gather_information(self):
#         """Search the web for answers to all research questions."""
#         for question in self.questions:
#             print(f"Searching for: {question}")
#             self.results[question] = self.search_web(question)

#     def generate_report(self):
#         """Compile a structured report based on collected information."""
#         report = f"# Research Report on {self.topic}\n\n"
#         report += "## Introduction\n"
#         report += f"This report explores the topic of {self.topic} by addressing key research questions generated through advanced reasoning. The findings are based on web searches conducted to gather relevant and recent information.\n\n"

#         for question in self.questions:
#             report += f"## {question}\n"
#             results = self.results.get(question, [])
#             if results:
#                 for i, result in enumerate(results, 1):
#                     report += f"### Source {i}: {result['title']}\n"
#                     report += f"{result['content']}\n\n"
#             else:
#                 report += "No information found for this question.\n\n"

#         report += "## Conclusion\n"
#         report += f"This report has provided a comprehensive overview of {self.topic} by addressing key questions. Further research could explore additional dimensions or emerging trends in this area.\n"

#         return report

#     def run(self):
#         """Execute the full ReAct pipeline."""
#         print(f"Generating research questions for topic: {self.topic}")
#         self.generate_research_questions()
#         print(f"Generated questions: {self.questions}")
#         print("Gathering information from the web...")
#         self.gather_information()
#         print("Compiling report...")
#         return self.generate_report()

# # Example usage
# if __name__ == "__main__":
#     topic = "Climate Change"
#     agent = ReActAgent(topic)
#     report = agent.run()
    
#     # Save the report to a markdown file
#     with open("research_report.md", "w", encoding="utf-8") as f:
#         f.write(report)
#     print("Report generated and saved to 'research_report.md'")

import os
import json
import logging
import asyncio
from google.generativeai import configure, GenerativeModel
from tavily import TavilyClient
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Configuration parameters
CONFIG = {
    "gemini_model": "gemini-1.5-flash",
    "question_count_min": 5,
    "question_count_max": 6,
    "tavily_max_results": 5,
    "tavily_search_depth": "basic",
    "report_output_file": "research_report.md",
    "retry_attempts": 3,
    "retry_wait_min": 1,
    "retry_wait_max": 5
}

# Configure Gemini API (Step 1: Set Up the LLM)
configure(api_key=GEMINI_API_KEY)

class ReActAgent:
    def __init__(self, topic):
        """Initialize the agent with a topic (Step 2: Implement the Agent Class)."""
        self.topic = topic.strip()
        self.questions = []
        self.results = {}
        self.gemini_model = GenerativeModel(CONFIG["gemini_model"])
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        logger.info(f"Initialized ReActAgent for topic: {self.topic}")

    @retry(stop=stop_after_attempt(CONFIG["retry_attempts"]), wait=wait_exponential(min=CONFIG["retry_wait_min"], max=CONFIG["retry_wait_max"]))
    def generate_research_questions(self):
        """Generate 5-6 research questions using Gemini API (Step 3: Use LLM for Planning)."""
        prompt = f"""
        You are a research assistant. Given the topic '{self.topic}', generate a list of {CONFIG["question_count_min"]}-{CONFIG["question_count_max"]} well-structured research questions covering different aspects of the topic. Return the questions in a JSON list.
        Example:
        ```json
        [
            "What are the main causes of {self.topic}?",
            "How has {self.topic} evolved over time?"
        ]
        ```
        """
        try:
            logger.info(f"Generating research questions for topic: {self.topic}")
            response = self.gemini_model.generate_content(prompt)
            questions_json = response.text.strip()
            if questions_json.startswith("```json") and questions_json.endswith("```"):
                questions_json = questions_json[7:-3].strip()
            questions = json.loads(questions_json)
            # Validate question count
            if not (CONFIG["question_count_min"] <= len(questions) <= CONFIG["question_count_max"]):
                raise ValueError(f"Expected {CONFIG['question_count_min']}-{CONFIG['question_count_max']} questions, got {len(questions)}")
            self.questions = questions
            logger.info(f"Generated {len(self.questions)} questions: {self.questions}")
            return self.questions
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            raise

    async def search_web(self, question):
        """Search the web for a question using Tavily API (Step 4: Use Web Search for Acting)."""
        @retry(stop=stop_after_attempt(CONFIG["retry_attempts"]), wait=wait_exponential(min=CONFIG["retry_wait_min"], max=CONFIG["retry_wait_max"]))
        async def _search():
            try:
                logger.info(f"Searching web for: {question}")
                response = self.tavily_client.search(
                    query=question,
                    search_depth=CONFIG["tavily_search_depth"],
                    max_results=CONFIG["tavily_max_results"]
                )
                results = [
                    {"title": result["title"], "content": result["content"] or "No content available"}
                    for result in response["results"]
                ]
                return question, results
            except Exception as e:
                logger.error(f"Error searching for '{question}': {e}")
                return question, []
        
        return await _search()

    async def gather_information(self):
        """Gather information for all questions concurrently (Step 4 continued)."""
        tasks = [self.search_web(question) for question in self.questions]
        results = await asyncio.gather(*tasks)
        for question, search_results in results:
            self.results[question] = search_results
            logger.info(f"Collected {len(search_results)} results for: {question}")

    def format_report(self, title, intro, sections, conclusion):
        """Format the report into markdown (Step 5: Compile the Final Report)."""
        report = f"# {title}\n\n"
        report += f"## {intro['header']}\n{intro['content']}\n\n"
        for section in sections:
            report += f"## {section['question']}\n"
            if section["results"]:
                for i, result in enumerate(section["results"], 1):
                    report += f"### Source {i}: {result['title']}\n{result['content']}\n\n"
            else:
                report += "No information found for this question.\n\n"
        report += f"## {conclusion['header']}\n{conclusion['content']}\n"
        return report

    def generate_report(self):
        """Compile a structured report (Step 5 continued)."""
        logger.info("Compiling research report")
        title = f"Research Report on {self.topic}"
        intro = {
            "header": "Introduction",
            "content": f"This report explores the topic of {self.topic} by addressing key research questions generated through advanced reasoning. The findings are based on web searches conducted to gather relevant and recent information."
        }
        sections = [
            {"question": question, "results": self.results.get(question, [])}
            for question in self.questions
        ]
        conclusion = {
            "header": "Conclusion",
            "content": f"This report has provided a comprehensive overview of {self.topic} by addressing key questions. Further research could explore additional dimensions or emerging trends in this area."
        }
        report = self.format_report(title, intro, sections, conclusion)
        with open(CONFIG["report_output_file"], "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to {CONFIG['report_output_file']}")
        return report

    async def run(self):
        """Execute the full ReAct pipeline (Steps 2-5)."""
        logger.info(f"Starting ReAct pipeline for topic: {self.topic}")
        self.generate_research_questions()
        if not self.questions:
            logger.error("No questions generated; aborting pipeline")
            return ""
        await self.gather_information()
        return self.generate_report()

# Example usage
if __name__ == "__main__":
    topic = "Climate Change"
    agent = ReActAgent(topic)
    # Run async pipeline
    report = asyncio.run(agent.run())
    if report:
        print(f"Report generated and saved to {CONFIG['report_output_file']}")
    else:
        print("Failed to generate report")