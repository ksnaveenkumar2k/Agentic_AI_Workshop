import re
from typing import TypedDict, Literal, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END

# Custom Mathematical Functions
def plus(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

# Define Tools for LangChain
math_tools = [
    Tool(
        name="plus",
        func=plus,
        description="Add two numbers. Input: two numbers (e.g., '5, 3')."
    ),
    Tool(
        name="subtract",
        func=subtract,
        description="Subtract two numbers. Input: two numbers (e.g., '5, 3')."
    ),
    Tool(
        name="multiply",
        func=multiply,
        description="Multiply two numbers. Input: two numbers (e.g., '5, 3')."
    ),
    Tool(
        name="divide",
        func=divide,
        description="Divide two numbers. Input: two numbers (e.g., '5, 3')."
    ),
]

# Initialize LLM (Gemini 1.5 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="your-google-api-key")
llm_with_tools = llm.bind_tools(math_tools)

# Define the State for LangGraph
class AgentState(TypedDict):
    query: str
    is_math: bool
    tool_call: Optional[dict]
    response: str

# Chatbot Node: Determines if the query is mathematical and prepares tool call
def chatbot_node(state: AgentState) -> AgentState:
    query = state["query"].lower()
    
    # Regex patterns to identify mathematical queries
    math_patterns = [
        r"\d+\s*(plus|\+|\-|\*|multiply|times|divide|divided by|/)\s*\d+",
        r"what is \d+\s*(plus|\+|\-|\*|multiply|times|divide|divided by|/)\s*\d+",
        r"how much is \d+\s*(plus|\+|\-|\*|multiply|times|divide|divided by|/)\s*\d+"
    ]
    state["is_math"] = any(re.search(pattern, query) for pattern in math_patterns)
    
    if state["is_math"]:
        # Extract numbers and operation
        numbers = re.findall(r"\d+", query)
        if len(numbers) != 2:
            state["response"] = "Please provide exactly two numbers for the operation."
            state["is_math"] = False
            state["tool_call"] = None
            return state
        
        a, b = map(float, numbers)
        operation = "plus"
        if any(word in query for word in ["subtract", "-"]):
            operation = "subtract"
        elif any(word in query for word in ["multiply", "times", "*"]):
            operation = "multiply"
        elif any(word in query for word in ["divide", "divided by", "/"]):
            operation = "divide"
        
        state["tool_call"] = {
            "name": operation,
            "args": {"a": a, "b": b}
        }
    else:
        # Non-mathematical query, use LLM directly
        response = llm.invoke(query).content
        state["response"] = response
        state["tool_call"] = None
    
    return state

# Tools Node: Executes the mathematical operation
def tools_node(state: AgentState) -> AgentState:
    tool_name = state["tool_call"]["name"]
    args = state["tool_call"]["args"]
    
    # Find and execute the corresponding tool
    tool = next((t for t in math_tools if t.name == tool_name), None)
    if tool:
        try:
            result = tool.func(**args)
            state["response"] = f"The result of the operation is: {result}"
        except Exception as e:
            state["response"] = f"Error during calculation: {str(e)}"
    else:
        state["response"] = "Unknown mathematical operation."
    
    return state

# Conditional Routing: Route to tools or end
def route_to_tools(state: AgentState) -> Literal["tools", "__end__"]:
    if state["is_math"] and state["tool_call"]:
        return "tools"
    return "__end__"

# Build the LangGraph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", tools_node)

# Define edges
workflow.add_conditional_edges(
    "chatbot",
    route_to_tools,
    {
        "tools": "tools",
        "__end__": END
    }
)
workflow.add_edge("tools", END)

# Set entry point
workflow.set_entry_point("chatbot")

# Compile the graph
graph = workflow.compile()

# Function to Run the Agent
def run_agent(query: str) -> str:
    state = {"query": query, "is_math": False, "tool_call": None, "response": ""}
    result = graph.invoke(state)
    return result["response"]

# Test the Agent
if __name__ == "__main__":
    test_queries = [
        "What is 10 plus 5?",
        "How much is 20 divided by 4?",
        "What is the time now?",
        "What is 6 times 3?",
        "What is 15 - 9?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print(f"Response: {run_agent(query)}\n")