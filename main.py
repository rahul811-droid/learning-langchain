from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
load_dotenv()

tavily = TavilySearch(max_results=3)

@tool
def search(query: str) -> str:
    """Search for information, such as the weather in a city."""
    print(f"Searching for {query}...")
    return str(tavily.invoke({"query": query}))

llm = ChatOllama(model="qwen3:4b",temperature=0.7)
tools = [search]
agent = create_agent(model =llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="Search for  3 job an ai engineer using langchain ")})
    print(result)
if __name__ == "__main__":
    main()