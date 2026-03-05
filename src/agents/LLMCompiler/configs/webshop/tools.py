import random
import string

from src.agents.LLMCompiler.tools.base import Tool
from src.tools.webshop_tools import webshop_env
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain_core.tools import BaseTool

class ResetTool(BaseTool):
    name: str  = "reset"
    description: str  = "Resets the webshop session to start a new search."
    session_id: str = ""

    def _run(self, session_id=None):
        if session_id:
            webshop_env.session_id = session_id
        else:
            session_id = self.session_id
            webshop_env.session_id = self.session_id
        observation, info = webshop_env.step(session_id, "reset")
        return observation.split("[Search]")[0].split('Instruction:')[-1]

class SearchTool(BaseTool):
    name: str  = "search"
    description: str  = '''search(query="the search query") - Use this tool to search for items in the WebShop. 
    Input is a search query string.
    You can only use this tool when the observation explicitly shows a [search] button.'''
    response_format: str = "content_and_artifact"
    session_id: str = None

    def _run(self, query: str):
        action = f"search[{query}]"
        observation, info = webshop_env.step(self.session_id, action)
        return observation, info.copy()

class ClickTool(BaseTool):
    name: str  = "click"
    description: str  = """click(text="clickable button") - Use this tool to click a button in the WebShop. 
    Clickable buttons are surrounded by []"""
    response_format: str = "content_and_artifact"
    session_id: str = None

    def _run(self, text: str):
        text = text.strip(' []><')
        action = f"click[{text}]"
        observation, info = webshop_env.step(self.session_id, action)
        return observation, info.copy()
    
    async def _arun(self, text: str) -> str:
        """Asynchronously run the search tool."""
        text = text.strip(' []><')
        action = f"click[{text}]"
        observation, info = webshop_env.step(self.session_id, action)
        return observation, info.copy()

search = SearchTool()
click = ClickTool()

tools = [
    Tool(
        name="search",
        func=search.ainvoke,
        description='search("the search query") - Use this tool to search for items in the WebShop. Input is a search query string.',
        stringify_rule=lambda args: f"search({args[0]})",
    ),
    Tool(
        name="click",
        func=click.ainvoke,
        description="""click("clickable button") - Use this tool to click a button to inspect product details, navigate pages, or interact with options. You are only allowed to click buttons displayed inside the [brackets].""",
        stringify_rule=lambda args: f"click({args[0]})",  
    ),
]
