import yaml
import json
import requests
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from src.tools import SearchPdfTool
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from unstructured.partition.html import partition_html
from langchain.prompts import PromptTemplate

class SearchToolInput(BaseModel):
    query: str = Field(..., description="The query which helps to do the internet search.")

class SearchToolInternet(BaseTool):
    name: str = "SearchToolInternet"
    description: str = "Search the internet relevant data about the query."
    args_schema: Type[BaseModel] = SearchToolInput

    def _run(self, query: str) -> str:
        top_result_to_return = 1
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': os.getenv("SERPER_API_KEY"), # SECURED
            'content-type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        
        if 'organic' not in response.json():
            return "Sorry, I couldn't find anything."
        else:
            results = response.json()['organic']
            string = []
            for result in results[:top_result_to_return]:
                try:
                    string.append('\n'.join([
                        f"Title: {result['title']}", f"Link: {result['link']}",
                        f"Snippet: {result['snippet']}", "\n-----------------"
                    ]))
                except KeyError:
                    next
            return '\n'.join(string)

class BrowserToolInput(BaseModel):
    website :str = Field(...,description="The website that we browse.")

class BrowserTool(BaseTool):
    name: str = "BrowserTool"
    description: str = "Browse the website"
    args_schema: Type[BaseModel]=BrowserToolInput
    
    def _run(self, website: str) -> str:
        token = os.getenv("BROWSERLESS_API_KEY") # SECURED
        url = f"https://chrome.browserless.io/content?token={token}"
        payload = json.dumps({"url": website})
        headers = {'cache-control': 'no-cache', 'content-type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=payload)
        
        elements = partition_html(text=response.text)
        content = "\n\n".join([str(el) for el in elements])
        content = [content[i:i + 1000] for i in range(0, len(content), 1000)]
        summaries = []
        for chunk in content:
            agent = Agent(
                role='Principal Researcher',
                goal='Do amazing researches and summaries',
                backstory="You're a Principal Researcher.",
                allow_delegation=False,
                verbose=False)
            task = Task(
                agent=agent,
                description=f'Summarize this: {chunk}'
            )
            summaries.append(task.execute())
        return "\n".join(summaries)


class AgenticWorkflow:
    def __init__(self, vector_db=None):
        self.vector_db = vector_db  # Stores the user's specific DB
        self.agents_config = self._load_yaml('config/agents.yaml')
        self.tasks_config = self._load_yaml('config/tasks.yaml')
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.llm_strict = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def _load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
        
    def _contextualize_query(self, query: str, chat_history: str) -> str:
        """Rewrites an ambiguous query into a standalone query using chat history."""
        
        # If there is no history, the query doesn't need rewriting
        if not chat_history or chat_history.strip() == "":
            return query

        # The industry-standard prompt for query rewriting
        contextualize_q_system_prompt = """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is.
        
        Chat History:
        {chat_history}
        
        Latest User Question: {query}
        
        Standalone Question:"""
        
        prompt = PromptTemplate(
            template=contextualize_q_system_prompt,
            input_variables=["chat_history", "query"]
        )
        
        # Chain the prompt with your strict, low-temperature LLM
        chain = prompt | self.llm_strict 
        
        # Generate the new query
        response = chain.invoke({"chat_history": chat_history, "query": query})
        
        # Return the clean text
        return response.content

    def run(self, query: str, chat_history: str):
        # Pass the database to your custom tool!
        query = self._contextualize_query(query, chat_history)

        search_tool = SearchPdfTool(vector_store=self.vector_db)
        search_tool_interent = SearchToolInternet()
        
        researcher = Agent(
            role=self.agents_config['researcher']['role'],
            goal=self.agents_config['researcher']['goal'],
            backstory=self.agents_config['researcher']['backstory'],
            tools=[search_tool, search_tool_interent],
            llm=self.llm_strict,
            verbose=True,
            allow_delegation=False
        )
        
        strategist = Agent(
            role=self.agents_config['strategist']['role'],
            goal=self.agents_config['strategist']['goal'],
            backstory=self.agents_config['strategist']['backstory'],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        task_research = Task(
            description=self.tasks_config['research_task']['description'].format(query=query),
            expected_output=self.tasks_config['research_task']['expected_output'],
            agent=researcher
        )

        task_answer = Task(
            description=self.tasks_config['answer_task']['description'].format(
                query=query, chat_history=chat_history
            ),
            expected_output=self.tasks_config['answer_task']['expected_output'],
            agent=strategist,
            context=[task_research]
        )

        crew = Crew(
            agents=[researcher, strategist],
            tasks=[task_research, task_answer],
            process=Process.sequential
        )

        return str(crew.kickoff())