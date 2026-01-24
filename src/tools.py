from langchain.tools import BaseTool
from typing import Type, Any
from pydantic import BaseModel, Field

class SearchPdfToolInput(BaseModel):
    query: str = Field(..., description="The query string to search in the PDF")

class SearchPdfTool(BaseTool):
    name: str = "SearchPdfTool"
    description: str = "Tool used to search information in a PDF given a query"
    args_schema: Type[BaseModel] = SearchPdfToolInput
    
    # Store the user's specific Pinecone database here
    vector_store: Any = None 

    def _run(self, query: str) -> str:
        if self.vector_store:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(query)
            return "\n\n".join([d.page_content for d in docs])
        else:
            return "No PDF uploaded yet."