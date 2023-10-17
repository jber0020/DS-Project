from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import os

class Chatbot:
    def __init__(self) -> None:
        
        db = SQLDatabase.from_uri("postgresql://ds4user:FIT3163!@ds4db.postgres.database.azure.com:5432/elec_db")
        toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

        self.agent_executor = create_sql_agent(
            llm=ChatOpenAI(temperature=0.5),
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

    def run_chatbot(self, question):        
        return self.agent_executor.run(question + "NOTE: When trying to access any tables do not put single quotes around the name")
