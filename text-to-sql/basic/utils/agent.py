import json, boto3, sqlglot
from langchain.llms.bedrock import Bedrock
from langchain.sql_database import SQLDatabase
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor
from langchain.tools.render import render_text_description
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from .prompts import few_shot_sql_prompt_with_history, text_generate_prompt_with_history, agent_prompt_template_with_history

with open("/home/ubuntu/config.json") as file:
    config = json.load(file)

database_uri = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**config["rds_connect"])

rds_db = SQLDatabase.from_uri(database_uri=database_uri)

llm = Bedrock(
    model_id="anthropic.claude-v2:1",
    client=boto3.client("bedrock-runtime"),
    model_kwargs={"max_tokens_to_sample": 512, "temperature": 0.0},
) 

def text_to_sql_tool(question, chain):
    sql_query = chain.invoke({ "question": question })
    return sql_query

def format_sql_query(raw_sql_query):
    sql_query = sqlglot.transpile(raw_sql_query, write='postgres', identify=True)[0]
    return sql_query

def get_db_response(x):
    sql_query = x["sql_query"]
    return rds_db.run(sql_query)    
    # return rds_db.run(sql_query, include_columns=True)

def get_tools(memory):
    create_query = few_shot_sql_prompt_with_history | llm | format_sql_query

    execute_summarize_query = (
        RunnablePassthrough.assign(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"))
        | RunnablePassthrough.assign(sql_query=create_query).assign(db_response=get_db_response)
        | text_generate_prompt_with_history 
        | llm
    )

    tools = [
        Tool(
            name="run_text_to_sql",
            func=lambda question: text_to_sql_tool(question, execute_summarize_query),
            description=(
                "Use when you are asked analytical questions about hotels, stay and other facilities provided."
                " The input should be the question itself."
            ),
        )
    ]
    return tools

def get_agent(memory):
    tools = get_tools(memory)

    agent_prompt = PromptTemplate(
        input_variables = ["chat_history", "question", "agent_scratchpad"],
        partial_variables = {
            "tools": render_text_description(tools),
            "tool_names": ", ".join([t.name for t in tools]),
        },
        template = agent_prompt_template_with_history,
    )

    agent = (
        {
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | agent_prompt
        | llm.bind(stop=["\nObservation"])
        | ReActJsonSingleInputOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )  

    return agent_executor 