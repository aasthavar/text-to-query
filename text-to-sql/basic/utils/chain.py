import json, boto3, sqlglot
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from .prompts import few_shot_sql_prompt_with_history, text_generate_prompt_with_history

with open("/home/ubuntu/config.json") as file:
    config = json.load(file)

database_uri = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**config["rds_connect"])

rds_db = SQLDatabase.from_uri(database_uri=database_uri)

llm = Bedrock(
    model_id="anthropic.claude-v2:1",
    client=boto3.client("bedrock-runtime"),
    model_kwargs={"max_tokens_to_sample": 512, "temperature": 0.0},
) 

def format_sql_query(raw_sql_query):
    sql_query = sqlglot.transpile(raw_sql_query, write='postgres', identify=True)[0]
    return sql_query

def get_db_response(x):
    sql_query = x["sql_query"]
    return rds_db.run(sql_query)

def get_chain(memory):
    create_query = few_shot_sql_prompt_with_history | llm | format_sql_query

    load_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")
    )
    execute_query = (
        load_memory
        | RunnablePassthrough.assign(sql_query=create_query).assign(db_response=get_db_response)
        | text_generate_prompt_with_history 
        | llm
    )
    return execute_query