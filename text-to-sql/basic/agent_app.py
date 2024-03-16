import chainlit as cl 
from langchain.schema.runnable.config import RunnableConfig
from utils.agent import get_agent
from langchain.memory import ConversationBufferMemory

@cl.on_chat_start
async def create_agent():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        ai_prefix="A",
        human_prefix="H",
    ) 
    agent_executor = get_agent(memory)
    cl.user_session.set("agent", agent_executor)

@cl.on_message
async def main(input_msg: cl.Message):
    agent = cl.user_session.get("agent") # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    response = await cl.make_async(agent)(
        {"question": input_msg.content}, 
        callbacks=[cb], 
        return_only_outputs=True
    )
    await cl.Message(content=response["output"]).send()