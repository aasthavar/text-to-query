import chainlit as cl 
from langchain.schema.runnable.config import RunnableConfig
from utils.chain import get_chain
from langchain.memory import ConversationBufferMemory

@cl.on_chat_start
async def create_chain():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        ai_prefix="A",
        human_prefix="H",
    )
    execute_query = get_chain(memory=memory)
    cl.user_session.set("chain", execute_query)
    cl.user_session.set("memory", memory)

@cl.on_message
async def main(input_msg: cl.Message):
    chain = cl.user_session.get("chain")
    memory = cl.user_session.get("memory")
    output_msg = cl.Message(content="")

    async for chunk in chain.astream(
        {"question": input_msg.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
    ):

        await output_msg.stream_token(chunk)

    memory.chat_memory.add_user_message(input_msg.content)
    memory.chat_memory.add_ai_message(output_msg.content)
    cl.user_session.set("memory", memory)

    await output_msg.send()