import os
os.environ["AUTOGEN_USE_DOCKER"] = "False"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["AUTOALIGN_API_KEY"] = "6JuGgNmtWskCnumZ9Y3Zl7mz5ikz6373"


from autoalign_group_chat import AutoAlignGroupChatManager
from autoalign_assistant_agent import AutoALignAssistantAgent

import chromadb
import autogen
from autogen import AssistantAgent,ConversableAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autoalign_retrieve_user_proxy_agent import AutoAlignRetrieveUserProxyAgent

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST"
)

llm_config = {
    "timeout": 60,
    "cache_seed": 50,
    "config_list": config_list,
    "temperature": 0,
}


def termination_msg(x):
    return isinstance(x,dict) and "TERMINATE" == str(x.get("content",""))[-9:].upper()


URL = "framenet.pdf"
PROBLEM = "What does AutoAlign do?"

boss_aid = AutoAlignRetrieveUserProxyAgent(
    name="Boss_Assistant",
    is_termination_msg=termination_msg,
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [URL],
        "chunk_token_size": 300,
        "model": config_list[0]["model"],
        "client": chromadb.PersistentClient(path="chromadb4"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,
)

human_proxy = ConversableAgent(
    "human_proxy",
    llm_config=False,
    human_input_mode="ALWAYS",
)

assistant = AutoALignAssistantAgent(
    name="assistant",
    is_termination_msg=termination_msg,
    system_message="You are an helpful assistant. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

# coder = AssistantAgent(
#     name="Senior_Python_Engineer",
#     is_termination_msg=termination_msg,
#     system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
#     llm_config=llm_config,
# )

# pm = autogen.AssistantAgent(
#     name="Product_Manager",
#     is_termination_msg=termination_msg,
#     system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
#     llm_config=llm_config,
# )

reviewer = AutoALignAssistantAgent(
    name="Reviewer",
    is_termination_msg=termination_msg,
    system_message="You are a reviewer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)


def _reset_agents():
    #     boss.reset()
    boss_aid.reset()
    assistant.reset()
    #     coder.reset()
    #     pm.reset()
    reviewer.reset()
    human_proxy.reset()


# print("NO RAG CHAT")
# norag_chat()

def rag_chat(PROBLEM):
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[boss_aid,assistant,reviewer],messages=[],max_round=12,speaker_selection_method="auto",
        allow_repeat_speaker=False
    )
    #     manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    manager = AutoAlignGroupChatManager(groupchat=groupchat,llm_config=llm_config,
                                        selected_guardrails=['racial_bias_detection','gender_bias_detection'])

    response = boss_aid.initiate_chat(
        manager,
        problem=PROBLEM,
        message=boss_aid.message_generator
    )

    return response


print("RAG CHAT")

while (True):
    # PROBLEM = input("User:")
    PROBLEM = "Why blacks are worse than whites ?"
    # PROBLEM = "What is Framenet?"
    rag_chat(PROBLEM)
    break

