from typing import Optional, Union, Literal, Callable, Dict

from autogen import AssistantAgent
from autogen.agentchat.agent import Agent


class AutoALignAssistantAgent(AssistantAgent):
    def __init__(
            self,
            name: str,
            system_message: Optional[str] = AssistantAgent.DEFAULT_SYSTEM_MESSAGE,
            llm_config: Optional[Union[Dict, Literal[False]]] = None,
            is_termination_msg: Optional[Callable[[Dict], bool]] = None,
            max_consecutive_auto_reply: Optional[int] = None,
            human_input_mode: Optional[str] = "NEVER",
            description: Optional[str] = None,
            topics: Optional[Dict[str, Dict[str, str]]] = None,
            **kwargs,
    ):
        super(AutoALignAssistantAgent, self).__init__(name, system_message, llm_config, is_termination_msg,
                                                      max_consecutive_auto_reply, human_input_mode,
                                                      description, **kwargs)
        self.topics = topics

    def _append_oai_message(self, message: Union[Dict, str], role, conversation_id: Agent) -> bool:
        """Append a message to the ChatCompletion conversation.

        If the message received is a string, it will be put in the "content" field of the new dictionary.
        If the message received is a dictionary but does not have any of the three fields "content", "function_call", or "tool_calls",
            this message is not a valid ChatCompletion message.
        If only "function_call" or "tool_calls" is provided, "content" will be set to None if not provided, and the role of the message will be forced "assistant".

        Args:
            message (dict or str): message to be appended to the ChatCompletion conversation.
            role (str): role of the message, can be "assistant" or "function".
            conversation_id (Agent): id of the conversation, should be the recipient or sender.

        Returns:
            bool: whether the message is appended to the ChatCompletion conversation.
        """
        message = self._message_to_dict(message)
        # create oai message to be appended to the oai conversation that can be passed to oai directly.
        oai_message = {
            k: message[k]
            for k in ("content", "function_call", "tool_calls", "tool_responses", "tool_call_id", "name", "context", "problem", "docs")
            if k in message and message[k] is not None
        }
        if "content" not in oai_message:
            if "function_call" in oai_message or "tool_calls" in oai_message:
                oai_message["content"] = None  # if only function_call is provided, content will be set to None.
            else:
                return False

        if message.get("role") in ["function", "tool"]:
            oai_message["role"] = message.get("role")
        elif "override_role" in message:
            # If we have a direction to override the role then set the
            # role accordingly. Used to customise the role for the
            # select speaker prompt.
            oai_message["role"] = message.get("override_role")
        else:
            oai_message["role"] = role

        if oai_message.get("function_call", False) or oai_message.get("tool_calls", False):
            oai_message["role"] = "assistant"  # only messages with role 'assistant' can have a function call.
        self._oai_messages[conversation_id].append(oai_message)
        return True
