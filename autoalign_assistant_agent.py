from typing import Optional, Union, Literal, Callable, Dict

from autogen import AssistantAgent


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
