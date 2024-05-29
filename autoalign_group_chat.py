import json
import os
import aiohttp
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Union

import requests

from autoalign_assistant_agent import AutoALignAssistantAgent
from autogen import ConversableAgent, GroupChat, Agent, GroupChatManager, NoEligibleSpeaker, ChatResult
from autogen.code_utils import content_str

NEW_MITIGATION_PROMPT = "We received the following response from you: {completion}. " \
                        "This response doesn't meant our expectation." \
                        "Please generate a new response that follows the" \
                        "previously given instructions (system message), your assigned role and" \
                        " 'past chat history or past interactions'." \
                        "Please also ensure that you are grounded on the current discussion. New Response:"

VERIFIER_AGENT_SYSTEM_PROMPT: str = """You are an AI assistant tasked with verifying the output of other AI language models. Your role is to carefully analyze the responses provided by the other models and determine if they are accurate, coherent, and appropriate for the given system message and chat history.

When presented with a response from another model, you should:
1. Carefully read through the entire response.
2. Assess the accuracy of the information provided.
3. Evaluate the coherence and logical flow of the response.
4. Determine if the response is appropriate and relevant to the original prompt.
5. Provide a concise verdict on whether the response is acceptable or not, along with a brief explanation.

Your own responses should be clear, well-structured, and provide a definitive assessment of the other model's output. Based on the chat history and bots\'s system message, check whether the response is appropriate or not. 
Remember, your role is to be an objective and impartial judge of the other models' responses. Avoid personal opinions or biases, and focus on providing a fair and accurate assessment based on the criteria outlined above.
 
Only provide \'Yes\' for answer,if it is appropriate, otherwise provide \'No\'"""

select_answer_multiple_template: str = """You provided more than one answer in your text, please return just the answer for the given speaker, its system message and its response. To determine the answer use these prioritised rules:
1. If the topic of chat history and response's topic are different, answer would be 'No'
2. If the response is a general sentence which can be used by anybody and does not have any topic associated with it then the answer would be 'Yes'
3. If the response is according to the system message then answer would be 'Yes'
4. Choose the answer based on the conversation that has occurred up until this point and the speaker's system message
The answers are case-sensitive and should not be abbreviated or changed.
Respond with ONLY the answer and DO NOT provide a reason."""

select_answer_none_template: str = """You didn't choose an answer. As a reminder, to determine the answer use these prioritised rules:
1. If the topic of chat history and response's topic are different, answer would be 'No'
2. If the response is a general sentence which can be used by anybody and does not have any topic associated with it then the answer would be 'Yes'
3. If the response is according to the system message then answer would be 'Yes'
4. Choose the answer based on the conversation that has occurred up until this point and the speaker's system message
The answers are case-sensitive and should not be abbreviated or changed.
Respond with ONLY the answer and DO NOT provide a reason.
The only names that are accepted are 'Yes' and 'No'.
Respond with ONLY the answer and DO NOT provide a reason."""

verification_prompt = """Following is the system message and response for the agent: {name}
Provide 'Yes' or 'No' as answer if the response is appropriate according to system message and chat history.

To determine the answer use these prioritised rules:
1. If the topic of chat history and response's topic are different, answer would be 'No'
2. If the response is a general sentence which can be used by anybody and does not have any topic associated with it then the answer would be 'Yes'
3. If the response is according to the system message then answer would be 'Yes'
4. Choose the answer based on the conversation that has occurred up until this point and the speaker's system message


By following above intructions, provide the answer below-
agent: {name}
system message:{system_message}
response: {response}
Answer ('Yes' or 'No'):"""

DEFAULT_CONFIG = {
    "pii_fast": {
        "mode": "OFF",
        "mask": False,
        "enabled_types": [
            "[BANK ACCOUNT NUMBER]",
            "[CREDIT CARD NUMBER]",
            "[DATE OF BIRTH]",
            "[DATE]",
            "[DRIVER LICENSE NUMBER]",
            "[EMAIL ADDRESS]",
            "[RACE/ETHNICITY]",
            "[GENDER]",
            "[IP ADDRESS]",
            "[LOCATION]",
            "[MONEY]",
            "[ORGANIZATION]",
            "[PASSPORT NUMBER]",
            "[PASSWORD]",
            "[PERSON NAME]",
            "[PHONE NUMBER]",
            "[PROFESSION]",
            "[SOCIAL SECURITY NUMBER]",
            "[USERNAME]",
            "[SECRET_KEY]",
            "[TRANSACTION_ID]",
            "[RELIGION]",
        ],
    },
    "confidential_detection": {"mode": "OFF"},
    "gender_bias_detection": {"mode": "OFF"},
    "harm_detection": {"mode": "OFF"},
    "text_toxicity_extraction": {"mode": "OFF"},
    "racial_bias_detection": {"mode": "OFF"},
    "tonal_detection": {"mode": "OFF"},
    "jailbreak_detection": {"mode": "OFF"},
    "intellectual_property": {"mode": "OFF"},
}


class AutoAlignVerifier:
    def __init__(self, model_client_cls: Any = None) -> None:
        """
        This is the class which holds all the verification logic
        You can find the system messages and some prompt templates above
        Args:
            model_client_cls (Any): This parameter is used when we are using a custom LLM agent like Gemini, which
            autogen natively does not support.
        """
        self.model_client_cls = model_client_cls

    def verify_and_mitigate_response(self, speaker: AutoALignAssistantAgent, reply: Union[str, Dict[str, Any]],
                                     messages: List[Union[str, Dict]],
                                     sender: ConversableAgent) -> Union[str, Dict[str, Any]]:
        """
        This function verifies and mitigates the response iteratively, at every try it verifies whether the generated
        response/reply is appropriate or not. Based on the verifier's response mitigation logic is applied which
        takes in the previous response and asks the speaker to generate a better response.

        Args:
            speaker (AutoALignAssistantAgent): The conversation agent that provided the reply.
            reply (Union[str, Dict[str, Any]]): The response which is under test
            messages (List[Union[str, Dict]]): chat history
            sender (ConversableAgent): the sender of message

        Returns:
            Union[str, Dict[str, Any]]: mitigated reply if verifier finds it not appropriate otherwise it will remain
            the same.
        """
        trial = 0
        while trial < 5:
            mitigation_required = self.validate_speaker_output(speaker, messages, reply)
            if mitigation_required:
                trial += 1
                reply = self.mitigate_response(speaker, reply, trial, sender)
                if trial == 5:
                    print("%%%%%%%%%%%%%%%%%% Max trials reached %%%%%%%%%%%%%%%%%%%%%%\n")
            else:
                break
        return reply

    async def a_verify_and_mitigate_response(self, speaker: AutoALignAssistantAgent, reply: Union[str, Dict[str, Any]],
                                             messages: List[Union[str, Dict]],
                                             sender: ConversableAgent) -> Union[str, Dict[str, Any]]:
        """
        This is the asynchronous version of above function.
        This function verifies and mitigates the response iteratively, at every try it verifies whether the generated
        response/reply is appropriate or not. Based on the verifier's response mitigation logic is applied which
        takes in the previous response and asks the speaker to generate a better response.

        Args:
            speaker (AutoALignAssistantAgent): The conversation agent that provided the reply.
            reply (Union[str, Dict[str, Any]]): The response which is under test
            messages (List[Union[str, Dict]]): chat history
            sender (ConversableAgent): the sender of message

        Returns:
            Union[str, Dict[str, Any]]: mitigated reply if verifier finds it not appropriate otherwise it will remain
            the same.
        """
        trial = 0
        while trial < 5:
            mitigation_required = self.a_validate_speaker_output(speaker, messages, reply)
            if mitigation_required:
                trial += 1
                reply = await self.a_mitigate_response(speaker, reply, trial, sender)
                if trial == 5:
                    print("%%%%%%%%%%%%%%%%%% Max trials reached %%%%%%%%%%%%%%%%%%%%%%\n")
            else:
                break
        return reply

    @staticmethod
    def mitigate_response(speaker: ConversableAgent, reply: Union[str, Dict[str, Any]], trial: int,
                          sender: ConversableAgent) -> Union[str, Dict[str, Any]]:
        """
        This function runs the mitigation logic that is asks the speaker to regenerate the response.

        Args:
            speaker (ConversableAgent): The conversation agent that provided the reply.
            reply (Union[str, Dict[str, Any]]): The response which is under test
            trial (int): Current trial number
            sender (ConversableAgent): the sender of message

        Returns:
            Union[str, Dict[str, Any]]: Mitigated response

        """
        completion = reply['content'] if isinstance(reply, dict) else reply

        def autoalign_mitigation_flow(received_messages: list[dict[str, Any]] | None):
            new_message_content = NEW_MITIGATION_PROMPT.format(completion=completion)
            if received_messages is not None:
                received_messages.append({"role": "user", "content": new_message_content})
                return received_messages
            else:
                return [{"role": "user", "content": new_message_content}]

        speaker.register_hook("process_all_messages_before_reply", autoalign_mitigation_flow)
        reply = speaker.generate_reply(sender=sender)

        print(f"%%%%%%%%%%% AutoAlign Log: Previous completion Trial-{trial} %%%%%%%%%%%\n")
        print(speaker.name)
        print()
        print(completion)
        print(f"%%%%%%%%%%% AutoAlign Log: New Mitigated completion Trial-{trial} %%%%%%%%%%%\n")
        print()
        print(speaker.name)
        print()
        print(reply['content'] if isinstance(reply, dict) else reply)
        speaker.hook_lists["process_all_messages_before_reply"] = []
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        return reply

    @staticmethod
    async def a_mitigate_response(speaker: ConversableAgent, reply: Union[str, Dict[str, Any]], trial: int,
                                  sender: ConversableAgent) -> Union[str, Dict[str, Any]]:
        """
        This is asynchronous version of above function.
        This function runs the mitigation logic that is asks the speaker to regenerate the response.

        Args:
            speaker (ConversableAgent): The conversation agent that provided the reply.
            reply (Union[str, Dict[str, Any]]): The response which is under test
            trial (int): Current trial number
            sender (ConversableAgent): the sender of message

        Returns:
            Union[str, Dict[str, Any]]: Mitigated response

        """
        completion = reply['content'] if isinstance(reply, dict) else reply

        def autoalign_mitigation_flow(received_messages: list[dict[str, Any]] | None):
            new_message_content = NEW_MITIGATION_PROMPT.format(completion=completion)
            if received_messages is not None:
                received_messages.append({"role": "user", "content": new_message_content})
                return received_messages
            else:
                return [{"role": "user", "content": new_message_content}]

        speaker.register_hook("process_all_messages_before_reply", autoalign_mitigation_flow)
        reply = speaker.a_generate_reply(sender=sender)
        trial += 1
        print(f"%%%%%%%%%%% AutoAlign Log: Previous completion Trial-{trial} %%%%%%%%%%%\n")
        print(speaker.name)
        print()
        print(completion)
        print(f"%%%%%%%%%%% AutoAlign Log: New Mitigated completion Trial-{trial} %%%%%%%%%%%\n")
        print()
        print(speaker.name)
        print()
        print(reply['content'] if isinstance(reply, dict) else reply)
        speaker.hook_lists["process_all_messages_before_reply"] = []
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        return reply

    @staticmethod
    def check_guardrails(request_url: str, text: str, selected_guardrails: List[str]):
        """Checks whether the given text passes through the applied guardrails."""
        config = DEFAULT_CONFIG.copy()
        for task in selected_guardrails:
            config[task]['mode'] = "DETECT"

        request_body = {
            "prompt": text,
            "config": config
        }
        api_key = os.getenv("AUTOALIGN_API_KEY")
        if api_key is None:
            raise ValueError("Please AUTOALIGN_API_KEY as an environment variable")

        header = {
            "x-api-key": api_key
        }
        json_data = json.dumps(request_body).encode('utf8')
        s = requests.Session()
        guard_response = []
        with s.post(request_url, data=json_data, headers=header, stream=True) as resp:
            for line in resp.iter_lines():
                guard_response.append(json.loads(line))
        for resp in guard_response:
            if resp['guarded']:
                return True
        return False

    @staticmethod
    async def a_check_guardrails(request_url: str, text: str, selected_guardrails: List[str]):
        """Checks whether the given text passes through the applied guardrails."""
        config = DEFAULT_CONFIG.copy()
        for task in selected_guardrails:
            config[task]['mode'] = "DETECT"

        request_body = {
            "prompt": text,
            "config": config
        }

        header = {
            "x-api-key": "6JuGgNmtWskCnumZ9Y3Zl7mz5ikz6373"
        }
        json_data = json.dumps(request_body).encode('utf8')
        guard_response = []
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    url=request_url,
                    headers=header,
                    json=request_body,
            ) as response:
                if response.status != 200:
                    raise ValueError(
                        f"AutoAlign call failed with status code {response.status}.\n"
                        f"Details: {await response.text()}"
                    )
                async for line in response.content:
                    guard_response.append(json.loads(line))
        for resp in guard_response:
            if resp['guarded']:
                return True
        return False

    def validate_speaker_output(self, speaker: ConversableAgent, messages: Optional[List[Dict]], reply: str) -> bool:
        """
        This function executes the validation/verification logic for the generated response.

        Args:
            speaker (ConversableAgent): The conversation agent that provided the reply.
            messages (Optional[List[Dict]]): Chat History
            reply (Union[str, Dict[str, Any]]): The response which is under test

        Returns:
            bool: if True then mitigation is required otherwise if False then mitigation is not required

        """
        completion = reply['content'] if isinstance(reply, dict) else reply

        checking_agent = ConversableAgent("checking_agent", default_auto_reply=5)
        checking_agent.register_reply(
            [ConversableAgent, None],
            reply_func=self.validate_verifier_answer,  # Validate each response
            remove_other_reply_funcs=True,
        )

        system_message = VERIFIER_AGENT_SYSTEM_PROMPT
        # Agent for selecting a single agent name from the response
        verifier_agent = ConversableAgent(
            "speaker_selection_agent",
            system_message=system_message,
            chat_messages={checking_agent: messages.copy()},
            llm_config=speaker.llm_config,
            human_input_mode="NEVER",
            # Suppresses some extra terminal outputs, outputs will be handled by select_speaker_auto_verbose
        )
        if self.model_client_cls is not None:
            verifier_agent.register_model_client(self.model_client_cls)

        # Run the speaker selection chat
        result = checking_agent.initiate_chat(
            verifier_agent,
            cache=None,  # don't use caching for the speaker selection chat
            message=verification_prompt.format(system_message=speaker.system_message, response=completion,
                                               name=speaker.name),
            max_turns=4,  # Limiting the chat to the number of attempts, including the initial one
            clear_history=False,
            silent=True,  # Base silence on the verbose attribute
        )

        answer = self._process_answer_selection_result(result)
        #  print(f"#### Verifier's output: {"Provided response is ok" if answer and 'yes' in answer.lower() else "Provided response is not ok, regenerate response"}####\n")
        if answer and 'yes' in answer.lower():
            return False
        else:
            return True

    async def a_validate_speaker_output(self, speaker: ConversableAgent, messages: Optional[List[Dict]],
                                        reply: str) -> bool:
        """
        Asynchronous version of above function.
        This function executes the validation/verification logic for the generated response.

        Args:
            speaker (ConversableAgent): The conversation agent that provided the reply.
            messages (Optional[List[Dict]]): Chat History
            reply (Union[str, Dict[str, Any]]): The response which is under test

        Returns:
            bool: if True then mitigation is required otherwise if False then mitigation is not required

        """
        completion = reply['content'] if isinstance(reply, dict) else reply

        checking_agent = ConversableAgent("checking_agent", default_auto_reply=5)
        checking_agent.register_reply(
            [ConversableAgent, None],
            reply_func=self.validate_verifier_answer,  # Validate each response
            remove_other_reply_funcs=True,
        )

        system_message = VERIFIER_AGENT_SYSTEM_PROMPT
        # Agent for selecting a single agent name from the response
        verifier_agent = ConversableAgent(
            "verifier_agent",
            system_message=system_message,
            chat_messages={checking_agent: messages.copy()},
            llm_config=speaker.llm_config,
            human_input_mode="NEVER",
            # Suppresses some extra terminal outputs, outputs will be handled by select_speaker_auto_verbose
        )
        if self.model_client_cls is not None:
            verifier_agent.register_model_client(self.model_client_cls)

        # Run the speaker selection chat
        result = await checking_agent.a_initiate_chat(
            verifier_agent,
            cache=None,  # don't use caching for the speaker selection chat
            message=verification_prompt.format(system_message=speaker.system_message, response=completion,
                                               name=speaker.name),
            max_turns=4,  # Limiting the chat to the number of attempts, including the initial one
            clear_history=False,
            silent=True,  # Base silence on the verbose attribute
        )

        answer = self._process_answer_selection_result(result)
        if answer and 'yes' in answer.lower():
            return False
        else:
            return True

    @staticmethod
    def validate_verifier_answer(recipient: ConversableAgent,
                                 messages: Optional[List[Dict]] = None,
                                 sender: Optional[Agent] = None,
                                 config: Optional[Any] = None,
                                 ) -> Tuple[bool, Union[str, Dict, None]]:
        """
        Validates the verifier's answer by adding constraints to it

        Args:
            messages (Optional[List[Dict]]): Chat History

        Returns:

        """
        select_answer = messages[-1]["content"].strip()

        mentions = _mentioned_answer(select_answer)

        if len(mentions) == 1:
            # Success on retry, we have just one name mentioned
            selected_answer = next(iter(mentions))

            # Add the selected agent to the response so we can return it
            messages.append({"role": "user", "content": f"[ANSWER SELECTED]{selected_answer}"})

        elif len(mentions) > 1:
            # More than one name on requery so add additional reminder prompt for next retry
            return True, {
                "content": select_answer_multiple_template,
                "override_role": "system",
            }

        else:
            return True, {
                "content": select_answer_none_template,
                "override_role": "system",
            }

        return True, None

    def _process_answer_selection_result(self, result: ChatResult) -> str:
        """Checks the result of the verifier agent

        Used by validate_speaker_output and a_validate_speaker_output.

        Args:
            result (ChatResult): output of the agent's chat

        Returns:
            str: filtered result which is either 'Yes' or 'No' """
        if len(result.chat_history) > 0:

            final_message = result.chat_history[-1]["content"]

            if "[ANSWER SELECTED]" in final_message:

                return self.answer_by_name(final_message.replace("[ANSWER SELECTED]", ""))

            else:  # "[ANSWER SELECTION FAILED]"
                return "Yes"

    def answer_by_name(self, provided_answer: str):
        """
        Applies stricter condition on answer provided by the agent
        Args:
            provided_answer (str): Answer provided by the agent

        Returns:
            str: filtered response

        """
        filtered_answers = [answer for answer in ["Yes", "No"] if answer == provided_answer]

        if len(filtered_answers) > 1:
            raise NotImplementedError("Multiple answers returned")

        return filtered_answers[0] if filtered_answers else None


@dataclass
class AutoAlignGroupChat(GroupChat):
    model_client_cls: Optional[Any] = None

    def _auto_select_speaker(
            self,
            last_speaker: Agent,
            selector: ConversableAgent,
            messages: Optional[List[Dict]],
            agents: Optional[List[Agent]],
    ) -> Agent:
        """Selects next speaker for the "auto" speaker selection method. Utilises its own two-agent chat to determine the next speaker and supports requerying.

        Speaker selection for "auto" speaker selection method:
        1. Create a two-agent chat with a speaker selector agent and a speaker validator agent, like a nested chat
        2. Inject the group messages into the new chat
        3. Run the two-agent chat, evaluating the result of response from the speaker selector agent:
            - If a single agent is provided then we return it and finish. If not, we add an additional message to this nested chat in an attempt to guide the LLM to a single agent response
        4. Chat continues until a single agent is nominated or there are no more attempts left
        5. If we run out of turns and no single agent can be determined, the next speaker in the list of agents is returned

        Args:
            last_speaker Agent: The previous speaker in the group chat
            selector ConversableAgent:
            messages Optional[List[Dict]]: Current chat messages
            agents Optional[List[Agent]]: Valid list of agents for speaker selection

        Returns:
            Dict: a counter for mentioned agents.
        """

        # If no agents are passed in, assign all the group chat's agents
        if agents is None:
            agents = self.agents

        # The maximum number of speaker selection attempts (including requeries)
        # is the initial speaker selection attempt plus the maximum number of retries.
        # We track these and use them in the validation function as we can't
        # access the max_turns from within validate_speaker_name.
        max_attempts = 1 + self.max_retries_for_selecting_speaker
        attempts_left = max_attempts
        attempt = 0

        # Registered reply function for checking_agent, checks the result of the response for agent names
        def validate_speaker_name(recipient, messages, sender, config) -> Tuple[bool, Union[str, Dict, None]]:

            # The number of retries left, starting at max_retries_for_selecting_speaker
            nonlocal attempts_left
            nonlocal attempt

            attempt = attempt + 1
            attempts_left = attempts_left - 1

            return self._validate_speaker_name(recipient, messages, sender, config, attempts_left, attempt, agents)

        # Two-agent chat for speaker selection

        # Agent for checking the response from the speaker_select_agent
        checking_agent = ConversableAgent("checking_agent", default_auto_reply=max_attempts)

        # Register the speaker validation function with the checking agent
        checking_agent.register_reply(
            [ConversableAgent, None],
            reply_func=validate_speaker_name,  # Validate each response
            remove_other_reply_funcs=True,
        )

        checking_agent.register_reply(
            [ConversableAgent, None],
            reply_func=self.validate_and_mitigate_selection_response,  # Validate each response
            position=0
        )

        # Agent for selecting a single agent name from the response
        speaker_selection_agent = ConversableAgent(
            "speaker_selection_agent",
            system_message=self.select_speaker_msg(agents),
            chat_messages={checking_agent: messages},
            llm_config=selector.llm_config,
            human_input_mode="NEVER",
            # Suppresses some extra terminal outputs, outputs will be handled by select_speaker_auto_verbose
        )
        if self.model_client_cls is not None:
            speaker_selection_agent.register_model_client(self.model_client_cls)

        # Run the speaker selection chat
        result = checking_agent.initiate_chat(
            speaker_selection_agent,
            cache=None,  # don't use caching for the speaker selection chat
            message={
                "content": self.select_speaker_prompt(agents),
                "override_role": self.role_for_select_speaker_messages,
            },
            max_turns=2
                      * max(1, max_attempts),  # Limiting the chat to the number of attempts, including the initial one
            clear_history=False,
            silent=not self.select_speaker_auto_verbose,  # Base silence on the verbose attribute
        )

        return self._process_speaker_selection_result(result, last_speaker, agents)

    async def a_auto_select_speaker(
            self,
            last_speaker: Agent,
            selector: ConversableAgent,
            messages: Optional[List[Dict]],
            agents: Optional[List[Agent]],
    ) -> Agent:
        """(Asynchronous) Selects next speaker for the "auto" speaker selection method. Utilises its own two-agent chat to determine the next speaker and supports requerying.

        Speaker selection for "auto" speaker selection method:
        1. Create a two-agent chat with a speaker selector agent and a speaker validator agent, like a nested chat
        2. Inject the group messages into the new chat
        3. Run the two-agent chat, evaluating the result of response from the speaker selector agent:
            - If a single agent is provided then we return it and finish. If not, we add an additional message to this nested chat in an attempt to guide the LLM to a single agent response
        4. Chat continues until a single agent is nominated or there are no more attempts left
        5. If we run out of turns and no single agent can be determined, the next speaker in the list of agents is returned

        Args:
            last_speaker Agent: The previous speaker in the group chat
            selector ConversableAgent:
            messages Optional[List[Dict]]: Current chat messages
            agents Optional[List[Agent]]: Valid list of agents for speaker selection

        Returns:
            Dict: a counter for mentioned agents.
        """

        # If no agents are passed in, assign all the group chat's agents
        if agents is None:
            agents = self.agents

        # The maximum number of speaker selection attempts (including requeries)
        # We track these and use them in the validation function as we can't
        # access the max_turns from within validate_speaker_name
        max_attempts = 1 + self.max_retries_for_selecting_speaker
        attempts_left = max_attempts
        attempt = 0

        # Registered reply function for checking_agent, checks the result of the response for agent names
        def validate_speaker_name(recipient, messages, sender, config) -> Tuple[bool, Union[str, Dict, None]]:
            # The number of retries left, starting at max_retries_for_selecting_speaker
            nonlocal attempts_left
            nonlocal attempt

            attempt = attempt + 1
            attempts_left = attempts_left - 1

            return self._validate_speaker_name(recipient, messages, sender, config, attempts_left, attempt, agents)

        # Two-agent chat for speaker selection

        # Agent for checking the response from the speaker_select_agent
        checking_agent = ConversableAgent("checking_agent", default_auto_reply=max_attempts)

        # Register the speaker validation function with the checking agent
        checking_agent.register_reply(
            [ConversableAgent, None],
            reply_func=validate_speaker_name,  # Validate each response
            remove_other_reply_funcs=True,
        )

        checking_agent.register_reply(
            [ConversableAgent, None],
            reply_func=self.a_validate_and_mitigate_selection_response,  # Validate each response
            position=0
        )

        # Agent for selecting a single agent name from the response
        speaker_selection_agent = ConversableAgent(
            "speaker_selection_agent",
            system_message=self.select_speaker_msg(agents),
            chat_messages={checking_agent: messages},
            llm_config=selector.llm_config,
            human_input_mode="NEVER",
            # Suppresses some extra terminal outputs, outputs will be handled by select_speaker_auto_verbose
        )
        if self.model_client_cls is not None:
            speaker_selection_agent.register_model_client(self.model_client_cls)

        # Run the speaker selection chat
        result = await checking_agent.a_initiate_chat(
            speaker_selection_agent,
            cache=None,  # don't use caching for the speaker selection chat
            message=self.select_speaker_prompt(agents),
            max_turns=2
                      * max(1, max_attempts),  # Limiting the chat to the number of attempts, including the initial one
            clear_history=False,
            silent=not self.select_speaker_auto_verbose,  # Base silence on the verbose attribute
        )

        return self._process_speaker_selection_result(result, last_speaker, agents)

    @staticmethod
    def validate_and_mitigate_selection_response(recipient: ConversableAgent, messages: Optional[List[Dict]] = None,
                                                 sender: Optional[Agent] = None,
                                                 config: Optional[Any] = None) -> Tuple[bool, Union[str, Dict, None]]:
        """
        Applies AutoAlign Verifier Agent on selection agent (the agent which selects the next speaker)
        This function modifies the chat history.
        """
        autoalign_verifier = AutoAlignVerifier()
        reply = autoalign_verifier.verify_and_mitigate_response(speaker=sender, reply=messages[-1],
                                                                messages=messages, sender=recipient)
        messages[-1] = reply
        return True, None

    @staticmethod
    async def a_validate_and_mitigate_selection_response(recipient: ConversableAgent,
                                                         messages: Optional[List[Dict]] = None,
                                                         sender: Optional[Agent] = None,
                                                         config: Optional[Any] = None) -> \
            Tuple[bool, Union[str, Dict, None]]:
        """
        Asynchronous method of above method.
        Applies AutoAlign Verifier Agent on selection agent (the agent which selects the next speaker)
        This function modifies the chat history.
        """
        autoalign_verifier = AutoAlignVerifier()
        reply = await autoalign_verifier.a_verify_and_mitigate_response(speaker=sender, reply=messages[-1],
                                                                        messages=messages, sender=recipient)
        messages[-1] = reply
        return True, None


def _mentioned_answer(message_content: Union[str, List]):
    if isinstance(message_content, dict):
        message_content = message_content["content"]
    message_content = content_str(message_content)

    mentions = dict()
    for answer in ['Yes', 'No']:
        # Finds answer mentions, taking word boundaries into account,
        # accommodates escaping underscores and underscores as spaces
        regex = (
                r"(?<=\W)("
                + re.escape(answer)
                + r"|"
                + re.escape(answer.replace("_", " "))
                + r"|"
                + re.escape(answer.replace("_", r"\_"))
                + r")(?=\W)"
        )
        count = len(re.findall(regex, f" {message_content} "))  # Pad the message to help with matching
        if count > 0:
            mentions[answer] = count
    return mentions


class AutoAlignGroupChatManager(GroupChatManager):
    """
    This is the chat manager which controls which agent is going to be called next and broadcasts its
    reply to all the agents. The design pattern it follows is the 'observer pattern'.
    """

    def __init__(self, groupchat: GroupChat, model_client_cls: Optional = None,
                 request_url: Optional[str] = None,
                 selected_guardrails: Optional[List[str]] = None, **kwargs):
        super().__init__(groupchat, **kwargs)
        # we will replace the existing reply functions with our functions that contain guardrail inference
        self.model_client_cls = model_client_cls
        self.request_url = request_url
        self.selected_guardrails = selected_guardrails
        self.replace_reply_func(GroupChatManager.run_chat, AutoAlignGroupChatManager.run_chat)
        self.replace_reply_func(GroupChatManager.a_run_chat, AutoAlignGroupChatManager.a_run_chat)

    def run_chat(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[GroupChat] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        This method is the controller function of AutoAlignGroupChatManager, it manages the entire chat
        """
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        speaker = sender
        groupchat = config
        send_introductions = getattr(groupchat, "send_introductions", False)

        if send_introductions:
            # Broadcast the intro
            # observer pattern: broadcasts the event to all observers
            intro = groupchat.introductions_msg()
            for agent in groupchat.agents:
                self.send(intro, agent, request_reply=False, silent=True)
            # NOTE: We do not also append to groupchat.messages,
            # since groupchat handles its own introductions

        if self.client_cache is not None:
            for a in groupchat.agents:
                a.previous_cache = a.client_cache
                a.client_cache = self.client_cache
        for i in range(groupchat.max_round):
            groupchat.append(message, speaker)
            # broadcast the message to all agents except the speaker
            # observer pattern: broadcasts the event to all observers
            for agent in groupchat.agents:
                if agent != speaker:
                    self.send(message, agent, request_reply=False, silent=True)
            if self._is_termination_msg(message) or i == groupchat.max_round - 1:
                # The conversation is over or it's the last round
                break
            try:
                # select the next speaker
                speaker = groupchat.select_speaker(speaker, self)
                if not isinstance(speaker, AutoALignAssistantAgent):
                    reply = speaker.generate_reply(sender=self)
                else:
                    autoalign_verifier = AutoAlignVerifier()
                    input_guardrail_flag = False

                    prompt = message['content'] if isinstance(message, dict) else message
                    if self.request_url and self.selected_guardrails and prompt:
                        input_guardrail_flag = autoalign_verifier.check_guardrails(
                            request_url=self.request_url,
                            text=prompt,
                            selected_guardrails=self.selected_guardrails)

                    if input_guardrail_flag:
                        print("----------------Policy violated-------------------")
                        reply = {"role": "assistant", "name": speaker.name,
                                 "content": "AutoAlign policy violated."}
                    else:
                        # let the speaker speak
                        reply = speaker.generate_reply(sender=self)
                        # AutoAlign Verification and Mitigation
                        # here 'speaker' refers to the agent, and we will apply the
                        # 'verify_and_mitigate_response' function just after we receive the reply from the speaker.
                        reply = autoalign_verifier.verify_and_mitigate_response(speaker, reply, messages, self)
                        output_guardrail_flag = False
                        if self.request_url and self.selected_guardrails:
                            output_guardrail_flag = autoalign_verifier.check_guardrails(
                                request_url=self.request_url,
                                text=reply['content'] if isinstance(reply, dict) else reply,
                                selected_guardrails=self.selected_guardrails)

                        if output_guardrail_flag:
                            print("----------------Policy violated-------------------")
                            reply = {"role": "assistant", "name": speaker.name,
                                     "content": "AutoAlign policy violated."}

            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = speaker.generate_reply(sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            except NoEligibleSpeaker:
                # No eligible speaker, terminate the conversation
                break

            if reply is None:
                # no reply is generated, exit the chat
                break

            # check for "clear history" phrase in reply and activate clear history function if found
            if (
                    groupchat.enable_clear_history
                    and isinstance(reply, dict)
                    and reply["content"]
                    and "CLEAR HISTORY" in reply["content"].upper()
            ):
                reply["content"] = self.clear_agents_history(reply, groupchat)

            # The speaker sends the message without requesting a reply
            speaker.send(reply, self, request_reply=False)
            message = self.last_message(speaker)
        if self.client_cache is not None:
            for a in groupchat.agents:
                a.client_cache = a.previous_cache
                a.previous_cache = None
        return True, None

    async def a_run_chat(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[GroupChat] = None,
    ):
        """
        This is an asynchronous implementation of above function.
        This method is the controller function of AutoAlignGroupChatManager, it manages the entire chat
        """
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        speaker = sender
        groupchat = config
        send_introductions = getattr(groupchat, "send_introductions", False)

        if send_introductions:
            # Broadcast the intro
            intro = groupchat.introductions_msg()
            for agent in groupchat.agents:
                await self.a_send(intro, agent, request_reply=False, silent=True)
            # NOTE: We do not also append to groupchat.messages,
            # since groupchat handles its own introductions

        if self.client_cache is not None:
            for a in groupchat.agents:
                a.previous_cache = a.client_cache
                a.client_cache = self.client_cache
        for i in range(groupchat.max_round):
            groupchat.append(message, speaker)

            if self._is_termination_msg(message):
                # The conversation is over
                break

            # broadcast the message to all agents except the speaker
            for agent in groupchat.agents:
                if agent != speaker:
                    await self.a_send(message, agent, request_reply=False, silent=True)
            if i == groupchat.max_round - 1:
                # the last round
                break
            try:
                # select the next speaker
                speaker = await groupchat.a_select_speaker(speaker, self)
                # let the speaker speak
                if not isinstance(speaker, AutoALignAssistantAgent):
                    reply = await speaker.a_generate_reply(sender=self)
                else:
                    autoalign_verifier = AutoAlignVerifier()
                    input_guardrail_flag = False

                    prompt = message['content'] if isinstance(message, dict) else message
                    if self.request_url and self.selected_guardrails and prompt:
                        input_guardrail_flag = await autoalign_verifier.a_check_guardrails(
                            request_url=self.request_url,
                            text=prompt,
                            selected_guardrails=self.selected_guardrails)
                    if input_guardrail_flag:
                        reply = {"role": "assistant", "name": speaker.name,
                                 "content": "AutoAlign policy violated."}
                    else:
                        # let the speaker speak
                        reply = await speaker.a_generate_reply(sender=self)
                        # AutoAlign Verification and Mitigation
                        # here 'speaker' refers to the agent, and we will apply the
                        # 'verify_and_mitigate_response' function just after we receive the reply from the speaker.
                        reply = await autoalign_verifier.a_verify_and_mitigate_response(speaker, reply, messages, self)
                        output_guardrail_flag = False
                        if self.request_url and self.selected_guardrails:
                            output_guardrail_flag = await autoalign_verifier.a_check_guardrails(
                                request_url=self.request_url,
                                text=reply['content'] if isinstance(reply, dict) else reply,
                                selected_guardrails=self.selected_guardrails)

                        if output_guardrail_flag:
                            reply = {"role": "assistant", "name": speaker.name,
                                     "content": "AutoAlign policy violated."}

            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = await speaker.a_generate_reply(sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            if reply is None:
                break
            # The speaker sends the message without requesting a reply
            await speaker.a_send(reply, self, request_reply=False)
            message = self.last_message(speaker)
        if self.client_cache is not None:
            for a in groupchat.agents:
                a.client_cache = a.previous_cache
                a.previous_cache = None
        return True, None
