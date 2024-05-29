from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
try:
    import chromadb
except ImportError:
    raise ImportError("Please install dependencies first. `pip install pyautogen[retrievechat]`")
from autogen.agentchat.contrib.vectordb.utils import (
    get_logger
)
from typing import  Callable, Dict, Optional, Union
from autoalign_group_chat import AutoAlignVerifier
from autogen.agentchat.contrib.vectordb.base import Document, QueryResults, VectorDB, VectorDBFactory
from autogen.token_count_utils import count_token
from autogen.retrieve_utils import (
    TEXT_FORMATS,
    create_vector_db_from_dir,
    get_files_from_dir,
    query_vector_db,
    split_files_to_chunks,
)
from IPython import get_ipython
from autogen.runtime_logging import log_event, logging_enabled
from autogen.formatting_utils import colored



# from .._pydantic import model_dump
# from ..code_utils import (
#     PYTHON_VARIANTS,
#     UNKNOWN,
#     check_can_use_docker_or_throw,
#     content_str,
#     decide_use_docker,
#     execute_code,
#     extract_code,
#     infer_lang,
# )
# from ..coding.base import CodeExecutor
# from ..coding.factory import CodeExecutorFactory
# from ..formatting_utils import colored
# from ..function_utils import get_function_schema, load_basemodels_if_needed, serialize_to_str
# from ..io.base import IOStream
# from ..oai.client import ModelClient, OpenAIWrapper
# from ..runtime_logging import log_event, log_new_agent, logging_enabled
# from .agent import Agent, LLMAgent
# from .chat import ChatResult, a_initiate_chats, initiate_chats
from autogen.agentchat.chat import ChatResult
from autogen.agentchat.utils import consolidate_chat_info, gather_usage_summary
from autogen.cache.cache import AbstractCache
from autogen.agentchat.agent import Agent

DEFAULT_SUMMARY_METHOD = "last_msg"

logger = get_logger(__name__)


class AutoAlignRetrieveUserProxyAgent(RetrieveUserProxyAgent):
    def __init__(
        self,
        name="RetrieveChatAgent",  # default set to RetrieveChatAgent
        human_input_mode: Optional[str] = "ALWAYS",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        retrieve_config: Optional[Dict] = None,  # config for the retrieve agent
        **kwargs,
    ):
        r"""
        Args:
            name (str): name of the agent.

            human_input_mode (str): whether to ask for human inputs every time a message is received.
                Possible values are "ALWAYS", "TERMINATE", "NEVER".
                1. When "ALWAYS", the agent prompts for human input every time a message is received.
                    Under this mode, the conversation stops when the human input is "exit",
                    or when is_termination_msg is True and there is no human input.
                2. When "TERMINATE", the agent only prompts for human input only when a termination
                    message is received or the number of auto reply reaches
                    the max_consecutive_auto_reply.
                3. When "NEVER", the agent will never prompt for human input. Under this mode, the
                    conversation stops when the number of auto reply reaches the
                    max_consecutive_auto_reply or when is_termination_msg is True.

            is_termination_msg (function): a function that takes a message in the form of a dictionary
                and returns a boolean value indicating if this received message is a termination message.
                The dict can contain the following keys: "content", "role", "name", "function_call".

            retrieve_config (dict or None): config for the retrieve agent.

                To use default config, set to None. Otherwise, set to a dictionary with the
                following keys:
                - `task` (Optional, str) - the task of the retrieve chat. Possible values are
                    "code", "qa" and "default". System prompt will be different for different tasks.
                     The default value is `default`, which supports both code and qa, and provides
                     source information in the end of the response.
                - `vector_db` (Optional, Union[str, VectorDB]) - the vector db for the retrieve chat.
                    If it's a string, it should be the type of the vector db, such as "chroma"; otherwise,
                    it should be an instance of the VectorDB protocol. Default is "chroma".
                    Set `None` to use the deprecated `client`.
                - `db_config` (Optional, Dict) - the config for the vector db. Default is `{}`. Please make
                    sure you understand the config for the vector db you are using, otherwise, leave it as `{}`.
                    Only valid when `vector_db` is a string.
                - `client` (Optional, chromadb.Client) - the chromadb client. If key not provided, a
                     default client `chromadb.Client()` will be used. If you want to use other
                     vector db, extend this class and override the `retrieve_docs` function.
                     **Deprecated**: use `vector_db` instead.
                - `docs_path` (Optional, Union[str, List[str]]) - the path to the docs directory. It
                     can also be the path to a single file, the url to a single file or a list
                     of directories, files and urls. Default is None, which works only if the
                     collection is already created.
                - `extra_docs` (Optional, bool) - when true, allows adding documents with unique IDs
                    without overwriting existing ones; when false, it replaces existing documents
                    using default IDs, risking collection overwrite., when set to true it enables
                    the system to assign unique IDs starting from "length+i" for new document
                    chunks, preventing the replacement of existing documents and facilitating the
                    addition of more content to the collection..
                    By default, "extra_docs" is set to false, starting document IDs from zero.
                    This poses a risk as new documents might overwrite existing ones, potentially
                    causing unintended loss or alteration of data in the collection.
                    **Deprecated**: use `new_docs` when use `vector_db` instead of `client`.
                - `new_docs` (Optional, bool) - when True, only adds new documents to the collection;
                    when False, updates existing documents and adds new ones. Default is True.
                    Document id is used to determine if a document is new or existing. By default, the
                    id is the hash value of the content.
                - `model` (Optional, str) - the model to use for the retrieve chat.
                    If key not provided, a default model `gpt-4` will be used.
                - `chunk_token_size` (Optional, int) - the chunk token size for the retrieve chat.
                    If key not provided, a default size `max_tokens * 0.4` will be used.
                - `context_max_tokens` (Optional, int) - the context max token size for the
                    retrieve chat.
                    If key not provided, a default size `max_tokens * 0.8` will be used.
                - `chunk_mode` (Optional, str) - the chunk mode for the retrieve chat. Possible values
                    are "multi_lines" and "one_line". If key not provided, a default mode
                    `multi_lines` will be used.
                - `must_break_at_empty_line` (Optional, bool) - chunk will only break at empty line
                    if True. Default is True.
                    If chunk_mode is "one_line", this parameter will be ignored.
                - `embedding_model` (Optional, str) - the embedding model to use for the retrieve chat.
                    If key not provided, a default model `all-MiniLM-L6-v2` will be used. All available
                    models can be found at `https://www.sbert.net/docs/pretrained_models.html`.
                    The default model is a fast model. If you want to use a high performance model,
                    `all-mpnet-base-v2` is recommended.
                    **Deprecated**: no need when use `vector_db` instead of `client`.
                - `embedding_function` (Optional, Callable) - the embedding function for creating the
                    vector db. Default is None, SentenceTransformer with the given `embedding_model`
                    will be used. If you want to use OpenAI, Cohere, HuggingFace or other embedding
                    functions, you can pass it here,
                    follow the examples in `https://docs.trychroma.com/embeddings`.
                - `customized_prompt` (Optional, str) - the customized prompt for the retrieve chat.
                    Default is None.
                - `customized_answer_prefix` (Optional, str) - the customized answer prefix for the
                    retrieve chat. Default is "".
                    If not "" and the customized_answer_prefix is not in the answer,
                    `Update Context` will be triggered.
                - `update_context` (Optional, bool) - if False, will not apply `Update Context` for
                    interactive retrieval. Default is True.
                - `collection_name` (Optional, str) - the name of the collection.
                    If key not provided, a default name `autogen-docs` will be used.
                - `get_or_create` (Optional, bool) - Whether to get the collection if it exists. Default is True.
                - `overwrite` (Optional, bool) - Whether to overwrite the collection if it exists. Default is False.
                    Case 1. if the collection does not exist, create the collection.
                    Case 2. the collection exists, if overwrite is True, it will overwrite the collection.
                    Case 3. the collection exists and overwrite is False, if get_or_create is True, it will get the collection,
                        otherwise it raise a ValueError.
                - `custom_token_count_function` (Optional, Callable) - a custom function to count the
                    number of tokens in a string.
                    The function should take (text:str, model:str) as input and return the
                    token_count(int). the retrieve_config["model"] will be passed in the function.
                    Default is autogen.token_count_utils.count_token that uses tiktoken, which may
                    not be accurate for non-OpenAI models.
                - `custom_text_split_function` (Optional, Callable) - a custom function to split a
                    string into a list of strings.
                    Default is None, will use the default function in
                    `autogen.retrieve_utils.split_text_to_chunks`.
                - `custom_text_types` (Optional, List[str]) - a list of file types to be processed.
                    Default is `autogen.retrieve_utils.TEXT_FORMATS`.
                    This only applies to files under the directories in `docs_path`. Explicitly
                    included files and urls will be chunked regardless of their types.
                - `recursive` (Optional, bool) - whether to search documents recursively in the
                    docs_path. Default is True.
                - `distance_threshold` (Optional, float) - the threshold for the distance score, only
                    distance smaller than it will be returned. Will be ignored if < 0. Default is -1.

            `**kwargs` (dict): other kwargs in [UserProxyAgent](../user_proxy_agent#__init__).

        Example:

        Example of overriding retrieve_docs - If you have set up a customized vector db, and it's
        not compatible with chromadb, you can easily plug in it with below code.
        **Deprecated**: Use `vector_db` instead. You can extend VectorDB and pass it to the agent.
        ```python
        class MyRetrieveUserProxyAgent(RetrieveUserProxyAgent):
            def query_vector_db(
                self,
                query_texts: List[str],
                n_results: int = 10,
                search_string: str = "",
                **kwargs,
            ) -> Dict[str, Union[List[str], List[List[str]]]]:
                # define your own query function here
                pass

            def retrieve_docs(self, problem: str, n_results: int = 20, search_string: str = "", **kwargs):
                results = self.query_vector_db(
                    query_texts=[problem],
                    n_results=n_results,
                    search_string=search_string,
                    **kwargs,
                )

                self._results = results
                print("doc_ids: ", results["ids"])
        ```
        """
        super().__init__(
            name=name,
            human_input_mode=human_input_mode,
            **kwargs,
        )

        self._retrieve_config = {} if retrieve_config is None else retrieve_config
        self._task = self._retrieve_config.get("task","default")
        self._vector_db = self._retrieve_config.get("vector_db","chroma")
        self._db_config = self._retrieve_config.get("db_config",{})
        self._client = self._retrieve_config.get("client",chromadb.Client())
        self._docs_path = self._retrieve_config.get("docs_path",None)
        self._extra_docs = self._retrieve_config.get("extra_docs",False)
        self._new_docs = self._retrieve_config.get("new_docs",True)
        self._collection_name = self._retrieve_config.get("collection_name","autogen-docs")
        if "docs_path" not in self._retrieve_config:
            logger.warning(
                "docs_path is not provided in retrieve_config. "
                f"Will raise ValueError if the collection `{self._collection_name}` doesn't exist. "
                "Set docs_path to None to suppress this warning."
            )
        self._model = self._retrieve_config.get("model","gpt-4")
        self._max_tokens = self.get_max_tokens(self._model)
        self._chunk_token_size = int(self._retrieve_config.get("chunk_token_size",self._max_tokens * 0.4))
        self._chunk_mode = self._retrieve_config.get("chunk_mode","multi_lines")
        self._must_break_at_empty_line = self._retrieve_config.get("must_break_at_empty_line",True)
        self._embedding_model = self._retrieve_config.get("embedding_model","all-MiniLM-L6-v2")
        self._embedding_function = self._retrieve_config.get("embedding_function",None)
        self.customized_prompt = self._retrieve_config.get("customized_prompt",None)
        self.customized_answer_prefix = self._retrieve_config.get("customized_answer_prefix","").upper()
        self.update_context = self._retrieve_config.get("update_context",True)
        self._get_or_create = self._retrieve_config.get("get_or_create",False) if self._docs_path is not None else True
        self._overwrite = self._retrieve_config.get("overwrite",False)
        self.custom_token_count_function = self._retrieve_config.get("custom_token_count_function",count_token)
        self.custom_text_split_function = self._retrieve_config.get("custom_text_split_function",None)
        self._custom_text_types = self._retrieve_config.get("custom_text_types",TEXT_FORMATS)
        self._recursive = self._retrieve_config.get("recursive",True)
        self._context_max_tokens = self._retrieve_config.get("context_max_tokens",self._max_tokens * 0.8)
        self._collection = True if self._docs_path is None else False  # whether the collection is created
        self._ipython = get_ipython()
        self._doc_idx = -1  # the index of the current used doc
        self._results = []  # the results of the current query
        self._intermediate_answers = set()  # the intermediate answers
        self._doc_contents = []  # the contents of the current used doc
        self._doc_ids = []  # the ids of the current used doc
        self._current_docs_in_context = []  # the ids of the current context sources
        self._search_string = ""  # the search string used in the current query
        self._distance_threshold = self._retrieve_config.get("distance_threshold",-1)
        # update the termination message function
        self._is_termination_msg = (
            self._is_termination_msg_retrievechat if is_termination_msg is None else is_termination_msg
        )
        if isinstance(self._vector_db,str):
            if not isinstance(self._db_config,dict):
                raise ValueError("`db_config` should be a dictionary.")
            if "embedding_function" in self._retrieve_config:
                self._db_config["embedding_function"] = self._embedding_function
            self._vector_db = VectorDBFactory.create_vector_db(db_type=self._vector_db,**self._db_config)

    def initiate_chat(
            self,
            recipient: "ConversableAgent",
            clear_history: bool = True,
            silent: Optional[bool] = False,
            cache: Optional[AbstractCache] = None,
            max_turns: Optional[int] = None,
            summary_method: Optional[Union[str,Callable]] = DEFAULT_SUMMARY_METHOD,
            summary_args: Optional[dict] = {},
            message: Optional[Union[Dict,str,Callable]] = None,
            **kwargs,
    ) -> ChatResult:
        """Initiate a chat with the recipient agent.

        Reset the consecutive auto reply counter.
        If `clear_history` is True, the chat history with the recipient agent will be cleared.


        Args:
            recipient: the recipient agent.
            clear_history (bool): whether to clear the chat history with the agent. Default is True.
            silent (bool or None): (Experimental) whether to print the messages for this conversation. Default is False.
            cache (AbstractCache or None): the cache client to be used for this conversation. Default is None.
            max_turns (int or None): the maximum number of turns for the chat between the two agents. One turn means one conversation round trip. Note that this is different from
                [max_consecutive_auto_reply](#max_consecutive_auto_reply) which is the maximum number of consecutive auto replies; and it is also different from [max_rounds in GroupChat](./groupchat#groupchat-objects) which is the maximum number of rounds in a group chat session.
                If max_turns is set to None, the chat will continue until a termination condition is met. Default is None.
            summary_method (str or callable): a method to get a summary from the chat. Default is DEFAULT_SUMMARY_METHOD, i.e., "last_msg".

            Supported strings are "last_msg" and "reflection_with_llm":
                - when set to "last_msg", it returns the last message of the dialog as the summary.
                - when set to "reflection_with_llm", it returns a summary extracted using an llm client.
                    `llm_config` must be set in either the recipient or sender.

            A callable summary_method should take the recipient and sender agent in a chat as input and return a string of summary. E.g.,

            ```python
            def my_summary_method(
                sender: ConversableAgent,
                recipient: ConversableAgent,
                summary_args: dict,
            ):
                return recipient.last_message(sender)["content"]
            ```
            summary_args (dict): a dictionary of arguments to be passed to the summary_method.
                One example key is "summary_prompt", and value is a string of text used to prompt a LLM-based agent (the sender or receiver agent) to reflect
                on the conversation and extract a summary when summary_method is "reflection_with_llm".
                The default summary_prompt is DEFAULT_SUMMARY_PROMPT, i.e., "Summarize takeaway from the conversation. Do not add any introductory phrases. If the intended request is NOT properly addressed, please point it out."
                Another available key is "summary_role", which is the role of the message sent to the agent in charge of summarizing. Default is "system".
            message (str, dict or Callable): the initial message to be sent to the recipient. Needs to be provided. Otherwise, input() will be called to get the initial message.
                - If a string or a dict is provided, it will be used as the initial message.        `generate_init_message` is called to generate the initial message for the agent based on this string and the context.
                    If dict, it may contain the following reserved fields (either content or tool_calls need to be provided).

                        1. "content": content of the message, can be None.
                        2. "function_call": a dictionary containing the function name and arguments. (deprecated in favor of "tool_calls")
                        3. "tool_calls": a list of dictionaries containing the function name and arguments.
                        4. "role": role of the message, can be "assistant", "user", "function".
                            This field is only needed to distinguish between "function" or "assistant"/"user".
                        5. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
                        6. "context" (dict): the context of the message, which will be passed to
                            [OpenAIWrapper.create](../oai/client#create).

                - If a callable is provided, it will be called to get the initial message in the form of a string or a dict.
                    If the returned type is dict, it may contain the reserved fields mentioned above.

                    Example of a callable message (returning a string):

            ```python
            def my_message(sender: ConversableAgent, recipient: ConversableAgent, context: dict) -> Union[str, Dict]:
                carryover = context.get("carryover", "")
                if isinstance(message, list):
                    carryover = carryover[-1]
                final_msg = "Write a blogpost." + "\\nContext: \\n" + carryover
                return final_msg
            ```

                    Example of a callable message (returning a dict):

            ```python
            def my_message(sender: ConversableAgent, recipient: ConversableAgent, context: dict) -> Union[str, Dict]:
                final_msg = {}
                carryover = context.get("carryover", "")
                if isinstance(message, list):
                    carryover = carryover[-1]
                final_msg["content"] = "Write a blogpost." + "\\nContext: \\n" + carryover
                final_msg["context"] = {"prefix": "Today I feel"}
                return final_msg
            ```
            **kwargs: any additional information. It has the following reserved fields:
                - "carryover": a string or a list of string to specify the carryover information to be passed to this chat.
                    If provided, we will combine this carryover (by attaching a "context: " string and the carryover content after the message content) with the "message" content when generating the initial chat
                    message in `generate_init_message`.
                - "verbose": a boolean to specify whether to print the message and carryover in a chat. Default is False.

        Raises:
            RuntimeError: if any async reply functions are registered and not ignored in sync chat.

        Returns:
            ChatResult: an ChatResult object.
        """
        _chat_info = locals().copy()
        _chat_info["sender"] = self
        consolidate_chat_info(_chat_info,uniform_sender=self)
        for agent in [self,recipient]:
            agent._raise_exception_on_async_reply_functions()
            agent.previous_cache = agent.client_cache
            agent.client_cache = cache

        problem = kwargs.get("problem", None)

        if isinstance(max_turns,int):
            self._prepare_chat(recipient,clear_history,reply_at_receive=False)
            for _ in range(max_turns):
                if _ == 0:
                    if isinstance(message,Callable):
                        msg2send = message(_chat_info["sender"],_chat_info["recipient"],kwargs)
                    else:
                        msg2send = self.generate_init_message(message,**kwargs)
                else:
                    msg2send = self.generate_reply(messages=self.chat_messages[recipient],sender=recipient)
                if msg2send is None:
                    break

                msg2send = {"content":msg2send, "problem": problem}
                self.send(msg2send,recipient,request_reply=True,silent=silent)
        else:
            self._prepare_chat(recipient,clear_history)
            if isinstance(message,Callable):
                msg2send = message(_chat_info["sender"],_chat_info["recipient"],kwargs)
            else:
                msg2send = self.generate_init_message(message,**kwargs)

            msg2send = {"content": msg2send[0], "problem": problem, "docs":msg2send[1]}
            self.send(msg2send,recipient,silent=silent)
        summary = self._summarize_chat(
            summary_method,
            summary_args,
            recipient,
            cache=cache,
        )
        for agent in [self,recipient]:
            agent.client_cache = agent.previous_cache
            agent.previous_cache = None
        chat_result = ChatResult(
            chat_history=self.chat_messages[recipient],
            summary=summary,
            cost=gather_usage_summary([self,recipient]),
            human_input=self._human_input,
        )
        return chat_result

    async def a_initiate_chat(
            self,
            recipient: "ConversableAgent",
            clear_history: bool = True,
            silent: Optional[bool] = False,
            cache: Optional[AbstractCache] = None,
            max_turns: Optional[int] = None,
            summary_method: Optional[Union[str,Callable]] = DEFAULT_SUMMARY_METHOD,
            summary_args: Optional[dict] = {},
            message: Optional[Union[str,Callable]] = None,
            **kwargs,
    ) -> ChatResult:
        """(async) Initiate a chat with the recipient agent.

        Reset the consecutive auto reply counter.
        If `clear_history` is True, the chat history with the recipient agent will be cleared.
        `a_generate_init_message` is called to generate the initial message for the agent.

        Args: Please refer to `initiate_chat`.

        Returns:
            ChatResult: an ChatResult object.
        """
        _chat_info = locals().copy()
        _chat_info["sender"] = self
        consolidate_chat_info(_chat_info,uniform_sender=self)
        for agent in [self,recipient]:
            agent.previous_cache = agent.client_cache
            agent.client_cache = cache
        if isinstance(max_turns,int):
            self._prepare_chat(recipient,clear_history,reply_at_receive=False)
            for _ in range(max_turns):
                if _ == 0:
                    if isinstance(message,Callable):
                        msg2send = message(_chat_info["sender"],_chat_info["recipient"],kwargs)
                    else:
                        msg2send = await self.a_generate_init_message(message,**kwargs)
                else:
                    msg2send = await self.a_generate_reply(messages=self.chat_messages[recipient],sender=recipient)
                if msg2send is None:
                    break
                await self.a_send(msg2send,recipient,request_reply=True,silent=silent)
        else:
            self._prepare_chat(recipient,clear_history)
            if isinstance(message,Callable):
                msg2send = message(_chat_info["sender"],_chat_info["recipient"],kwargs)
            else:
                msg2send = await self.a_generate_init_message(message,**kwargs)
            await self.a_send(msg2send,recipient,silent=silent)
        summary = self._summarize_chat(
            summary_method,
            summary_args,
            recipient,
            cache=cache,
        )
        for agent in [self,recipient]:
            agent.client_cache = agent.previous_cache
            agent.previous_cache = None
        chat_result = ChatResult(
            chat_history=self.chat_messages[recipient],
            summary=summary,
            cost=gather_usage_summary([self,recipient]),
            human_input=self._human_input,
        )
        return chat_result

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        """Send a message to another agent.

        Args:
            message (dict or str): message to be sent.
                The message could contain the following fields:
                - content (str or List): Required, the content of the message. (Can be None)
                - function_call (str): the name of the function to be called.
                - name (str): the name of the function to be called.
                - role (str): the role of the message, any role that is not "function"
                    will be modified to "assistant".
                - context (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](../oai/client#create).
                    For example, one agent can send a message A as:
        ```python
        {
            "content": lambda context: context["use_tool_msg"],
            "context": {
                "use_tool_msg": "Use tool X if they are relevant."
            }
        }
        ```
                    Next time, one agent can send a message B with a different "use_tool_msg".
                    Then the content of message A will be refreshed to the new "use_tool_msg".
                    So effectively, this provides a way for an agent to send a "link" and modify
                    the content of the "link" later.
            recipient (Agent): the recipient of the message.
            request_reply (bool or None): whether to request a reply from the recipient.
            silent (bool or None): (Experimental) whether to print the message sent.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        message = self._process_message_before_send(message, recipient, silent)
        # When the agent composes and sends the message, the role of the message is "assistant"
        # unless it's "function".
        valid = self._append_oai_message(message, "assistant", recipient)
        if valid:
            recipient.receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    async def a_send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        """(async) Send a message to another agent.

        Args:
            message (dict or str): message to be sent.
                The message could contain the following fields:
                - content (str or List): Required, the content of the message. (Can be None)
                - function_call (str): the name of the function to be called.
                - name (str): the name of the function to be called.
                - role (str): the role of the message, any role that is not "function"
                    will be modified to "assistant".
                - context (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](../oai/client#create).
                    For example, one agent can send a message A as:
        ```python
        {
            "content": lambda context: context["use_tool_msg"],
            "context": {
                "use_tool_msg": "Use tool X if they are relevant."
            }
        }
        ```
                    Next time, one agent can send a message B with a different "use_tool_msg".
                    Then the content of message A will be refreshed to the new "use_tool_msg".
                    So effectively, this provides a way for an agent to send a "link" and modify
                    the content of the "link" later.
            recipient (Agent): the recipient of the message.
            request_reply (bool or None): whether to request a reply from the recipient.
            silent (bool or None): (Experimental) whether to print the message sent.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        message = self._process_message_before_send(message, recipient, silent)
        # When the agent composes and sends the message, the role of the message is "assistant"
        # unless it's "function".
        valid = self._append_oai_message(message, "assistant", recipient)
        if valid:
            await recipient.a_receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )


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

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        """Receive a message from another agent.

        Once a message is received, this function sends a reply to the sender or stop.
        The reply can be generated automatically or entered manually by a human.

        Args:
            message (dict or str): message from the sender. If the type is dict, it may contain the following reserved fields (either content or function_call need to be provided).
                1. "content": content of the message, can be None.
                2. "function_call": a dictionary containing the function name and arguments. (deprecated in favor of "tool_calls")
                3. "tool_calls": a list of dictionaries containing the function name and arguments.
                4. "role": role of the message, can be "assistant", "user", "function", "tool".
                    This field is only needed to distinguish between "function" or "assistant"/"user".
                5. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
                6. "context" (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](../oai/client#create).
            sender: sender of an Agent instance.
            request_reply (bool or None): whether a reply is requested from the sender.
                If None, the value is determined by `self.reply_at_receive[sender]`.
            silent (bool or None): (Experimental) whether to print the message received.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        self._process_received_message(message, sender, silent)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            self.send(reply, sender, silent=silent)

    async def a_receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        """(async) Receive a message from another agent.

        Once a message is received, this function sends a reply to the sender or stop.
        The reply can be generated automatically or entered manually by a human.

        Args:
            message (dict or str): message from the sender. If the type is dict, it may contain the following reserved fields (either content or function_call need to be provided).
                1. "content": content of the message, can be None.
                2. "function_call": a dictionary containing the function name and arguments. (deprecated in favor of "tool_calls")
                3. "tool_calls": a list of dictionaries containing the function name and arguments.
                4. "role": role of the message, can be "assistant", "user", "function".
                    This field is only needed to distinguish between "function" or "assistant"/"user".
                5. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
                6. "context" (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](../oai/client#create).
            sender: sender of an Agent instance.
            request_reply (bool or None): whether a reply is requested from the sender.
                If None, the value is determined by `self.reply_at_receive[sender]`.
            silent (bool or None): (Experimental) whether to print the message received.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        self._process_received_message(message, sender, silent)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        reply = await self.a_generate_reply(sender=sender)
        if reply is not None:
            await self.a_send(reply, sender, silent=silent)


    def _process_received_message(self, message: Union[Dict, str], sender: Agent, silent: bool):
        # When the agent receives a message, the role of the message is "user". (If 'role' exists and is 'function', it will remain unchanged.)
        valid = self._append_oai_message(message, "user", sender)
        if logging_enabled():
            log_event(self, "received_message", message=message, sender=sender.name, valid=valid)

        if not valid:
            raise ValueError(
                "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )
        if not silent:
            self._print_received_message(message, sender)

    def _get_context(self, results: QueryResults):
        docs = []
        doc_contents = ""
        self._current_docs_in_context = []
        current_tokens = 0
        _doc_idx = self._doc_idx
        _tmp_retrieve_count = 0
        for idx, doc in enumerate(results[0]):
            doc = doc[0]
            if idx <= _doc_idx:
                continue
            if doc["id"] in self._doc_ids:
                continue
            _doc_tokens = self.custom_token_count_function(doc["content"], self._model)
            if _doc_tokens > self._context_max_tokens:
                func_print = f"Skip doc_id {doc['id']} as it is too long to fit in the context."
                print(colored(func_print, "green"), flush=True)
                self._doc_idx = idx
                continue
            if current_tokens + _doc_tokens > self._context_max_tokens:
                break
            func_print = f"Adding content of doc {doc['id']} to context."
            print(colored(func_print, "green"), flush=True)
            current_tokens += _doc_tokens
            doc_contents += doc["content"] + "\n"
            docs.append(doc)
            _metadata = doc.get("metadata")
            if isinstance(_metadata, dict):
                self._current_docs_in_context.append(_metadata.get("source", ""))
            self._doc_idx = idx
            self._doc_ids.append(doc["id"])
            self._doc_contents.append(doc["content"])
            _tmp_retrieve_count += 1
            if _tmp_retrieve_count >= self.n_results:
                break
        return doc_contents, docs

    @staticmethod
    def message_generator(sender, recipient, context):
        """
        Generate an initial message with the given context for the RetrieveUserProxyAgent.
        Args:
            sender (Agent): the sender agent. It should be the instance of RetrieveUserProxyAgent.
            recipient (Agent): the recipient agent. Usually it's the assistant agent.
            context (dict): the context for the message generation. It should contain the following keys:
                - `problem` (str) - the problem to be solved.
                - `n_results` (int) - the number of results to be retrieved. Default is 20.
                - `search_string` (str) - only docs that contain an exact match of this string will be retrieved. Default is "".
        Returns:
            str: the generated message ready to be sent to the recipient agent.
        """
        sender._reset()

        problem = context.get("problem", "")
        n_results = context.get("n_results", 20)
        search_string = context.get("search_string", "")

        sender.retrieve_docs(problem, n_results, search_string)
        sender.problem = problem
        sender.n_results = n_results
        doc_contents, docs  = sender._get_context(sender._results)
        message = sender._generate_message(doc_contents, sender._task)
        return message, docs
