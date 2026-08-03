from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

from agent_lab import AgentUtils, WorkflowAgentBase, discoverable_agent
from agent_lab.interface.api.messages.schema import MessageRequest


class ReactAgentState(MessagesState):
    agent_id: str
    schema: str
    query: str
    generation: str


@discoverable_agent("cosimo_react")
class ReactAgent(WorkflowAgentBase):
    """Single-node ReAct workflow.

    The node runs a ReAct agent over the query and the persisted
    chain-of-thought history; ``messages`` in the checkpointed state stores
    one human/AI pair per turn (the AI side holds a ``create_thought_chain``
    summary), so past turns feed the next invocation as conversation memory
    while keeping strict role alternation for providers that require it.
    """

    # The fine-tuned Cosimo checkpoint is trained at model.max_seq_length 8192
    # (jobs/fine-tune/configs/base.yaml), of which the persona alone spends
    # ~645 tokens before any tool schema or conversation turn. A ReAct loop adds
    # an assistant tool call and a tool result per iteration on top of whatever
    # history is replayed here, so the replayed history is bounded rather than
    # unbounded. Twelve messages is six human/AI turns.
    MAX_HISTORY_MESSAGES = 12

    def __init__(self, agent_utils: AgentUtils):
        super().__init__(agent_utils)

    def get_react_tools(self) -> list:
        """Tools bound to the ReAct agent.

        Empty by default, because this project has no tool surface yet:
        ``cosimo/mcp/tools.py`` still ships the placeholder ``cosimo_hello_tool``.
        Override or extend this once real tools exist -- everything downstream is
        already wired for them:

        * the served checkpoint's chat template renders ``tools``,
          ``tool_calls`` and tool results (jobs/fine-tune/configs/chat_template.jinja);
        * the model is trained on the matching ``<tool_call>`` format
          (jobs/fine-tune/scripts/02_prepare_tool_data.py);
        * vLLM must be started with ``--tool-call-parser hermes`` to turn that
          format back into OpenAI ``tool_calls`` (see jobs/fine-tune/README.md).

        Inherited helpers are available for the common cases: ``get_bash_tool()``
        here, and the web tools on ``WebAgentBase`` if this agent is moved onto
        that base (note it requires a ``cdp_url`` entry in the app config).
        """
        return []

    def create_default_settings(self, agent_id: str, schema: str):
        current_dir = Path(__file__).parent
        prompt = self.read_file_content(
            f"{current_dir}/default_execution_system_prompt.txt"
        )
        self.agent_setting_service.create_agent_setting(
            agent_id=agent_id,
            setting_key="execution_system_prompt",
            setting_value=prompt,
            schema=schema,
        )

    def get_workflow_builder(self, agent_id: str):
        workflow_builder = StateGraph(ReactAgentState)
        workflow_builder.add_node("react", self.react)
        workflow_builder.add_edge(START, "react")
        workflow_builder.add_edge("react", END)
        return workflow_builder

    def react(self, state: ReactAgentState):
        agent_id = state["agent_id"]
        schema = state["schema"]
        query = state["query"]

        settings = self.agent_setting_service.get_agent_settings(agent_id, schema)
        settings_dict = {
            setting.setting_key: setting.setting_value for setting in settings
        }
        execution_system_prompt = self.parse_prompt_template(
            settings_dict,
            "execution_system_prompt",
            {"CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z")},
        )

        # No inner checkpointer: the outer WorkflowAgentBase graph owns
        # persistence for this thread. Prior turns in state ``messages``
        # provide the conversation memory.
        react_agent = create_react_agent(
            model=self.get_chat_model(agent_id, schema),
            tools=self.get_react_tools(),
            prompt=execution_system_prompt,
        )
        human_message = HumanMessage(content=self.QUERY_FORMAT.format(query=query))
        # Only the most recent turns are replayed: the tuned checkpoint has a
        # bounded trained context and a tool-using loop consumes it from the
        # other end. Older turns stay in the checkpointed state, they are just
        # not re-sent.
        history = state["messages"][-self.MAX_HISTORY_MESSAGES :]
        response = react_agent.invoke({"messages": [*history, human_message]})
        generation = response["messages"][-1].content

        # Persist an explicit human/AI pair: a bare string would be coerced
        # to a HumanMessage by the messages reducer, breaking role
        # alternation on the next turn.
        return {
            "generation": generation,
            "messages": [
                human_message,
                AIMessage(
                    content=self.create_thought_chain(
                        human_input=query, ai_response=generation
                    )
                ),
            ],
        }

    def get_input_params(self, message_request: MessageRequest, schema: str) -> dict:
        return {
            "agent_id": message_request.agent_id,
            "schema": schema,
            "query": message_request.message_content,
        }

    def format_response(self, workflow_state: ReactAgentState) -> (str, dict):
        return workflow_state.get("generation"), {
            "agent_id": workflow_state.get("agent_id"),
            "query": workflow_state.get("query"),
            "generation": workflow_state.get("generation"),
            "thought_chain": [
                message.content for message in workflow_state.get("messages", [])
            ],
        }
