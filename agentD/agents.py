
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from langchain.docstore.in_memory import InMemoryDocstore
from agentD.prompts.base_template import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX
import faiss
from agentD.providers.deepseek import get_chat_model
from agentD.local_embeddings import get_embeddings
from agentD.runtime_config import configure_runtime
import warnings
warnings.filterwarnings('ignore')


class agentD:
    def __init__(
        self,
        tools,
        model=None,
        temp=0.1,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=10,
        **kwargs
    ):
        self.tools = tools
        self.suffix = kwargs.get('suffix',
                                 SUFFIX.format(tool_desc="{tool_desc}",
                                        input="{input}",
                                        agent_scratchpad="{agent_scratchpad}"))
        self.prefix = kwargs.get('prefix')
        self.format_instructions = kwargs.get('format_instructions',
                                              FORMAT_INSTRUCTIONS.format(tool_names="{tool_names}"))
        self.callbacks = kwargs.get('callbacks', None)
        chat_history = ChatMessageHistory(variable_name="chat_history")
        
        # Initialize Language Model
        if model:
            configure_runtime({'model':model})
        self.model = get_chat_model(model=model,temperature=temp)
        
        # Initialize Embedding-based Memory
        embeddings = get_embeddings()
        faiss_index = faiss.IndexFlatL2(embeddings.dimension)
        docstore = InMemoryDocstore({})  # Stores vector embeddings
        id_mapping = {}
        vector_store = FAISS(embeddings, faiss_index, docstore, id_mapping)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        memory = VectorStoreRetrieverMemory(
            retriever=retriever,
            memory_key="chat_history",
            input_key='input',
            output_key="output",
            return_messages=True
        )
        
        # Initialize Agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.model,
            agent=agent_type,
            verbose=False,
            memory=memory,
            handle_parsing_errors=True,
            max_iterations=max_iterations,
            agent_kwargs={
                "system_message": """You are an expert drug discovery AI assistant. You must:
                    1. Always check chat history before responding
                    2. Use information from previous messages (like names, preferences, or context)
                    3. Maintain consistent knowledge of previous interactions
                    4. Never reintroduce yourself if you've already done so""",
                "prefix": self.prefix,
                "suffix": self.suffix,
                "format_instructions": self.format_instructions,
                "memory_prompts": [chat_history],
                "input_variables": [
                    "input",
                    "agent_scratchpad",
                    "chat_history"
                ],
            },
            return_intermediate_steps=True,
            callbacks=self.callbacks
        )
    
    def __call__(self, prompt):
        tool_desc = [tool.description for tool in self.tools]
        return self.agent.invoke({"input": prompt, "tool_desc": tool_desc})
    
    async def iter(self, prompt):
        """
        This method enables step-by-step iteration through the agent’s response, 
        which is helpful for managing intermediate stages like user feedback.
        """
        tool_desc = [tool.description for tool in self.tools]
        input_data = {"input": prompt, "tool_desc": tool_desc}
        async for step in self.agent.aiter(input_data):
            yield step
