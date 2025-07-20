# Agent Service Toolkit - Complete Code Analysis & Documentation

## Table of Contents

1. Overview
2. Architecture & Design Patterns
3. Entry Points & Execution Scripts
4. Core Infrastructure Layer
5. Data Models Layer
6. Memory Persistence Layer
7. Agent Implementations Layer
8. Service & Client Layer
9. Complete System Flow
10. Dependency Graph
11. Key Technologies & Packages

## Overview

This is a **multi-layered AI agent service** built with **LangGraph**, **FastAPI**, and **Streamlit**. It provides multiple ways to interact with different types of AI agents, each designed for specific use cases like chatting, research, RAG (Retrieval-Augmented Generation), and human-in-the-loop workflows.

### Core Architecture

The system follows a **layered architecture** with clear separation of concerns:

- **Entry Points** - Multiple ways to interact with the system
- **Service Layer** - FastAPI web service
- **Agent Layer** - Different AI agents with specific capabilities
- **Memory Layer** - Persistent conversation storage
- **Core Layer** - Configuration and LLM abstraction

---

## Architecture & Design Patterns

### Key Design Patterns Used

1. **Factory Pattern** - Agent creation, LLM instantiation
2. **Repository Pattern** - Memory backends (MongoDB, PostgreSQL, SQLite)
3. **Strategy Pattern** - Different agents for different use cases
4. **Command Pattern** - LangGraph Commands for flow control
5. **Observer Pattern** - Streaming responses via SSE
6. **Dependency Injection** - Settings and models injected throughout
7. **State Machine Pattern** - LangGraph workflows
8. **Adapter Pattern** - LLM provider abstraction
9. **Registry Pattern** - Agent discovery and registration
10. **Singleton Pattern** - Settings configuration

### Architectural Principles

- **Async-First**: All operations are asynchronous for scalability
- **Type-Safe**: Extensive use of Pydantic and type hints
- **Modular**: Each component can be developed and tested independently
- **Extensible**: Easy to add new agents, tools, and memory backends
- **Safety-First**: Built-in content filtering and safety checks

---

## Entry Points & Execution Scripts

### 1. `run_service.py` - Production FastAPI Server

```python
# Purpose: Production web service entry point
# Dependencies: service.app, core.settings, uvicorn

def main():
    """Start the FastAPI service"""
    # Flow:
    load_dotenv()  # Load environment variables
    settings = get_settings()  # Get configuration
    app = create_app(settings)  # Create FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Start server

# Execution: python run_service.py
# Result: FastAPI server running on http://localhost:8000
```

### 2. `run_agent.py` - Direct Agent Testing

```python
# Purpose: Development testing without web layer
# Dependencies: agents.agents, core.llm, asyncio

async def main():
    """Test agent directly without web service"""
    # Flow:
    agent_name = "chatbot"  # Default agent
    agent = get_agent(agent_name)  # Load agent from registry
    result = await agent.ainvoke(
        {"messages": [("user", "Hello!")]},
        config={"configurable": {"thread_id": "test"}}
    )
    print(result["messages"][-1].content)

# Execution: python run_agent.py
# Result: Direct agent response in console
```

### 3. `run_client.py` - Client SDK Demo

```python
# Purpose: Demonstrates client SDK usage
# Dependencies: client.client, asyncio

async def main():
    """Demonstrate client SDK usage"""
    # Flow:
    client = AgentClient(base_url="http://localhost:8000")
    response = await client.ainvoke(
        message="Hello, how are you?",
        agent_name="chatbot",
        model="gpt-4o"
    )
    print(response)

# Execution: python run_client.py (requires service running)
# Result: Response from service via client SDK
```

### 4. `streamlit_app.py` - Interactive Web UI

```python
# Purpose: User-friendly web interface
# Dependencies: streamlit, client.client, requests

# Flow:
# 1. Sidebar inputs (agent selection, model, temperature)
# 2. Chat interface with message history
# 3. Send button → client.invoke()
# 4. Stream button → client.astream()
# 5. Display responses in chat format

# Execution: streamlit run streamlit_app.py
# Result: Web UI at http://localhost:8501
```

---

## Core Infrastructure Layer

### `core/settings.py` - Configuration Management

```python
# Purpose: Centralized configuration with environment variable support
# Pattern: Singleton pattern via lru_cache
# Dependencies: pydantic, functools

class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(env_file=".env")

    # LLM Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    llamaguard_model: str = "meta-llama/Llama-Guard-3-8B"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20240620"

    # Memory Configuration
    memory_store: str = "in_memory"  # Options: in_memory, mongodb, postgres, sqlite
    mongodb_connection_string: Optional[str] = None
    postgres_connection_string: Optional[str] = None
    sqlite_db_path: str = "agent_service.db"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Key Features:
# - Environment variable loading from .env file
# - Type validation via Pydantic
# - Singleton pattern for configuration
# - Multiple LLM provider support
# - Flexible memory backend configuration
```

### `core/llm.py` - LLM Abstraction Layer

```python
# Purpose: Provider-agnostic model instantiation
# Pattern: Factory pattern for LLM creation
# Dependencies: langchain_openai, langchain_anthropic, langchain_ollama

def get_model(model_name: Optional[str] = None, **kwargs) -> BaseChatModel:
    """Factory function for creating LLM instances"""

    settings = get_settings()
    model_name = model_name or settings.openai_model

    # Factory logic:
    if "gpt" in model_name:
        return ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            **kwargs
        )
    elif "claude" in model_name:
        return ChatAnthropic(
            model=model_name,
            api_key=settings.anthropic_api_key,
            **kwargs
        )
    elif "llama" in model_name:
        return ChatOllama(
            model=model_name,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# Key Features:
# - Multi-provider support (OpenAI, Anthropic, Ollama)
# - Dynamic model selection
# - Configuration injection
# - Extensible for new providers
```

### `core/__init__.py` - Public API

```python
# Purpose: Clean public interface for core module
# Exports: get_model function and settings

from .llm import get_model
from .settings import get_settings

__all__ = ["get_model", "get_settings"]
```

---

## Data Models Layer

### `schema/models.py` - Core Data Models

```python
# Purpose: Database entity models
# Pattern: Pydantic BaseModel for validation
# Dependencies: pydantic, datetime, typing

class User(BaseModel):
    """User entity model"""
    user_id: str
    name: Optional[str] = None

class Chat(BaseModel):
    """Chat session entity model"""
    chat_id: str
    user_id: str
    title: Optional[str] = None
    created_at: datetime

class Message(BaseModel):
    """Message entity model"""
    message_id: str
    chat_id: str
    content: str
    role: Literal["user", "assistant"]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

# Key Features:
# - Type safety with Pydantic validation
# - Automatic datetime handling
# - Flexible metadata support
# - Clear entity relationships
```

### `schema/schema.py` - API Schemas

```python
# Purpose: API contract definition
# Pattern: Request/Response DTOs
# Dependencies: pydantic, typing

class AgentRequest(BaseModel):
    """Request schema for agent endpoints"""
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_name: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Additional kwargs for agent configuration

class AgentResponse(BaseModel):
    """Response schema for agent endpoints"""
    response: str
    session_id: str
    agent_name: str
    model: str
    metadata: Optional[Dict[str, Any]] = None

class StreamResponse(BaseModel):
    """Streaming response chunk schema"""
    content: str
    done: bool = False

# Key Features:
# - Clear API contracts
# - Optional parameters with defaults
# - Extensible metadata fields
# - Streaming support
```

### `schema/task_data.py` - Task Definitions

```python
# Purpose: Background task management
# Pattern: Task-oriented data modeling
# Dependencies: pydantic, datetime

class TaskData(BaseModel):
    """Task entity for background processing"""
    task_id: str
    task_type: str
    description: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None

# Key Features:
# - Task lifecycle management
# - Status tracking
# - Flexible metadata and results
# - Timestamp tracking
```

---

## Memory Persistence Layer

### `memory/mongodb.py` - MongoDB Implementation

```python
# Purpose: MongoDB persistence backend
# Pattern: Repository pattern with async operations
# Dependencies: motor, pymongo, asyncio

class MongoDBMemory:
    """MongoDB-based conversation memory"""

    def __init__(self, connection_string: str):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client.agent_service
        self.messages = self.db.messages

    async def save_message(self, message: Message) -> None:
        """Save message to MongoDB"""
        await self.messages.insert_one(message.model_dump())

    async def get_messages(self, chat_id: str) -> List[Message]:
        """Retrieve messages for a chat session"""
        cursor = self.messages.find({"chat_id": chat_id}).sort("timestamp", 1)
        docs = await cursor.to_list(None)
        return [Message(**doc) for doc in docs]

    async def delete_messages(self, chat_id: str) -> None:
        """Delete all messages for a chat session"""
        await self.messages.delete_many({"chat_id": chat_id})

    async def close(self) -> None:
        """Close database connection"""
        self.client.close()

# Key Features:
# - Async MongoDB operations
# - Document-based storage
# - Automatic serialization/deserialization
# - Connection management
```

### `memory/postgres.py` - PostgreSQL Implementation

```python
# Purpose: PostgreSQL persistence backend
# Pattern: Repository pattern with connection pooling
# Dependencies: asyncpg, asyncio

class PostgresMemory:
    """PostgreSQL-based conversation memory"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None

    async def _ensure_pool(self):
        """Ensure connection pool exists"""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.connection_string)
            await self._create_tables()

    async def _create_tables(self):
        """Create database tables if they don't exist"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    role TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    metadata JSONB
                )
            """)

    async def save_message(self, message: Message) -> None:
        """Save message to PostgreSQL"""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO messages VALUES ($1, $2, $3, $4, $5, $6)",
                message.message_id, message.chat_id, message.content,
                message.role, message.timestamp, json.dumps(message.metadata)
            )

    async def get_messages(self, chat_id: str) -> List[Message]:
        """Retrieve messages for a chat session"""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM messages WHERE chat_id = $1 ORDER BY timestamp",
                chat_id
            )
            return [Message(
                message_id=row['message_id'],
                chat_id=row['chat_id'],
                content=row['content'],
                role=row['role'],
                timestamp=row['timestamp'],
                metadata=json.loads(row['metadata']) if row['metadata'] else None
            ) for row in rows]

# Key Features:
# - Connection pooling for performance
# - JSONB support for metadata
# - Automatic table creation
# - SQL-based querying
```

### `memory/sqlite.py` - SQLite Implementation

```python
# Purpose: Lightweight local persistence
# Pattern: Repository pattern with file-based storage
# Dependencies: aiosqlite, asyncio

class SQLiteMemory:
    """SQLite-based conversation memory"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    async def _get_connection(self):
        """Get database connection"""
        conn = await aiosqlite.connect(self.db_path)
        await self._create_tables(conn)
        return conn

    async def _create_tables(self, conn):
        """Create database tables if they don't exist"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                content TEXT NOT NULL,
                role TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        """)
        await conn.commit()

    async def save_message(self, message: Message) -> None:
        """Save message to SQLite"""
        async with await self._get_connection() as conn:
            await conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (message.message_id, message.chat_id, message.content,
                 message.role, message.timestamp.isoformat(),
                 json.dumps(message.metadata) if message.metadata else None)
            )
            await conn.commit()

# Key Features:
# - File-based storage
# - No external dependencies
# - Automatic database creation
# - JSON metadata serialization
```

---

## Agent Implementations Layer

### `agents/agents.py` - Agent Registry & Factory

```python
# Purpose: Dynamic agent loading and discovery
# Pattern: Registry + Factory + Dynamic imports
# Dependencies: importlib, typing

# Agent Registry - Maps agent names to their implementation modules
AGENTS = {
    "chatbot": "agents.chatbot:create_chatbot_agent",
    "research_assistant": "agents.research_assistant:create_research_assistant_agent",
    "rag_assistant": "agents.rag_assistant:create_rag_assistant_agent",
    "knowledge_base_agent": "agents.knowledge_base_agent:create_knowledge_base_agent",
    "interrupt_agent": "agents.interrupt_agent:create_interrupt_agent",
    "command_agent": "agents.command_agent:create_command_agent",
    "bg_task_agent": "agents.bg_task_agent.agent:create_bg_task_agent"
}

DEFAULT_AGENT = "chatbot"

def get_agent(agent_name: Optional[str] = None) -> CompiledStateGraph:
    """Factory function to create and return compiled agent"""

    agent_name = agent_name or DEFAULT_AGENT

    if agent_name not in AGENTS:
        raise ValueError(f"Unknown agent: {agent_name}")

    # Dynamic import and instantiation
    agent_path = AGENTS[agent_name]
    module_name, func_name = agent_path.split(":")

    # Import the module dynamically
    module = importlib.import_module(f"agents.{module_name}")

    # Get the creation function
    create_func = getattr(module, func_name)

    # Create and return the compiled agent
    return create_func()

def list_agents() -> List[str]:
    """List all available agents"""
    return list(AGENTS.keys())

# Key Features:
# - Dynamic agent discovery
# - Lazy loading of agent modules
# - Extensible registry system
# - Error handling for unknown agents
```

### `agents/chatbot.py` - Basic Conversational Agent

```python
# Purpose: Simple chat with tool calling capabilities
# Pattern: LangGraph state machine
# Dependencies: langgraph, langchain_core, agents.tools

from typing import TypedDict
from typing_extensions import Annotated
from langgraph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

# State Definition
class AgentState(TypedDict):
    """State for chatbot agent"""
    messages: Annotated[list[AnyMessage], add_messages]

async def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Call the language model with tools"""
    from agents.tools import get_tools
    from core.llm import get_model

    # Get model with tools attached
    model = get_model().bind_tools(get_tools())

    # Call model with conversation history
    response = await model.ainvoke(state["messages"], config)

    # Return updated state
    return {"messages": [response]}

def create_chatbot_agent() -> CompiledStateGraph:
    """Create and compile the chatbot agent"""

    # Create state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("model", call_model)

    # Define flow
    workflow.add_edge(START, "model")
    workflow.add_edge("model", END)

    # Compile with memory
    memory_store = get_memory_store()
    return workflow.compile(checkpointer=memory_store)

# Flow:
# User Input → Model (with tools) → Tool Calls (if needed) → Response

# Key Features:
# - Simple linear flow
# - Tool calling capability
# - Memory persistence
# - Async execution
```

### `agents/research_assistant.py` - Research-Focused Agent

```python
# Purpose: Research with safety filtering
# Pattern: Conditional routing with safety checks
# Dependencies: agents.llama_guard, agents.tools

class AgentState(TypedDict):
    """Enhanced state with safety assessment"""
    messages: Annotated[list[AnyMessage], add_messages]
    safety_assessment: Optional[Dict[str, Any]]

async def check_safety(state: AgentState, config: RunnableConfig) -> AgentState:
    """Check message safety using LlamaGuard"""
    from agents.llama_guard import assess_safety

    last_message = state["messages"][-1]
    is_safe = await assess_safety(last_message.content)

    return {
        "safety_assessment": {
            "is_safe": is_safe,
            "message": last_message.content
        }
    }

def route_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    """Route based on safety assessment"""
    if state.get("safety_assessment", {}).get("is_safe", True):
        return "safe"
    return "unsafe"

async def handle_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    """Handle unsafe content with appropriate response"""
    response = AIMessage(content="I cannot assist with that request as it may violate safety guidelines.")
    return {"messages": [response]}

def create_research_assistant_agent() -> CompiledStateGraph:
    """Create research assistant with safety filtering"""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("check_safety", check_safety)
    workflow.add_node("model", call_model)
    workflow.add_node("tools", call_tools)
    workflow.add_node("unsafe", handle_unsafe_content)

    # Define conditional routing
    workflow.add_edge(START, "check_safety")
    workflow.add_conditional_edges(
        "check_safety",
        route_safety,
        {"safe": "model", "unsafe": "unsafe"}
    )
    workflow.add_conditional_edges(
        "model",
        should_continue,
        {"continue": "tools", "end": END}
    )
    workflow.add_edge("tools", "model")
    workflow.add_edge("unsafe", END)

    return workflow.compile(checkpointer=get_memory_store())

# Flow:
# Input → Safety Check → (Safe: Model → Tools → Response) | (Unsafe: Safety Response)

# Key Features:
# - Built-in safety filtering
# - Research-oriented tools
# - Conditional flow routing
# - LlamaGuard integration
```

### `agents/rag_assistant.py` - RAG-Enabled Agent

```python
# Purpose: Retrieval-augmented generation with safety
# Pattern: RAG + Safety + Tool routing
# Dependencies: chromadb, langchain_chroma

async def call_model_with_rag(state: AgentState, config: RunnableConfig) -> AgentState:
    """Call model with RAG tools for enhanced context"""
    from agents.tools import get_rag_tools
    from core.llm import get_model

    # Get model with RAG tools (includes ChromaDB integration)
    rag_tools = get_rag_tools()
    model = get_model().bind_tools(rag_tools)

    # Call model - tools will automatically retrieve relevant documents
    response = await model.ainvoke(state["messages"], config)

    return {"messages": [response]}

def create_rag_assistant_agent() -> CompiledStateGraph:
    """Create RAG assistant with document retrieval"""

    workflow = StateGraph(AgentState)

    # Add nodes with RAG-specific model call
    workflow.add_node("check_safety", check_safety)
    workflow.add_node("model", call_model_with_rag)
    workflow.add_node("tools", call_tools)
    workflow.add_node("unsafe", handle_unsafe_content)

    # Same conditional routing as research assistant
    workflow.add_edge(START, "check_safety")
    workflow.add_conditional_edges(
        "check_safety",
        route_safety,
        {"safe": "model", "unsafe": "unsafe"}
    )
    workflow.add_conditional_edges(
        "model",
        should_continue,
        {"continue": "tools", "end": END}
    )
    workflow.add_edge("tools", "model")
    workflow.add_edge("unsafe", END)

    return workflow.compile(checkpointer=get_memory_store())

# Flow:
# Input → Safety → Model + RAG Tools → Document Retrieval → Enhanced Response

# Key Features:
# - Automatic document retrieval
# - Vector similarity search
# - Enhanced context for responses
# - ChromaDB integration
```

### `agents/knowledge_base_agent.py` - Pure RAG Agent

```python
# Purpose: Pure RAG without tool calling
# Pattern: Sequential retrieve → generate
# Dependencies: langchain_chroma, langchain_openai

class KnowledgeState(TypedDict):
    """State for knowledge base agent"""
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved_documents: Optional[List[str]]

async def retrieve_documents(state: KnowledgeState, config: RunnableConfig) -> KnowledgeState:
    """Retrieve relevant documents based on query"""
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    # Get the last user message as query
    last_message = state["messages"][-1]
    query = last_message.content

    # Initialize vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        collection_name="knowledge_base",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    # Retrieve similar documents
    docs = vectorstore.similarity_search(query, k=5)
    doc_contents = [doc.page_content for doc in docs]

    return {"retrieved_documents": doc_contents}

async def generate_response(state: KnowledgeState, config: RunnableConfig) -> KnowledgeState:
    """Generate response using retrieved documents as context"""
    from core.llm import get_model

    # Format retrieved documents as context
    documents = state.get("retrieved_documents", [])
    context = "\n\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(documents))

    # Create augmented prompt
    last_message = state["messages"][-1]
    augmented_prompt = f"""
    Based on the following documents, please answer the user's question:

    {context}

    User Question: {last_message.content}

    Please provide a comprehensive answer based on the provided documents.
    """

    # Generate response
    model = get_model()
    response = await model.ainvoke([HumanMessage(content=augmented_prompt)], config)

    return {"messages": [response]}

def create_knowledge_base_agent() -> CompiledStateGraph:
    """Create pure RAG agent"""

    workflow = StateGraph(KnowledgeState)

    # Add nodes for sequential RAG
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_response)

    # Simple linear flow
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile(checkpointer=get_memory_store())

# Flow:
# Query → Document Retrieval → Context Augmentation → Response Generation

# Key Features:
# - Pure retrieval-augmented generation
# - No tool calling complexity
# - Direct vector similarity search
# - Context-aware responses
```

### `agents/interrupt_agent.py` - Human-in-the-Loop Agent

```python
# Purpose: Demonstrates stateful workflows with pauses
# Pattern: Interrupt-driven state machine
# Dependencies: langgraph (NodeInterrupt)

from langgraph import StateGraph, START, END
from langgraph.errors import NodeInterrupt

class InterruptState(TypedDict):
    """State for interrupt agent"""
    messages: Annotated[list[AnyMessage], add_messages]
    step_1_complete: bool
    human_feedback: Optional[str]
    step_2_complete: bool

async def step_1(state: InterruptState, config: RunnableConfig) -> InterruptState:
    """First step of the workflow"""

    # Simulate some work
    await asyncio.sleep(1)

    # Mark step 1 as complete
    state_update = {
        "step_1_complete": True,
        "messages": [AIMessage(content="Step 1 completed. Please provide feedback.")]
    }

    # Interrupt for human feedback
    raise NodeInterrupt("Please provide feedback before continuing to step 2")

    return state_update

async def human_feedback(state: InterruptState, config: RunnableConfig) -> InterruptState:
    """Process human feedback"""

    # Get feedback from the latest human message
    last_message = state["messages"][-1]
    if last_message.type == "human":
        feedback = last_message.content
    else:
        feedback = "No feedback provided"

    return {
        "human_feedback": feedback,
        "messages": [AIMessage(content=f"Received feedback: {feedback}")]
    }

async def step_2(state: InterruptState, config: RunnableConfig) -> InterruptState:
    """Second step incorporating human feedback"""

    feedback = state.get("human_feedback", "No feedback")

    # Simulate work with feedback
    await asyncio.sleep(1)

    final_response = f"Step 2 completed using feedback: {feedback}"

    return {
        "step_2_complete": True,
        "messages": [AIMessage(content=final_response)]
    }

def create_interrupt_agent() -> CompiledStateGraph:
    """Create agent with human interrupts"""

    workflow = StateGraph(InterruptState)

    # Add nodes
    workflow.add_node("step_1", step_1)
    workflow.add_node("human_feedback", human_feedback)
    workflow.add_node("step_2", step_2)

    # Define flow with interrupts
    workflow.add_edge(START, "step_1")
    workflow.add_edge("step_1", "human_feedback")  # After interrupt, continues here
    workflow.add_edge("human_feedback", "step_2")
    workflow.add_edge("step_2", END)

    return workflow.compile(checkpointer=get_memory_store())

# Flow:
# Start → Step 1 → (INTERRUPT: Wait for human) → Human Feedback → Step 2 → End

# Key Features:
# - Stateful workflow with pauses
# - Human-in-the-loop processing
# - Resumable execution
# - NodeInterrupt mechanism
```

### `agents/command_agent.py` - Flow Control Demo

```python
# Purpose: Demonstrates LangGraph's Command API
# Pattern: Dynamic routing with Commands
# Dependencies: langgraph (Command)

from langgraph import StateGraph, START, END, Command
import random

class CommandState(TypedDict):
    """State for command agent"""
    messages: Annotated[list[AnyMessage], add_messages]
    route_taken: Optional[str]

async def node_a(state: CommandState, config: RunnableConfig) -> CommandState:
    """Starting node that makes routing decision"""

    # Random choice for demonstration
    choice = random.choice(["b", "c"])

    message = f"Node A: Choosing route {choice.upper()}"

    # Use Command API for dynamic routing
    if choice == "b":
        return Command(
            update={"messages": [AIMessage(content=message)]},
            goto="node_b"
        )
    else:
        return Command(
            update={"messages": [AIMessage(content=message)]},
            goto="node_c"
        )

async def node_b(state: CommandState, config: RunnableConfig) -> CommandState:
    """Route B processing"""
    return {
        "route_taken": "B",
        "messages": [AIMessage(content="Node B: Processing complete")]
    }

async def node_c(state: CommandState, config: RunnableConfig) -> CommandState:
    """Route C processing"""
    return {
        "route_taken": "C",
        "messages": [AIMessage(content="Node C: Processing complete")]
    }

def create_command_agent() -> CompiledStateGraph:
    """Create agent demonstrating Command API"""

    workflow = StateGraph(CommandState)

    # Add nodes
    workflow.add_node("node_a", node_a)
    workflow.add_node("node_b", node_b)
    workflow.add_node("node_c", node_c)

    # Define flow (Commands handle dynamic routing)
    workflow.add_edge(START, "node_a")
    workflow.add_edge("node_b", END)
    workflow.add_edge("node_c", END)

    return workflow.compile(checkpointer=get_memory_store())

# Flow:
# Start → Node A → (Random Choice: Node B OR Node C) → End

# Key Features:
# - Dynamic routing decisions
# - Command API usage
# - Non-deterministic flow
# - Conditional execution paths
```

### `agents/bg_task_agent/` - Background Processing Agent

```python
# Purpose: Async background processing demo
# Pattern: Background task + model response
# Dependencies: asyncio, langgraph

# File: agents/bg_task_agent/agent.py

class BgTaskState(TypedDict):
    """State for background task agent"""
    messages: Annotated[list[AnyMessage], add_messages]
    task_result: Optional[str]
    task_complete: bool

async def background_task(state: BgTaskState, config: RunnableConfig) -> BgTaskState:
    """Simulate long-running background task"""

    # Simulate work (could be API calls, file processing, etc.)
    await asyncio.sleep(2)

    # Generate task result
    task_result = f"Background task completed at {datetime.now()}"

    return {
        "task_result": task_result,
        "task_complete": True,
        "messages": [AIMessage(content="Background processing started...")]
    }

async def call_model_with_result(state: BgTaskState, config: RunnableConfig) -> BgTaskState:
    """Call model with background task result"""
    from core.llm import get_model

    task_result = state.get("task_result", "No result")
    last_user_message = state["messages"][-2]  # Get user message before our processing message

    # Create enhanced prompt with task result
    prompt = f"""
    User request: {last_user_message.content}

    Background task result: {task_result}

    Please provide a response incorporating both the user's request and the background task result.
    """

    model = get_model()
    response = await model.ainvoke([HumanMessage(content=prompt)], config)

    return {"messages": [response]}

def create_bg_task_agent() -> CompiledStateGraph:
    """Create background task agent"""

    workflow = StateGraph(BgTaskState)

    # Add nodes
    workflow.add_node("background_task", background_task)
    workflow.add_node("model", call_model_with_result)

    # Simple sequential flow
    workflow.add_edge(START, "background_task")
    workflow.add_edge("background_task", "model")
    workflow.add_edge("model", END)

    return workflow.compile(checkpointer=get_memory_store())

# Flow:
# Start → Background Task (async processing) → Model Response → End

# Key Features:
# - Async background processing
# - Task result integration
# - Long-running operation simulation
# - Result incorporation in responses
```

### `agents/tools.py` - Tool Implementations

```python
# Purpose: Reusable tools for agent workflows
# Pattern: @tool decorator for LangChain integration
# Dependencies: langchain_core.tools, numexpr

from langchain_core.tools import tool
import numexpr

@tool
def calculator_func(expression: str) -> str:
    """Calculate mathematical expressions safely using numexpr.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")

    Returns:
        String representation of the calculation result

    Example:
        calculator_func("2 + 3 * 4") -> "14"
    """
    try:
        # Use numexpr for safe mathematical evaluation
        result = numexpr.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

@tool
def database_search_func(query: str) -> str:
    """Search database for information.

    Args:
        query: Search query string

    Returns:
        Search results as formatted string
    """
    # Mock database search implementation
    # In real implementation, this would query actual database
    mock_results = {
        "users": ["Alice (ID: 1)", "Bob (ID: 2)", "Charlie (ID: 3)"],
        "products": ["Laptop (ID: 101)", "Mouse (ID: 102)", "Keyboard (ID: 103)"],
        "orders": ["Order #1001", "Order #1002", "Order #1003"]
    }

    # Simple keyword matching
    results = []
    for category, items in mock_results.items():
        if query.lower() in category.lower():
            results.extend(items[:2])  # Limit results

    if not results:
        return f"No results found for query: '{query}'"

    return f"Database search results for '{query}':\n" + "\n".join(f"- {item}" for item in results)

def get_tools() -> list:
    """Get list of available tools for agents"""
    return [calculator_func, database_search_func]

def get_rag_tools() -> list:
    """Get RAG-specific tools (includes document retrieval)"""
    # Additional RAG tools would be defined here
    # For now, return standard tools
    return get_tools()

# Key Features:
# - Safe mathematical evaluation
# - Extensible tool system
# - LangChain integration via @tool decorator
# - Mock database functionality
# - Easy tool discovery and registration
```

### `agents/llama_guard.py` - Safety Filter

```python
# Purpose: Content safety filtering using Meta's LlamaGuard
# Pattern: LLM-based safety assessment
# Dependencies: langchain_ollama, langchain_core

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def get_llama_guard_model():
    """Get LlamaGuard model for safety assessment"""
    return ChatOllama(
        model="llama-guard3:8b",
        temperature=0,  # Deterministic for safety assessments
    )

SAFETY_PROMPT = """
You are a safety classifier. Classify the following text as either "safe" or "unsafe".

Text to classify: {text}

Classification:"""

async def assess_safety(message: str) -> bool:
    """Assess if a message is safe using LlamaGuard

    Args:
        message: Message content to assess

    Returns:
        True if safe, False if unsafe
    """
    try:
        llama_guard = get_llama_guard_model()

        # Format safety prompt
        prompt = SAFETY_PROMPT.format(text=message)

        # Get safety assessment
        response = await llama_guard.ainvoke([HumanMessage(content=prompt)])

        # Parse response
        classification = response.content.strip().lower()

        return "safe" in classification

    except Exception as e:
        # Default to unsafe if assessment fails
        print(f"Safety assessment error: {e}")
        return False

# Key Features:
# - LlamaGuard integration for content safety
# - Async safety assessment
# - Error handling with safe defaults
# - Deterministic temperature for consistency
```

### `agents/utils.py` - Shared Utilities

```python
# Purpose: Shared utilities for agent implementations
# Dependencies: Various based on utility functions

from typing import Dict, Any, Optional
from langchain_core.messages import AnyMessage

def format_agent_response(message: AnyMessage, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Format agent response for consistent output"""
    return {
        "content": message.content,
        "type": message.type,
        "metadata": metadata or {}
    }

def extract_user_query(messages: list[AnyMessage]) -> str:
    """Extract the latest user query from message history"""
    for message in reversed(messages):
        if message.type == "human":
            return message.content
    return ""

def get_memory_store():
    """Get configured memory store for agents"""
    from core.settings import get_settings

    settings = get_settings()

    if settings.memory_store == "mongodb":
        from memory.mongodb import MongoDBMemory
        return MongoDBMemory(settings.mongodb_connection_string)
    elif settings.memory_store == "postgres":
        from memory.postgres import PostgresMemory
        return PostgresMemory(settings.postgres_connection_string)
    elif settings.memory_store == "sqlite":
        from memory.sqlite import SQLiteMemory
        return SQLiteMemory(settings.sqlite_db_path)
    else:
        # Default to in-memory store
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

# Key Features:
# - Response formatting utilities
# - Message parsing helpers
# - Memory store factory
# - Configuration-based selection
```

---

## Service & Client Layer

### `service/app.py` - FastAPI Application

```python
# Purpose: FastAPI web service with CORS
# Pattern: Router-based modular API
# Dependencies: fastapi, fastapi.middleware.cors

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import agent_router, health_router

def create_app(settings) -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title="Agent Service Toolkit",
        description="Multi-agent AI service with various capabilities",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure based on environment
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(agent_router, prefix="/agent", tags=["agents"])
    app.include_router(health_router, prefix="/health", tags=["health"])

    return app

# Key Features:
# - CORS support for web clients
# - Modular router architecture
# - OpenAPI documentation
# - Environment-based configuration
```

### `service/routers.py` - API Endpoints

```python
# Purpose: FastAPI route definitions
# Pattern: Router-based endpoint organization
# Dependencies: fastapi, agents.agents, schema.schema

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from schema.schema import AgentRequest, AgentResponse
from agents.agents import get_agent, list_agents

agent_router = APIRouter()
health_router = APIRouter()

@agent_router.post("/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    """Invoke agent with a message"""
    try:
        agent = get_agent(request.agent_name)

        result = await agent.ainvoke(
            {"messages": [("user", request.message)]},
            config={
                "configurable": {
                    "thread_id": request.session_id or "default"
                }
            }
        )

        return AgentResponse(
            response=result["messages"][-1].content,
            session_id=request.session_id or "default",
            agent_name=request.agent_name or "chatbot",
            model=request.model or "gpt-4o-mini"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@agent_router.post("/stream")
async def stream_agent(request: AgentRequest):
    """Stream agent response using Server-Sent Events"""

    async def generate():
        try:
            agent = get_agent(request.agent_name)

            async for chunk in agent.astream(
                {"messages": [("user", request.message)]},
                config={
                    "configurable": {
                        "thread_id": request.session_id or "default"
                    }
                }
            ):
                # Format chunk for SSE
                yield f"data: {json.dumps({'content': str(chunk)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/plain")

@agent_router.get("/agents")
async def get_agents():
    """List available agents"""
    return {"agents": list_agents()}

@health_router.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "agent-service-toolkit"}

# Key Features:
# - RESTful API design
# - Streaming support via SSE
# - Error handling with HTTP status codes
# - Agent discovery endpoint
# - Health monitoring
```

### `client/client.py` - SDK Implementation

```python
# Purpose: Client SDK for service interaction
# Pattern: Sync/Async wrapper around HTTP API
# Dependencies: httpx, asyncio, json

import httpx
import asyncio
import json
from typing import Optional, Dict, Any, AsyncGenerator

class AgentClient:
    """Client SDK for Agent Service Toolkit"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._async_client = None

    def _get_sync_client(self) -> httpx.Client:
        """Get synchronous HTTP client"""
        return httpx.Client(timeout=self.timeout)

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get asynchronous HTTP client"""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    def invoke(
        self,
        message: str,
        agent_name: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Synchronously invoke agent"""

        payload = {
            "message": message,
            "agent_name": agent_name,
            "model": model,
            "session_id": session_id,
            **kwargs
        }

        with self._get_sync_client() as client:
            response = client.post(
                f"{self.base_url}/agent/invoke",
                json=payload
            )
            response.raise_for_status()
            return response.json()["response"]

    async def ainvoke(
        self,
        message: str,
        agent_name: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Asynchronously invoke agent"""

        payload = {
            "message": message,
            "agent_name": agent_name,
            "model": model,
            "session_id": session_id,
            **kwargs
        }

        client = await self._get_async_client()
        response = await client.post(
            f"{self.base_url}/agent/invoke",
            json=payload
        )
        response.raise_for_status()
        return response.json()["response"]

    def stream(
        self,
        message: str,
        agent_name: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Synchronously stream agent response"""

        payload = {
            "message": message,
            "agent_name": agent_name,
            "model": model,
            "session_id": session_id,
            **kwargs
        }

        with self._get_sync_client() as client:
            with client.stream(
                "POST",
                f"{self.base_url}/agent/stream",
                json=payload
            ) as response:
                response.raise_for_status()
                for chunk in self._handle_sse_stream(response.iter_text()):
                    yield chunk

    async def astream(
        self,
        message: str,
        agent_name: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Asynchronously stream agent response"""

        payload = {
            "message": message,
            "agent_name": agent_name,
            "model": model,
            "session_id": session_id,
            **kwargs
        }

        client = await self._get_async_client()
        async with client.stream(
            "POST",
            f"{self.base_url}/agent/stream",
            json=payload
        ) as response:
            response.raise_for_status()
            async for chunk in self._handle_async_sse_stream(response.aiter_text()):
                yield chunk

    def _handle_sse_stream(self, text_stream) -> Generator[str, None, None]:
        """Parse Server-Sent Events stream"""
        for line in text_stream:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    if "content" in data:
                        yield data["content"]
                    elif "error" in data:
                        raise Exception(data["error"])
                except json.JSONDecodeError:
                    continue

    async def _handle_async_sse_stream(self, text_stream) -> AsyncGenerator[str, None]:
        """Parse Server-Sent Events stream asynchronously"""
        async for line in text_stream:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    if "content" in data:
                        yield data["content"]
                    elif "error" in data:
                        raise Exception(data["error"])
                except json.JSONDecodeError:
                    continue

    async def close(self):
        """Close async client"""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

# Key Features:
# - Both sync and async interfaces
# - Streaming support via SSE
# - Proper resource management
# - Error handling and HTTP status checking
# - Flexible parameter passing

---

## Complete System Flow

### Production Flow
```

HTTP Request → FastAPI Middleware → Router → Request Validation (Pydantic) →
Agent Factory → Agent Selection → LangGraph Execution → Tool Calls (if needed) →
LLM API Call → Response Processing → Memory Storage → Response Validation →
HTTP Response

```

### Development Flow
```

run_agent.py → Agent Factory → Direct Agent Creation → LangGraph Execution →
Tool Integration → LLM Processing → Console Output

```

### Streaming Flow
```

HTTP Request → FastAPI → SSE Stream → Agent.astream() → Chunk Processing →
Real-time Response Streaming → Client Receives Chunks → UI Updates

```

### Memory Flow
```

Message → Memory Interface → Backend Selection (MongoDB/PostgreSQL/SQLite) →
Serialization → Storage → Retrieval → Deserialization → Message History

```

---

## Dependency Graph

```

Entry Points (run_*.py)
    ↓
Service Layer (service/)
    ↓
Agent Registry (agents/agents.py)
    ↓
Individual Agents (agents/*.py)
    ↓
Tools & Utilities (agents/tools.py, agents/utils.py)
    ↓
Core Infrastructure (core/)
    ↓
Data Models (schema/)
    ↓
Memory Backends (memory/)
    ↓
External Dependencies (LangGraph, FastAPI, etc.)

```

### Cross-Dependencies
- **Agents** ←→ **Tools** (bidirectional tool usage)
- **Agents** → **Memory** (conversation persistence)
- **Service** → **Agents** (API endpoints to agents)
- **Client** → **Service** (SDK to API)
- **Core** → **All Layers** (settings and LLM throughout)

---

## Key Technologies & Packages

### Core Framework Stack
- **LangGraph** - Agent workflow orchestration and state management
- **LangChain** - LLM abstraction, tool integration, and message handling
- **FastAPI** - High-performance async web framework
- **Streamlit** - Interactive web UI development
- **Pydantic** - Data validation, serialization, and type safety

### AI/ML Stack
- **OpenAI GPT models** - Primary language model provider
- **Anthropic Claude** - Alternative language model provider
- **Ollama** - Local language model deployment
- **LlamaGuard** - Content safety filtering
- **ChromaDB** - Vector database for RAG applications
- **OpenAI Embeddings** - Text embedding generation

### Infrastructure Stack
- **asyncio** - Asynchronous programming foundation
- **uvicorn** - ASGI server for FastAPI
- **httpx** - Modern async HTTP client
- **python-dotenv** - Environment variable management

### Database & Storage
- **MongoDB** - Document-based conversation storage
- **PostgreSQL** - Relational database with JSONB support
- **SQLite** - Lightweight local database
- **aiosqlite** - Async SQLite interface
- **asyncpg** - High-performance async PostgreSQL driver
- **motor** - Async MongoDB driver

### Development & Testing
- **typing** - Type hints and annotations
- **functools** - Utility functions (lru_cache for singletons)
- **importlib** - Dynamic module importing
- **json** - Data serialization
- **datetime** - Timestamp handling
- **numexpr** - Safe mathematical expression evaluation

### Web & Networking
- **CORS middleware** - Cross-origin resource sharing
- **Server-Sent Events** - Real-time streaming
- **WebSocket support** - Bidirectional communication (optional)

---

## Innovation & Extension Points

### Easy Extension Areas

1. **New Agents** - Add to `agents/` directory with create function
2. **New Tools** - Add to `agents/tools.py` with @tool decorator
3. **New Memory Backends** - Implement in `memory/` with common interface
4. **New LLM Providers** - Add to `core/llm.py` factory function
5. **New API Endpoints** - Add to `service/routers.py`
6. **Safety Filters** - Extend `agents/llama_guard.py`

### Advanced Patterns Demonstrated

1. **Multi-Agent Orchestration** - Different agents for different tasks
2. **Human-in-the-Loop** - Workflow interruption and resumption
3. **Streaming Responses** - Real-time communication via SSE
4. **Tool Composability** - Dynamic tool selection and chaining
5. **State Persistence** - Multiple backend conversation memory
6. **Safety Integration** - Built-in content filtering
7. **Dynamic Routing** - Conditional flow control in agents
8. **Background Processing** - Async task handling
9. **Error Recovery** - Graceful failure handling throughout

```

## MEMORIZED: Complete Code Analysis & Flow

### Architecture Overview

The Agent Service Toolkit is a comprehensive framework for building, deploying, and managing AI agents with different capabilities. It follows a layered architecture:

1. **Entry Points Layer** - Scripts that provide different ways to run agents
2. **Service Layer** - FastAPI-based API endpoints and authentication
3. **Client Layer** - SDK for programmatic access to agent service
4. **Agent Layer** - Various agent implementations using LangGraph
5. **Tools Layer** - Reusable tools that agents can leverage
6. **Core Layer** - Fundamental services like LLM access and settings
7. **Memory Layer** - Persistence mechanisms for conversation history
8. **Data Models Layer** - Pydantic schemas and database models

### Key Components and Relationships

1. **Entry Points**
   - `run_service.py` - Starts FastAPI server
   - `run_agent.py` - Direct agent testing
   - `run_client.py` - Client SDK demo
   - `streamlit_app.py` - Interactive web UI

2. **Service Components**
   - FastAPI app with CORS support
   - Modular routers for API organization
   - Synchronous and streaming endpoints
   - Health check endpoints

3. **Agent Types**
   - `chatbot.py` - Basic conversational agent
   - `research_assistant.py` - Research-focused with safety filtering
   - `rag_assistant.py` - Retrieval-augmented generation
   - `knowledge_base_agent.py` - Pure RAG implementation
   - `interrupt_agent.py` - Human-in-the-loop with pauses
   - `command_agent.py` - Flow control demonstration
   - `bg_task_agent/` - Background processing agent

4. **Core Infrastructure**
   - Factory pattern for LLM creation
   - Pydantic-based settings with environment variables
   - Centralized configuration management

5. **Memory Systems**
   - MongoDB implementation with async operations
   - PostgreSQL with connection pooling
   - SQLite for lightweight local storage
   - Common interface for all storage backends

6. **Tools & Utilities**
   - Calculator tool with safe evaluation
   - Database search functionality
   - Document retrieval for RAG
   - LlamaGuard integration for content safety

### Flow Patterns

1. **LangGraph State Machine Pattern**

   ```
   User Input → Initial State → Node Processing →
   Conditional Routing → Tool Execution →
   State Updates → Response Generation
   ```

2. **API Request Flow**

   ```
   HTTP Request → FastAPI Middleware → Router →
   Request Validation → Agent Factory →
   LangGraph Execution → Memory Storage → HTTP Response
   ```

3. **Streaming Pattern**

   ```
   Client Request → SSE Stream Setup →
   Agent.astream() → Chunk Processing →
   Real-time Delivery → UI Updates
   ```

4. **Background Processing Pattern**

   ```
   Task Initiation → Async Queue →
   Background Worker → State Updates →
   Result Integration → Response
   ```

### Extension Mechanisms

1. **Agent Registry** - Central registration in `agents.py`
2. **Dynamic Imports** - Lazy loading of agent modules
3. **Tool Decorator** - Simple tool creation with `@tool`
4. **Memory Backend Interface** - Common persistence API
5. **LLM Provider Factory** - Extensible model support

### Execution Models

1. **Synchronous API** - `client.invoke()` and `/agent/invoke`
2. **Asynchronous API** - `client.ainvoke()` with `await`
3. **Streaming API** - `client.stream()` and `client.astream()`
4. **Direct Execution** - `run_agent.py` for development
5. **Web UI** - Streamlit app for interactive usage

### Safety & Error Handling

1. **Content Filtering** - LlamaGuard integration
2. **Exception Handling** - Try/except with proper HTTP responses
3. **Connection Management** - Proper resource cleanup
4. **Timeout Handling** - Configurable client timeouts
5. **Resource Limits** - Token and request size management

### Integration Patterns

1. **LangChain Tool Integration** - Seamless tool usage
2. **Vector Database** - ChromaDB for document retrieval
3. **State Persistence** - Multiple backend options
4. **Environment Configuration** - .env file and variables
5. **Client SDK** - Clean interface for service integration

### Deployment Options

1. **Local Development** - Direct script execution
2. **API Service** - FastAPI server with uvicorn
3. **Web UI** - Streamlit interface
4. **Containerization** - Docker support via Dockerfiles
5. **Production Ready** - CORS, proper error handling, health checks
