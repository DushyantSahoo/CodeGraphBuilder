from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# -----------------------------------
# Shared State
# -----------------------------------
class AgentState(dict):
    query: str
    retrieved_docs: list
    retrieval_verdict: str
    generated_code: str
    code_verdict: str
    execution_result: str


# -----------------------------------
# Retriever Agent (Hybrid: Vector + Graph)
# -----------------------------------
def retriever_agent(state: AgentState) -> AgentState:
    query = state["query"]

    # Replace with your hybrid retriever
    results = hybrid_retriever.get_relevant_documents(query)
    state["retrieved_docs"] = results
    return state


# -----------------------------------
# Retrieval Critic Agent
# -----------------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

def retrieval_critic_agent(state: AgentState) -> AgentState:
    query = state["query"]
    docs = state["retrieved_docs"]

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a codebase expert.
User query: {query}

Retrieved context:
{context}

Check if the retrieved context is sufficient to answer the query.

Answer strictly with:
- "sufficient" if the retrieved docs clearly contain the needed classes/functions.
- "insufficient" if critical pieces are missing or irrelevant.
"""

    verdict = llm.invoke(prompt).content.strip().lower()
    state["retrieval_verdict"] = verdict
    return state


# -----------------------------------
# Code Generator Agent
# -----------------------------------
def code_generator_agent(state: AgentState) -> AgentState:
    query = state["query"]
    docs = state["retrieved_docs"]

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a coding assistant.
User query: {query}

Relevant code snippets from the codebase:
{context}

Generate Python code that satisfies the query using only the retrieved classes and functions.
"""

    code = llm.invoke(prompt).content
    state["generated_code"] = code
    return state


# -----------------------------------
# Code Critic Agent
# -----------------------------------
def code_critic_agent(state: AgentState) -> AgentState:
    query = state["query"]
    code = state["generated_code"]

    prompt = f"""
You are a senior code reviewer.
User query: {query}

Generated code:
{code}

Check:
1. Does this code fully satisfy the query?
2. Does it correctly use the retrieved functions/classes?

Answer strictly with either:
- "sufficient"
- "insufficient"
"""

    verdict = llm.invoke(prompt).content.strip().lower()
    state["code_verdict"] = verdict
    return state


# -----------------------------------
# Execution Agent
# -----------------------------------
def execution_agent(state: AgentState) -> AgentState:
    code = state["generated_code"]

    try:
        exec_locals = {}
        exec(code, {}, exec_locals)
        state["execution_result"] = "Execution succeeded"
    except Exception as e:
        state["execution_result"] = f"Execution error: {e}"

    return state


# -----------------------------------
# LangGraph Wiring
# -----------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("Retriever", retriever_agent)
workflow.add_node("RetrievalCritic", retrieval_critic_agent)
workflow.add_node("CodeGen", code_generator_agent)
workflow.add_node("CodeCritic", code_critic_agent)
workflow.add_node("Executor", execution_agent)

workflow.set_entry_point("Retriever")

# Retriever → RetrievalCritic
workflow.add_edge("Retriever", "RetrievalCritic")

# RetrievalCritic branching
def check_retrieval(state: AgentState):
    return "CodeGen" if state["retrieval_verdict"] == "sufficient" else "Retriever"

workflow.add_conditional_edges("RetrievalCritic", check_retrieval, {
    "CodeGen": "CodeGen",
    "Retriever": "Retriever"
})

# CodeGen → CodeCritic
workflow.add_edge("CodeGen", "CodeCritic")

# CodeCritic branching
def check_code(state: AgentState):
    return "Executor" if state["code_verdict"] == "sufficient" else "CodeGen"

workflow.add_conditional_edges("CodeCritic", check_code, {
    "Executor": "Executor",
    "CodeGen": "CodeGen"
})

# Executor → END
workflow.add_edge("Executor", END)

app = workflow.compile()
