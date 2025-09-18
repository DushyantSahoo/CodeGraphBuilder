from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_retriever_tool
from typing import TypedDict, List, Dict

# ----------------------------
# Define State
# ----------------------------
class State(TypedDict):
    query: str
    retrieved_docs: List[str]
    generated_code: str
    feedback: str
    retry_count: int

# ----------------------------
# Agents Setup
# ----------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# Retriever Tool (Hybrid: vector + graph)
# Assume you already built `hybrid_retriever`
retriever_tool = create_retriever_tool(
    hybrid_retriever,
    "code_retriever",
    "Retrieve relevant functions/classes/docstrings from the codebase"
)

# Code Generator Prompt
codegen_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior Python engineer. Generate robust code using retrieved docs."),
    ("user", "Query: {query}\n\nRetrieved Docs:\n{retrieved_docs}")
])

# Critic Prompt
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a critical reviewer. Check if retrieved docs are sufficient."),
    ("user", "Query: {query}\n\nGenerated Code:\n{generated_code}\n\nDocs:\n{retrieved_docs}\n\n"
             "Answer with 'SUFFICIENT' or 'INSUFFICIENT' and explain why.")
])

# Query Expander Prompt
query_expander_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user query to improve retrieval for code context."),
    ("user", "Original query: {query}\n\nFeedback: {feedback}")
])

# ----------------------------
# Nodes (Graph Functions)
# ----------------------------
def retrieve_code(state: State):
    docs = retriever_tool.invoke(state["query"])
    return {"retrieved_docs": [d.page_content for d in docs]}

def generate_code(state: State):
    result = llm.invoke(
        codegen_prompt.format(
            query=state["query"],
            retrieved_docs="\n---\n".join(state["retrieved_docs"])
        )
    )
    return {"generated_code": result.content}

def critic_review(state: State):
    review = llm.invoke(
        critic_prompt.format(
            query=state["query"],
            generated_code=state["generated_code"],
            retrieved_docs="\n---\n".join(state["retrieved_docs"])
        )
    )
    return {"feedback": review.content}

def retry_retrieval(state: State):
    # If critic said insufficient, retry either by graph expansion or query expansion
    if "SUFFICIENT" in state["feedback"]:
        return END
    elif state["retry_count"] >= 2:  # fail-safe limit
        return END
    else:
        # Query Expansion
        expanded = llm.invoke(
            query_expander_prompt.format(
                query=state["query"],
                feedback=state["feedback"]
            )
        )
        new_query = expanded.content.strip()
        return {"query": new_query, "retry_count": state["retry_count"] + 1}

# ----------------------------
# Build Graph
# ----------------------------
workflow = StateGraph(State)

workflow.add_node("retriever", retrieve_code)
workflow.add_node("codegen", generate_code)
workflow.add_node("critic", critic_review)
workflow.add_node("retry", retry_retrieval)

workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "codegen")
workflow.add_edge("codegen", "critic")
workflow.add_edge("critic", "retry")

# Retry either loops back to retriever or ends
workflow.add_conditional_edges(
    "retry",
    lambda state: END if ("SUFFICIENT" in state["feedback"] or state["retry_count"] >= 2) else "retriever",
    {"__default__": "retriever", END: END}
)

graph = workflow.compile()
