from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.agents import create_retriever_tool

# ----------------------------
# State
# ----------------------------
class State(dict):
    query: str
    retrieved_docs: list
    generated_code: str
    missing_entities: list
    retry_count: int
    history: list

# ----------------------------
# Initialize LLM and Retriever
# ----------------------------
llm = ChatOpenAI(model="gpt-4o-mini")
retriever_tool = create_retriever_tool(hybrid_retriever, "code_retriever", "Retrieve code from repo")

# ----------------------------
# Graph-aware Query Expander
# ----------------------------
def expand_query(query, missing_entities, graph):
    """
    Expand query using missing entities and their neighbors from code graph.
    """
    graph_terms = []
    for entity in missing_entities:
        if entity in graph:
            graph_terms.extend(graph.neighbors(entity))
    return query + " " + " ".join(missing_entities + graph_terms)

# ----------------------------
# Agents
# ----------------------------
def retriever_agent(state):
    docs = retriever_tool.invoke(state["query"])
    state["retrieved_docs"] = [d.page_content for d in docs]
    return state

def retrieval_critic(state):
    """
    Check if retrieved docs cover all expected entities.
    This could use LLM or AST comparison.
    """
    # Example: LLM identifies missing entities
    prompt = f"""
    Check if the following retrieved docs cover all functions/classes needed for:
    Query: {state['query']}
    Retrieved Docs: {state['retrieved_docs']}
    Return list of missing entities.
    """
    result = llm.invoke(prompt)
    state["missing_entities"] = result.content.splitlines()
    state["retry_retrieval"] = bool(state["missing_entities"])
    return state

def code_generator(state):
    prompt = f"""
    Generate Python code for query: {state['query']}
    Use the following retrieved docs: {state['retrieved_docs']}
    """
    state["generated_code"] = llm.invoke(prompt).content
    return state

def code_critic(state):
    prompt = f"""
    Check if the generated code is complete and correct.
    Generated Code: {state['generated_code']}
    Retrieved Docs: {state['retrieved_docs']}
    """
    review = llm.invoke(prompt).content
    state["retry_code"] = "INSUFFICIENT" in review or "TODO" in state["generated_code"]
    return state

def executor(state):
    try:
        exec(state["generated_code"], {})
        state["execution_result"] = "success"
        state["retry_code"] = False
    except Exception as e:
        state["execution_result"] = str(e)
        state["retry_code"] = True
    return state

def retry_retrieval(state, graph):
    if state.get("retry_retrieval"):
        state["query"] = expand_query(state["query"], state["missing_entities"], graph)
        state["retry_count"] += 1
    return state

# ----------------------------
# Build Graph
# ----------------------------
workflow = StateGraph(State)
workflow.add_node("retriever", retriever_agent)
workflow.add_node("retrieval_critic", retrieval_critic)
workflow.add_node("retry_retrieval", lambda s: retry_retrieval(s, graph=knowledge_graph))
workflow.add_node("code_generator", code_generator)
workflow.add_node("code_critic", code_critic)
workflow.add_node("executor", executor)

workflow.set_entry_point("retriever")

# Retrieval loop with query expander
workflow.add_edge("retriever", "retrieval_critic")
workflow.add_conditional_edges(
    "retrieval_critic",
    lambda s: "retry_retrieval" if s.get("retry_retrieval") else "code_generator"
)
workflow.add_edge("retry_retrieval", "retriever")

# Code generation loop
workflow.add_edge("code_generator", "code_critic")
workflow.add_conditional_edges(
    "code_critic",
    lambda s: "code_generator" if s.get("retry_code") else "executor"
)

# Executor loop
workflow.add_conditional_edges(
    "executor",
    lambda s: "code_generator" if s.get("retry_code") else END
)

app = workflow.compile()
