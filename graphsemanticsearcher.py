import numpy as np
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import networkx as nx

class GraphSemanticSearcher:
    def __init__(self, knowledge_graph: nx.DiGraph, embeddings=None):
        """
        :param knowledge_graph: NetworkX graph with nodes storing {'summary', 'docstring', 'code', 'file'}
                                Edges represent relationships (inherits, has_method)
        :param embeddings: LangChain Embedding model
        """
        self.graph = knowledge_graph
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")

        # Precompute embeddings for each node (full docstring + code)
        self.node_texts = []
        self.node_metadata = []
        for node, data in self.graph.nodes(data=True):
            text = (data.get("summary", "") + " " + data.get("docstring", "") + "\n" + data.get("code", "")).strip()
            if text:
                self.node_texts.append(text)
                self.node_metadata.append({
                    "node": node,
                    "file": data.get("file"),
                    "type": data.get("type")
                })

        if self.node_texts:
            self.node_embeddings = self.embeddings.embed_documents(self.node_texts)
        else:
            self.node_embeddings = []

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Semantic + graph-aware retrieval from codebase."""
        if not self.node_embeddings:
            return []

        # Embed query
        query_vec = self.embeddings.embed_query(query)

        # Cosine similarity
        sims = []
        for idx, node_vec in enumerate(self.node_embeddings):
            score = np.dot(query_vec, node_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(node_vec)
            )
            sims.append((idx, score))

        # Sort by similarity
        sims = sorted(sims, key=lambda x: x[1], reverse=True)

        # Take top-k, then propagate along graph edges
        results: List[Tuple[Document, float]] = []
        visited_nodes = set()
        for idx, score in sims:
            if len(results) >= k:
                break
            node_meta = self.node_metadata[idx]
            node_name = node_meta["node"]
            if node_name in visited_nodes:
                continue

            # Include this node
            doc = Document(page_content=self.node_texts[idx], metadata=node_meta)
            results.append((doc, float(score)))
            visited_nodes.add(node_name)

            # Propagate to related nodes (children, parents)
            for neighbor in self.graph.neighbors(node_name):
                if neighbor not in visited_nodes:
                    neighbor_data = self.graph.nodes[neighbor]
                    neighbor_text = (neighbor_data.get("summary", "") +
                                     " " + neighbor_data.get("docstring", "") +
                                     "\n" + neighbor_data.get("code", "")).strip()
                    neighbor_doc = Document(page_content=neighbor_text, metadata={
                        "node": neighbor,
                        "file": neighbor_data.get("file"),
                        "type": neighbor_data.get("type")
                    })
                    # Slightly reduce score for propagated nodes
                    results.append((neighbor_doc, float(score) * 0.8))
                    visited_nodes.add(neighbor)

        # Sort final results by score and return top-k
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results[:k]
