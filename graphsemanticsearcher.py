import numpy as np
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

class GraphSemanticSearcher:
    def __init__(self, knowledge_graph, embeddings=None):
        """
        :param knowledge_graph: NetworkX graph with nodes storing {'summary', 'docstring', 'file'}
        :param embeddings: LangChain Embedding model
        """
        self.graph = knowledge_graph
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")

        # Precompute embeddings for each node (summary + docstring)
        self.node_texts = []
        self.node_metadata = []
        for node, data in self.graph.nodes(data=True):
            text = (data.get("summary", "") + " " + data.get("docstring", "")).strip()
            if text:
                self.node_texts.append(text)
                self.node_metadata.append({"node": node, "file": data.get("file")})

        if self.node_texts:
            self.node_embeddings = self.embeddings.embed_documents(self.node_texts)
        else:
            self.node_embeddings = []

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Semantic retrieval from the knowledge graph using embeddings."""
        if not self.node_embeddings:
            return []

        query_vec = self.embeddings.embed_query(query)

        # Cosine similarity
        sims = []
        for idx, node_vec in enumerate(self.node_embeddings):
            score = np.dot(query_vec, node_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(node_vec)
            )
            sims.append((idx, score))

        # Rank by similarity
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:k]

        results = []
        for idx, score in sims:
            text = self.node_texts[idx]
            metadata = self.node_metadata[idx]
            results.append((
                Document(page_content=text, metadata=metadata),
                float(score)
            ))

        return results
