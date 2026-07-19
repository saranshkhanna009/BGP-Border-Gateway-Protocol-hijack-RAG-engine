

from typing import Dict, Any
import ollama
from ..storage.graph_storage import BGPGraphStorage
from ..storage.vector_storage import BGPVectorStore
from ..models.embedding_engine import BGPPlaybookEmbedder

class BGPTriageEngine:
    """
    Coordinates topological context and semantic vector retrieval 
    to prompt a local offline LLM for incident mitigation plans.
    Handles the analytical 'triage' layer of an anomaly event.
    """
    def __init__(
        self, 
        graph_db: BGPGraphStorage, 
        vector_db: BGPVectorStore, 
        embedder: BGPPlaybookEmbedder,
        llm_model: str = "llama3.2"
    ):
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.embedder = embedder
        self.llm_model = llm_model

    def generate_mitigation_plan(self, target_asn: int, incident_alert: str) -> Dict[str, Any]:
        """
        Retrieves graph topology, pulls matching runbook chunks, 
        and orchestrates the final prompt to generate a step-by-step solution.
        """
        # 1. Pull topological peers from Graph cache
        peers = self.graph_db.get_neighbors(target_asn)
        peer_names = [
            f"AS-{p} ({self.graph_db.nodes[p]['org_name']})" 
            for p in peers if p in self.graph_db.nodes
        ]
        topology_context = f"Target ASN {target_asn} is actively peering with: {', '.join(peer_names)}."

        # 2. Retrieve semantic operational playbooks
        query_vector = self.embedder.embed_query(incident_alert)
        retrieved_chunks = self.vector_db.search_similar_playbooks(query_vector, limit=2)
        
        playbook_context = "\n\n".join([
            f"[Source: {chunk['document_id']}] {chunk['text_content']}" 
            for chunk in retrieved_chunks
        ])

        # 3. Construct the RAG prompt
        system_prompt = (
            "You are a Senior Network Reliability Engineer (SRE) specializing in BGP routing security.\n"
            "Use the provided Network Topology and Playbook Context to write a precise, step-by-step "
            "mitigation plan. Do not guess or invent commands outside of the provided context."
        )

        user_prompt = f"""
=== Live Incident Alert ===
{incident_alert}

=== Network Topology Context ===
{topology_context}

=== Retrieved Playbook Context ===
{playbook_context}

=== Request ===
Generate a step-by-step action plan to mitigate this issue immediately. Include command line examples based on the playbooks if available.
"""

        # 4. Invoke local LLM
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0.2}  # Keep low for consistent technical output
            )
            response_content = response["message"]["content"]
        except Exception as e:
            response_content = f"Error communicating with local LLM engine: {str(e)}"

        return {
            "incident": incident_alert,
            "target_asn": target_asn,
            "retrieved_nodes": peers,
            "solution": response_content
        }