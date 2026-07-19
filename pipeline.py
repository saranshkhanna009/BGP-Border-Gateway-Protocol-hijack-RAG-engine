

import torch
from typing import List, Dict, Any
from ..data.stream_consumer import BGPRouteTelemetry, ASNTopologyRegistry
from ..models.graph_detector import BGPGraphSAGEEncoder
from ..storage.graph_storage import BGPGraphStorage
from .llm_triage import BGPTriageEngine

class BGPAnomalyResponsePipeline:
    """
    The main coordinator loop. Ingests raw routing telemetry streams, 
    evaluates them against the GraphSAGE spatial anomaly detector, 
    and pipes high-risk anomalies directly to the LLM Triage Engine.
    """
    def __init__(
        self,
        gnn_model: BGPGraphSAGEEncoder,
        graph_db: BGPGraphStorage,
        triage_engine: BGPTriageEngine,
        topology_registry: ASNTopologyRegistry,
        anomaly_threshold: float = 0.75
    ):
        self.gnn_model = gnn_model
        self.graph_db = graph_db
        self.triage_engine = triage_engine
        self.registry = topology_registry
        self.anomaly_threshold = anomaly_threshold

    def _prepare_gnn_inputs(self) -> tuple:
        """
        Converts the active graph topology state from graph_db into 
        parallel tensor indexes compatible with the GNN model forward pass.
        """
        # 1. Export edge pairs from our topology store
        src_asns, dst_asns = self.graph_db.export_edge_index()
        
        # 2. Map global ASNs to dense, zero-indexed local node IDs
        edges = []
        for u_asn, v_asn in zip(src_asns, dst_asns):
            u_id = self.registry.get_local_index(u_asn)
            v_id = self.registry.get_local_index(v_asn)
            edges.append([u_id, v_id])
            
        if not edges:
            # Fallback if the graph cache is completely empty
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        num_nodes = self.registry.total_nodes
        # Generate an identity feature matrix representing node features
        # Width matches the in_channels parameter (8) established during model init
        x = torch.eye(num_nodes, 8) if num_nodes > 0 else torch.empty((0, 8))
        
        return x, edge_index

    def process_telemetry(self, telemetry: BGPRouteTelemetry) -> Dict[str, Any]:
        """
        Evaluates an incoming live route announcement. 
        If an anomalous path is flagged, it launches LLM containment orchestration.
        """
        print(f"\n[Monitor] Analyzing incoming route announcement: Prefix={telemetry.prefix} via AS-Path={telemetry.as_path}")
        
        # 1. Ensure all ASNs in this path are registered in our local mapper
        for asn in telemetry.as_path:
            self.registry.get_local_index(asn)

        # 2. Prepare tensors from active graph topology
        x, edge_index = self._prepare_gnn_inputs()
        
        # 3. Calculate anomaly score for each sequential hop in the announced path
        highest_score = 0.0
        anomalous_hop = None

        self.gnn_model.eval()
        with torch.no_grad():
            node_embeddings = self.gnn_model(x, edge_index)
            
            for i in range(len(telemetry.as_path) - 1):
                u_asn = telemetry.as_path[i]
                v_asn = telemetry.as_path[i + 1]
                
                u_id = torch.tensor([self.registry.get_local_index(u_asn)])
                v_id = torch.tensor([self.registry.get_local_index(v_asn)])
                
                # Check link prediction probability (1.0 - sigmoid(dot_product))
                score = self.gnn_model.compute_anomaly_score(node_embeddings, u_id, v_id).item()
                
                if score > highest_score:
                    highest_score = score
                    anomalous_hop = (u_asn, v_asn)

        print(f" -> Assessment: Max Path Deviation Score = {highest_score:.4f}")

        # 4. If the score breaches threshold boundaries, escalate to LLM triage
        if highest_score >= self.anomaly_threshold:
            offending_asn = anomalous_hop[1] if anomalous_hop else telemetry.origin_as
            print(f" [!] ANOMALY DETECTED: Hop {anomalous_hop} exceeded threshold ({self.anomaly_threshold})!")
            print(f" [!] Initiating LLM mitigation triage workflow for AS-{offending_asn}...")
            
            alert_context = (
                f"GNN Alert: Uncharacteristic link transition detected between AS-{anomalous_hop[0]} "
                f"and AS-{anomalous_hop[1]} during route prefix announcement of {telemetry.prefix}. "
                "Possible route leak or path hijacking event in progress."
            )
            
            triage_report = self.triage_engine.generate_mitigation_plan(
                target_asn=offending_asn,
                incident_alert=alert_context
            )
            return {
                "status": "escalated",
                "max_score": highest_score,
                "offending_hop": anomalous_hop,
                "report": triage_report
            }
            
        print(" -> Status: Path verified. No anomalies detected.")
        return {
            "status": "verified",
            "max_score": highest_score,
            "offending_hop": None,
            "report": None
        }