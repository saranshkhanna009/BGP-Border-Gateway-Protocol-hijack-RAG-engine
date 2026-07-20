

import torch
from src.data.stream_consumer import BGPRouteTelemetry, ASNTopologyRegistry
from src.data.document_parsar import OperationalDocumentParser
from src.models.graph_detector import BGPGraphSAGEEncoder
from src.models.train_gnn import BGPTopologyTrainer  # <-- Our new import!
from src.models.embedding_engine import BGPPlaybookEmbedder
from src.storage.graph_storage import BGPGraphStorage
from src.storage.vector_storage import BGPVectorStore
from src.orchestrator.llm_triage import BGPTriageEngine
from src.orchestrator.pipeline import BGPAnomalyResponsePipeline

def run_end_to_end_orchestration():
    print("=================================================================")
    print("                 STARTING COOPERATIVE BGP-RAG PIPELINE           ")
    print("=================================================================\n")

    # Step 1: Initialize Registry and Topology Cache
    registry = ASNTopologyRegistry()
    graph_db = BGPGraphStorage()

    # Create normal baseline peering map (AS-15169 -> AS-2914 -> AS-3356)
    graph_db.add_as_node(15169, "Google Cloud", "US")
    graph_db.add_as_node(2914, "NTT Global", "JP")
    graph_db.add_as_node(3356, "Lumen Technologies", "US")
    
    graph_db.add_peering_link(15169, 2914)
    graph_db.add_peering_link(2914, 3356)
    
    # Initialize registry mappings for the baseline
    for node in graph_db.nodes.keys():
        registry.get_local_index(node)

    # Step 2: Index Operational Manuals
    print("[Pipeline] Embedding operational BGP manuals to Vector DB...")
    mock_playbook = """
    INCIDENT MANUAL 112: MITIGATING UNEXPECTED PEERING HOPS
    When a BGP neighbor advertises an illegitimate path, isolate that peer immediately.
    Execute these CLI commands:
    1. 'neighbor [IP_ADDRESS] route-map BLOCK-IN in' to drop path propagation.
    2. 'clear ip bgp [IP_ADDRESS] soft in' to force routing recalculation.
    """
    
    parser = OperationalDocumentParser(chunk_size=25, chunk_overlap=3)
    chunks = parser.generate_chunks(document_id="runbook_112", raw_content=mock_playbook)
    chunk_texts = [c["text_content"] for c in chunks]
    
    embedder = BGPPlaybookEmbedder(model_name="all-MiniLM-L6-v2")
    embeddings = embedder.embed_documents(chunk_texts)
    
    vector_db = BGPVectorStore(vector_size=embedder.embedding_dimension)
    vector_db.upsert_playbook_chunks(chunks, embeddings)

    # Step 3: Initialize and TRAIN the GraphSAGE model
    print("[Pipeline] Initializing GraphSAGE model and Optimizer...")
    gnn_model = BGPGraphSAGEEncoder(in_channels=8, hidden_channels=16, out_channels=8)
    
    # Instantiate the new trainer
    trainer = BGPTopologyTrainer(model=gnn_model, lr=0.01)
    
    # Prepare training tensors from our graph
    src_asns, dst_asns = graph_db.export_edge_index()
    edges = [[registry.get_local_index(u), registry.get_local_index(v)] for u, v in zip(src_asns, dst_asns)]
    pos_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.eye(registry.total_nodes, 8)

    print("[Pipeline] Training GNN on baseline topology to learn normal peering structures...")
    for epoch in range(1, 51):
        loss = trainer.train_epoch(x, pos_edge_index, num_nodes=registry.total_nodes)
        if epoch % 10 == 0:
            print(f" -> Epoch {epoch:02d} | Loss: {loss:.4f}")

    # Step 4: Save the trained model weights to disk
    weights_path = "src/models/gnn_weights.pt"
    torch.save(gnn_model.state_dict(), weights_path)
    print(f" -> Model trained successfully! Saved weights to: '{weights_path}'")

    # Step 5: Initialize Triage and Main Pipeline Engines
    triage_engine = BGPTriageEngine(
        graph_db=graph_db,
        vector_db=vector_db,
        embedder=embedder,
        llm_model="llama3.2"
    )

    pipeline = BGPAnomalyResponsePipeline(
        gnn_model=gnn_model,
        graph_db=graph_db,
        triage_engine=triage_engine,
        topology_registry=registry,
        anomaly_threshold=0.65  # Optimized threshold bounds
    )

    # --- SIMULATE PACKET STREAMS ---
    
    # Packet A: A normal expected routing path
    normal_packet = BGPRouteTelemetry(
        prefix="8.8.8.0/24",
        as_path=[15169, 2914, 3356],
        origin_as=3356,
        peer_ip="192.0.2.1"
    )
    pipeline.process_telemetry(normal_packet)

    # Packet B: An unexpected malicious route path (AS-15169 suddenly peering with AS-3356 directly)
    hijack_packet = BGPRouteTelemetry(
        prefix="1.1.1.0/24",
        as_path=[15169, 3356],  # This direct hop does not exist in our baseline topology!
        origin_as=3356,
        peer_ip="198.51.100.2"
    )
    
    execution_result = pipeline.process_telemetry(hijack_packet)

    if execution_result["status"] == "escalated":
        print("\n" + "="*40 + " GENERATED MITIGATION REPORT " + "="*40)
        print(f"Target Threat Entity : AS-{execution_result['report']['target_asn']}")
        print(f"GNN Detected Hop     : {execution_result['offending_hop']}")
        print(f"Path Anomaly Score   : {execution_result['max_score']:.4f}")
        print(f"\nAI Mitigation Action Plan:\n{execution_result['report']['solution']}")
        print("="*109 + "\n")

if __name__ == "__main__":
    run_end_to_end_orchestration()