# train_simulation.py

import os
import torch
import random
from src.data.stream_consumer import ASNTopologyRegistry
from src.storage.graph_storage import BGPGraphStorage
from src.models.graph_detector import BGPGraphSAGEEncoder
from src.models.train_gnn import BGPTopologyTrainer

def run_large_scale_training():
    print("=================================================================")
    print("           BGP-RAG: LARGE TOPOLOGY GNN TRAINING SIMULATION       ")
    print("=================================================================\n")

    # 1. Initialize Registry and Storage
    registry = ASNTopologyRegistry()
    graph_db = BGPGraphStorage()

    # 2. Build a mock ISP transit and peering topology (15 ASNs)
    # Tier 1 Carriers
    tier1_asns = [174, 2914, 3356, 1299, 3257]
    # Tier 2 Regional Providers
    tier2_asns = [5089, 8220, 9002, 1273, 6461]
    # Enterprise/Stub Networks
    stub_asns = [15169, 16509, 32934, 13335, 54113]

    all_asns = tier1_asns + tier2_asns + stub_asns

    # Register all nodes
    print(f"[Prep] Registering {len(all_asns)} ASNs in the topology map...")
    for asn in all_asns:
        graph_db.add_as_node(asn, f"Simulated-AS-{asn}", "Global")
        registry.get_local_index(asn)  # Lock into dense tensor mappings

    # Build Peerings: Tier 1 full-mesh core
    print("[Prep] Establishing baseline peering links...")
    for i in range(len(tier1_asns)):
        for j in range(i + 1, len(tier1_asns)):
            graph_db.add_peering_link(tier1_asns[i], tier1_asns[j])

    # Connect Tier 2s to Tier 1s (Transit agreements)
    for t2 in tier2_asns:
        # Each Tier 2 connects to 2 random Tier 1s
        for t1 in random.sample(tier1_asns, 2):
            graph_db.add_peering_link(t2, t1)

    # Connect Stubs to Tier 2s (Enterprise boundaries)
    for stub in stub_asns:
        for t2 in random.sample(tier2_asns, 2):
            graph_db.add_peering_link(stub, t2)

    # 3. Export active training edges
    src_asns, dst_asns = graph_db.export_edge_index()
    edges = [
        [registry.get_local_index(u), registry.get_local_index(v)] 
        for u, v in zip(src_asns, dst_asns)
    ]
    pos_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Feature matrix: One-hot / Identity matrix for local nodes
    x = torch.eye(registry.total_nodes, 8)

    print(f" -> Topology construction complete: {registry.total_nodes} nodes, {pos_edge_index.size(1)} links.")

    # 4. Initialize the Model and the Trainer
    print("\n[Init] Initializing neural network parameters...")
    gnn_model = BGPGraphSAGEEncoder(in_channels=8, hidden_channels=16, out_channels=8)
    trainer = BGPTopologyTrainer(model=gnn_model, lr=0.01, weight_decay=1e-4)

    # 5. Training Loop (100 Epochs)
    print("\n[Train] Optimizing structural embeddings (Self-Supervised Link Prediction)...")
    for epoch in range(1, 101):
        loss = trainer.train_epoch(x, pos_edge_index, num_nodes=registry.total_nodes)
        if epoch % 10 == 0 or epoch == 1:
            print(f" -> Epoch {epoch:03d} / 100 | Link Prediction Loss: {loss:.6f}")

    # 6. Save the trained weight vectors
    os.makedirs("src/models", exist_ok=True)
    weights_path = "src/models/gnn_weights.pt"
    torch.save(gnn_model.state_dict(), weights_path)
    
    print("\n=================================================================")
    print(f" SUCCESS: Optimized GraphSAGE weights saved to '{weights_path}'")
    print("=================================================================\n")

if __name__ == "__main__":
    run_large_scale_training()