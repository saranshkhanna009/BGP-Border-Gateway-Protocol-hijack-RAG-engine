# simulations on 150 enttries


import os
import sys
import torch
import random

# Ensure the parent directory is in the Python path so it can find the 'src' package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.stream_consumer import ASNTopologyRegistry
from src.storage.graph_storage import BGPGraphStorage
from src.models.graph_detector import BGPGraphSAGEEncoder
from src.models.train_gnn import BGPTopologyTrainer

def run_large_scale_training():
    print("=================================================================")
    print("       BGP-RAG: 150+ ENTRY TOPOLOGY GNN TRAINING SIMULATION      ")
    print("=================================================================\n")

    # 1. Initialize Registry and Storage
    registry = ASNTopologyRegistry()
    graph_db = BGPGraphStorage()

    # 2. Programmatically generate 150 unique ASNs
    tier1_asns = list(range(1000, 1010))       # 10 Tier-1s (1000-1009)
    tier2_asns = list(range(5000, 5040))       # 40 Tier-2s (5000-5039)
    stub_asns = list(range(15000, 15100))      # 100 Stubs (15000-15099)

    all_asns = tier1_asns + tier2_asns + stub_asns
    total_entries = len(all_asns)

    print(f"[Prep] Registering {total_entries} unique ASNs in the topology map...")
    for asn in all_asns:
        if asn in tier1_asns:
            org_name = f"Tier1-Global-Carrier-{asn}"
        elif asn in tier2_asns:
            org_name = f"Tier2-Regional-ISP-{asn}"
        else:
            org_name = f"Enterprise-Stub-{asn}"
            
        graph_db.add_as_node(asn, org_name, "Global")
        registry.get_local_index(asn)

    # 3. Build Peerings programmatically to connect our 150 entries
    print("[Prep] Establishing baseline transit and peering links...")
    
    # Core: Full-mesh among Tier-1s
    for i in range(len(tier1_asns)):
        for j in range(i + 1, len(tier1_asns)):
            graph_db.add_peering_link(tier1_asns[i], tier1_asns[j])

    # Middle-mile: Connect each Tier-2 ISP to 3 random Tier-1 cores
    for t2 in tier2_asns:
        for t1 in random.sample(tier1_asns, 3):
            graph_db.add_peering_link(t2, t1)
            
    # Last-mile: Connect each Stub enterprise to 2 random Tier-2 regional ISPs
    for stub in stub_asns:
        for t2 in random.sample(tier2_asns, 2):
            graph_db.add_peering_link(stub, t2)

    # 4. Export active training edges
    src_asns, dst_asns = graph_db.export_edge_index()
    edges = [
        [registry.get_local_index(u), registry.get_local_index(v)] 
        for u, v in zip(src_asns, dst_asns)
    ]
    pos_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.eye(registry.total_nodes, 8)

    print(f" -> Topology construction complete!")
    print(f"    Total Node Entries : {registry.total_nodes}")
    print(f"    Total Peering Links: {pos_edge_index.size(1)}")

    # 5. Initialize the Model and the Trainer
    print("\n[Init] Initializing neural network parameters...")
    gnn_model = BGPGraphSAGEEncoder(in_channels=8, hidden_channels=16, out_channels=8)
    trainer = BGPTopologyTrainer(model=gnn_model, lr=0.01, weight_decay=1e-4)

    # 6. Training Loop (100 Epochs)
    print("\n[Train] Optimizing structural embeddings for 150-node graph...")
    for epoch in range(1, 101):
        loss = trainer.train_epoch(x, pos_edge_index, num_nodes=registry.total_nodes)
        if epoch % 10 == 0 or epoch == 1:
            print(f" -> Epoch {epoch:03d} / 100 | Link Prediction Loss: {loss:.6f}")

    # 7. Save the trained weight vectors
    os.makedirs("src/models", exist_ok=True)
    weights_path = "src/models/gnn_weights.pt"
    torch.save(gnn_model.state_dict(), weights_path)
    
    print("\n=================================================================")
    print(f" SUCCESS: Trained on {registry.total_nodes} entries!")
    print(f" Optimized GraphSAGE weights saved to '{weights_path}'")
    print("=================================================================\n")

if __name__ == "__main__":
    run_large_scale_training()