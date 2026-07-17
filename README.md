# Autonomous BGP Route Anomaly Triage Agent (GNN + RAG)

A localized network security pipeline combining a GraphSAGE Graph Neural Network (GNN) for spatial topology analysis with a Semantic Retrieval-Augmented Generation (RAG) engine to automate Border Gateway Protocol (BGP) route hijack detection and mitigation.

---

## 1. System Architecture & Component Flow

```text
 [ BGP Telemetry Stream ]
            │
            ▼
 ┌───────────────────────────────────┐
 │ 2. GraphSAGE Anomaly Detector     │◀─── [ 1. 150-Node Topology Map ]
 │    (Link Prediction / Conv Layer) │
 └───────────────────────────────────┘
            │
      [ Score > 0.65 ]
            │
            ▼
 ┌───────────────────────────────────┐
 │ 3. Semantic Vector Database       │◀─── [ Infrastructure Runbooks ]
 │    (all-MiniLM-L6-v2 Embeddings)  │
 └───────────────────────────────────┘
            │
     [ Playbook Context ]
            │
            ▼
 ┌───────────────────────────────────┐
 │ 4. Local Triage Engine            │
 │    (Ollama / Llama 3.2 Model)     │
 └───────────────────────────────────┘
            │
            ▼
 [ Automated CLI Containment Plan ]

Topology Core: Evaluates structural configurations across a synthetic network containing 150 unique Autonomous Systems (ASNs) partitioned into 10 Tier-1 carriers, 40 Tier-2 regional ISPs, and 100 enterprise stub networks.

Neural Framework: Implements a multi-layer GraphSAGE Encoder built in PyTorch, mapping topology adjacency structures to dense vector representations using self-supervised link prediction.

Optimization Profile: Trained over 100 epochs using Binary Cross-Entropy (BCE) loss combined with dynamic negative-edge sampling to distinguish legitimate routing updates from topological shortcuts.

Vector Dimensionality: Chunks and encodes infrastructure manuals into a localized vector registry using 384-dimensional dense vectors via the all-MiniLM-L6-v2 transformer model.
Mitigation Triage: Integrates local orchestration via Ollama running a Llama 3.2 model, mapping real-time structural graph anomalies to contextual network playbooks entirely offline.
=================================================================
                 STARTING COOPERATIVE BGP-RAG PIPELINE           
=================================================================

[1] Loading baseline topology: 150 Autonomous Systems | 420 active peering links.
[2] Loading neural parameters: 'src/models/gnn_weights.pt' initialized.
[3] Indexing documentation: 12 vector spaces generated from runbook files.

[Monitor] Intercepted announcement: Prefix=1.1.1.0/24 | Path=[15169, 3356]
[Evaluate] Link verification mapping execution...
 -> Path Deviation Score: 0.8942 / 1.0000

[!] CRITICAL: Direct peer hop [15169 -> 3356] violates structural baseline bounds.
[!] Triggering semantic retrieval query matching index 'unauthorized_advertisement'...
[!] Context match found. Processing prompt payload via local Llama 3.2 instance...

=================================== GENERATED MITIGATION REPORT ===================================
Target Threat Entity : AS-3356 (Lumen Technologies)
GNN Path Score       : 0.8942 (Threshold: 0.6500)

Automated Containment Matrix:
1. Isolate Peering Vector:
   # neighbor 192.0.2.100 route-map BLOCK-IN in
2. Flush Route Cache:
   # clear ip bgp 192.0.2.100 soft in
3. Verification: Verify target prefix withdrawal via local Routing Information Base (RIB).
====================================================================================================
