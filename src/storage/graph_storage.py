

from typing import Dict, List, Set, Tuple

class BGPGraphStorage:
   
    def __init__(self):
        # Maps an ASN to its basic metadata payload
        self.nodes: Dict[int, Dict] = {}
        # Adjacency list: ASN -> Set of peer ASNs
        self.adjacency: Dict[int, Set[int]] = {}
        # Tracks active prefixes announced by each ASN: ASN -> List of IP CIDRs
        self.prefix_announcements: Dict[int, List[str]] = {}

    def add_as_node(self, asn: int, org_name: str, country: str = "Unknown") -> None:
        """Registers a verified Autonomous System inside the topology network."""
        if asn not in self.nodes:
            self.nodes[asn] = {
                "asn": asn,
                "org_name": org_name,
                "country": country
            }
            self.adjacency[asn] = set()
            self.prefix_announcements[asn] = []

    def add_peering_link(self, asn_a: int, asn_b: int) -> None:
        
        if asn_a not in self.nodes or asn_b not in self.nodes:
            raise KeyError("Both Autonomous Systems must be registered node entities before linking.")
        self.adjacency[asn_a].add(asn_b)
        self.adjacency[asn_b].add(asn_a)

    def register_prefix(self, asn: int, prefix: str) -> None:
        
        if asn not in self.nodes:
            raise KeyError(f"ASN {asn} is not registered in the topology database.")
        if prefix not in self.prefix_announcements[asn]:
            self.prefix_announcements[asn].append(prefix)

    def get_neighbors(self, asn: int) -> List[int]:
        """Returns direct BGP peers connected to this ASN."""
        return list(self.adjacency.get(asn, []))

    def export_edge_index(self) -> Tuple[List[int], List[int]]:
        """
        Exports the structural graph topology directly as parallel source and target indices.
        Perfect for direct feeding into our PyTorch Geometric GraphSAGE tensor model!
        """
        sources = []
        targets = []
        visited_edges = set()

        for node, peers in self.adjacency.items():
            for peer in peers:
                edge = tuple(sorted((node, peer)))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    # Create standard undirected edges represented as directed pairs
                    sources.append(node)
                    targets.append(peer)
                    
        return sources, targets

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)