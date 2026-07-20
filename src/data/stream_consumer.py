

import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass(frozen=True)
class BGPRouteTelemetry:
    
    prefix: str
    as_path: List[int]
    origin_as: int
    peer_ip: str

    def __post_init__(self):
        
        if not self.prefix or "/" not in self.prefix:
            raise ValueError(f"Invalid CIDR Network Prefix notation: '{self.prefix}'")
        if not self.as_path:
            raise ValueError("BGP route announcement cannot have an empty AS-Path.")
        if self.as_path[-1] != self.origin_as:
            raise ValueError(f"Origin AS mismatch. Path ends with {self.as_path[-1]} but origin is {self.origin_as}")


class ASNTopologyRegistry:

    def __init__(self):
        self._asn_to_index: Dict[int, int] = {}
        self._index_to_asn: Dict[int, int] = {}

    def get_local_index(self, asn: int) -> int:
        """Looks up or issues a sequential identity coordinate for an incoming ASN."""
        if asn not in self._asn_to_index:
            assigned_id = len(self._asn_to_index)
            self._asn_to_index[asn] = assigned_id
            self._index_to_asn[assigned_id] = asn
        return self._asn_to_index[asn]

    @property
    def total_nodes(self) -> int:
        return len(self._asn_to_index)