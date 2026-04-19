from typing import Annotated, List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict
import operator

class Claim(TypedDict):
    id: str
    text: str
    status: Optional[str]
    confidence: Optional[float]
    evidence: List[Dict[str, Any]]
    human_review_needed: bool
    reasoning: Optional[str]

class AuditState(TypedDict):
    document_id: str
    document_text: str
    document_metadata: Dict[str, Any]
    extracted_citations: List[str]
    vector_store_path: Optional[str]
    claims: List[Claim]
    current_claim_index: int
    queries: Dict[str, List[str]] # claim_id -> queries
    retrieved_evidence: Dict[str, List[Dict[str, Any]]] # claim_id -> evidence
    verification_results: Dict[str, Dict[str, Any]] # claim_id -> results
    final_report: Optional[Dict[str, Any]]

# Global buffer for LLM token streaming updates
stream_status: Dict[str, Dict[str, Any]] = {}
