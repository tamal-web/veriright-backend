from langgraph.graph import StateGraph, END
from .state import AuditState
from .nodes import (
    ingest_document,
    extract_citations,
    fetch_and_build_rag,
    extract_claims,
    generate_search_queries,
    retrieve_evidence,
    rerank_evidence,
    verify_claim_vs_evidence,
    score_confidence,
    route_ambiguous_cases_to_human_review,
    aggregate_final_audit_report
)

def build_workflow():
    workflow = StateGraph(AuditState)

    # 1. Ingest document
    workflow.add_node("ingest_document", ingest_document)
    # 1.1 Extract citations
    workflow.add_node("extract_citations", extract_citations)
    # 1.2 Fetch and Build RAG
    workflow.add_node("fetch_and_build_rag", fetch_and_build_rag)
    # 2. Extract claims
    workflow.add_node("extract_claims", extract_claims)
    # 3. Generate search queries
    workflow.add_node("generate_search_queries", generate_search_queries)
    # 4. Retrieve evidence
    workflow.add_node("retrieve_evidence", retrieve_evidence)
    # 5. Rerank evidence
    workflow.add_node("rerank_evidence", rerank_evidence)
    # 6. Verify claim vs evidence
    workflow.add_node("verify_claim_vs_evidence", verify_claim_vs_evidence)
    # 7. Score confidence
    workflow.add_node("score_confidence", score_confidence)
    # 8. Route ambiguous cases to human review
    workflow.add_node("route_ambiguous_cases", route_ambiguous_cases_to_human_review)
    # 9. Aggregate final audit report
    workflow.add_node("aggregate_final_report", aggregate_final_audit_report)

    # Add edges
    workflow.add_edge("ingest_document", "extract_citations")
    workflow.add_edge("extract_citations", "fetch_and_build_rag")
    workflow.add_edge("fetch_and_build_rag", "extract_claims")
    workflow.add_edge("extract_claims", "generate_search_queries")
    workflow.add_edge("generate_search_queries", "retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "rerank_evidence")
    workflow.add_edge("rerank_evidence", "verify_claim_vs_evidence")
    workflow.add_edge("verify_claim_vs_evidence", "score_confidence")
    workflow.add_edge("score_confidence", "route_ambiguous_cases")
    workflow.add_edge("route_ambiguous_cases", "aggregate_final_report")
    workflow.add_edge("aggregate_final_report", END)

    workflow.set_entry_point("ingest_document")
    
    return workflow.compile()

app_graph = build_workflow()
