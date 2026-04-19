import uuid
import json
import ast
from typing import Dict, Any, List
from .state import AuditState, Claim
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize the LLM pointing to the local endpoint
llm = ChatOpenAI(
    # base_url="http://172.16.75.94:1234/v1",
    base_url="http://10.139.23.47:1234/v1",
    api_key="not-needed", # typically not required for local LM Studio/Ollama endpoints
    # model="google/gemma-2-9b",
    model="qwen2.5-coder-7b-instruct-mlx",
    temperature=0.0
)

# Pydantic schemas for LLM structured output
class ClaimExtraction(BaseModel):
    claims: List[str] = Field(description="A list of factual claims extracted from the document.")

class ClaimVerification(BaseModel):
    reasoning: str = Field(description="Step by step reasoning about the evidence.")
    status: str = Field(description="Must be strictly one of: 'verified', 'unverified', or 'contradicted'.")

def _parse_llm_json(response_content: str) -> dict:
    """Helper to parse JSON out of gemma/local model responses which might include markdown code blocks."""
    text = response_content.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # Fallback to ast if json fails
        try:
            return ast.literal_eval(text)
        except:
            return {}

def ingest_document(state: AuditState) -> AuditState:
    print(f"Ingesting doc {state.get('document_id', 'unknown')}")
    # Initialize lists if empty
    return {**state, "extracted_citations": state.get("extracted_citations", []), "vector_store_path": None}

def extract_citations(state: AuditState) -> AuditState:
    doc_text = state.get("document_text", "")
    print("Extracting citations (URLs/references) from document via LLM...")
    
    prompt = (
        "Extract all explicit citations, web links, or URLs mentioned in the following text as sources.\n"
        "Return ONLY a valid JSON object with a single key 'citations' containing a list of strings (URLs).\n\n"
        f"Text:\n{doc_text[:3000]}"
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_llm_json(response.content)
        citations = parsed.get("citations", [])
    except Exception as e:
        print(f"Error extracting citations: {e}")
        citations = []
        
    if not isinstance(citations, list):
        citations = []
        
    return {**state, "extracted_citations": citations}

def fetch_and_build_rag(state: AuditState) -> AuditState:
    print("Fetching citations and storing extracted texts...")
    citations = state.get("extracted_citations", [])
    
    if not citations:
        print("No citations found.")
        return {**state, "vector_store_path": None}
        
    docs_metadata = {}
    import urllib.request
    import fitz
    import ssl
    from html.parser import HTMLParser
    import json
    
    class SimpleHTMLParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text_parts = []
            
        def handle_data(self, data):
            clean = data.strip()
            if clean:
                self.text_parts.append(clean)
                
        def get_text(self):
            return " ".join(self.text_parts)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print(f"Loading {len(citations)} external citations...")
    failed_citations = []
    
    for url in citations:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            # Expand timeout to 300 seconds (5 minutes) to ensure large PDFs or slow sites fully download
            response = urllib.request.urlopen(req, context=ctx, timeout=300)
            content_type = response.headers.get('Content-Type', '')
            raw_data = response.read()
            
            if 'pdf' in url.lower() or 'pdf' in content_type.lower():
                doc = fitz.open(stream=raw_data, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
            else:
                html_text = raw_data.decode('utf-8', errors='ignore')
                parser = SimpleHTMLParser()
                parser.feed(html_text)
                text = parser.get_text()
                
            docs_metadata[url] = text
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            docs_metadata[url] = f"Error fetching content: {e}"
            failed_citations.append(url)

    if len(failed_citations) > 0:
        raise Exception(f"Failed to fetch citation source(s): {', '.join(failed_citations)}. Process forcefully stopped before LLM evaluation.")

    # Save to local JSON since FAISS is not universally available here
    index_path = f"fetched_texts_{state.get('document_id', 'tmp')}.json"
    with open(index_path, 'w') as f:
        json.dump(docs_metadata, f)
        
    print(f"Citation texts saved to {index_path}.")
    return {**state, "vector_store_path": index_path}

def extract_claims(state: AuditState) -> AuditState:
    doc_text = state.get("document_text", "")
    print("Extracting claims from text via LLM...")
    
    prompt = (
        "You are an expert fact-checker. Extract EVERY SINGLE factual sentence, statistic, or verifiable line from the text below.\n"
        "CRITICAL INSTRUCTION: You MUST quote the facts EXACTLY as they appear in the original text, word-for-word. Do not paraphrase. "
        "Extract as many sentences as possible so that the majority of the report is converted into verifiable claims.\n"
        "Return ONLY a valid JSON object with a single key 'claims' containing a list of these exact-match strings.\n\n"
        f"Text:\n{doc_text[:8000]}"
    )
    
    try:
        from .state import stream_status
        import re
        job_id = state.get("document_id", "tmp")
        if job_id not in stream_status:
            stream_status[job_id] = {}
            
        full_response_text = ""
        for chunk in llm.stream([HumanMessage(content=prompt)]):
            full_response_text += chunk.content
            
            # Progressively parse out quotes that look like claims mid-flight
            matches = re.findall(r'"([^"]+)"', full_response_text)
            claims_so_far = [m for m in matches if m.lower() != "claims" and len(m) > 15]
            
            # Format and surface chunks to the frontend polling memory instantly
            temp_claims = []
            for i, c in enumerate(claims_so_far):
                temp_claims.append({
                    "id": f"temp_{i}", 
                    "text": c, 
                    "status": "unverified", 
                    "confidence": 0, 
                    "evidence": [], 
                    "human_review_needed": False
                })
            stream_status[job_id]["claims"] = temp_claims
            
        # Parse cleanly at the end
        parsed = _parse_llm_json(full_response_text)
        raw_claims = parsed.get("claims", [])
        if not raw_claims and len(parsed) > 0 and isinstance(list(parsed.values())[0], list):
            raw_claims = list(parsed.values())[0]
            
        if not isinstance(raw_claims, list):
            raw_claims = [str(raw_claims)]
            
    except Exception as e:
        print(f"Error extracting claims: {e}")
        raw_claims = ["Failed to extract claims due to LLM error."]
        
    claims = []
    for c in raw_claims:
        claims.append({
            "id": str(uuid.uuid4()), 
            "text": str(c), 
            "status": None, 
            "confidence": None, 
            "evidence": [], 
            "human_review_needed": False
        })
        
    if not claims: # Fallback
         claims = [{"id": str(uuid.uuid4()), "text": "No claims extracted", "status": None, "confidence": None, "evidence": [], "human_review_needed": False}]
         
    return {**state, "claims": claims}

def generate_search_queries(state: AuditState) -> AuditState:
    print("Generating queries for claims...")
    queries = {}
    for claim in state.get("claims", []):
        text = claim["text"]
        # In a real scenario we'd use LLM to generate keywords, but doing it via string manip for speed:
        queries[claim["id"]] = [text] 
    return {**state, "queries": queries}

def retrieve_evidence(state: AuditState) -> AuditState:
    print("Retrieving evidence from fetched texts...")
    evidence = {}
    vector_store_path = state.get("vector_store_path")
    
    texts_db = {}
    if vector_store_path and vector_store_path.endswith(".json"):
        import json
        try:
             with open(vector_store_path, 'r') as f:
                 texts_db = json.load(f)
        except Exception as e:
             print(f"Failed to load index json: {e}")
    
    for claim_id, claim_queries in state.get("queries", {}).items():
        query = claim_queries[0]
        
        ev_list = []
        if texts_db:
             import re
             # Extract significant keywords from query
             keywords = [w.lower() for w in re.sub(r'[^a-zA-Z0-9\s]', '', query).split() if len(w) > 3][:6]
             
             for url, text in texts_db.items():
                  # A rapid sliding window chunk matcher fallback for FAISS
                  best_score = 0
                  best_chunk = "Text unavailable due to parsing bounds."
                  
                  # Create chunks
                  chunk_size = 500
                  chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - 100)]
                  
                  for chunk in chunks:
                      score = sum(1 for kw in keywords if kw in chunk.lower())
                      if score > best_score:
                          best_score = score
                          best_chunk = chunk
                          
                  if best_score > 0:
                      ev_list.append({
                          "source": url,
                          "content": best_chunk.strip() + "...",
                          "score": best_score / max(1, len(keywords)) + 0.1
                      })
                  else:
                       ev_list.append({
                          "source": url,
                          "content": text[:500].strip() + "...",
                          "score": 0.1
                       })
             evidence[claim_id] = ev_list
        else:
             # Fallback mock retrieval
             evidence[claim_id] = [{"source": "Mock DB", "content": f"Found mock evidence related to: {query}", "score": 0.85}]
             
    return {**state, "retrieved_evidence": evidence}

def rerank_evidence(state: AuditState) -> AuditState:
    print("Reranking evidence...")
    evidence = state.get("retrieved_evidence", {})
    for k, v in evidence.items():
        evidence[k] = sorted(v, key=lambda x: x.get("score", 0), reverse=True)
    return {**state, "retrieved_evidence": evidence}

def verify_claim_vs_evidence(state: AuditState) -> AuditState:
    print("Verifying claims against evidence via LLM...")
    results = {}
    
    for claim in state.get("claims", []):
        c_id = claim["id"]
        ev_list = state.get("retrieved_evidence", {}).get(c_id, [])
        claim_text = claim["text"]
        
        if not ev_list:
            results[c_id] = {"status": "unverified", "reasoning": "No evidence found."}
            continue
            
        combined_evidence = "\\n".join([f"- {ev['content']}" for ev in ev_list])
        
        prompt = (
            "You are an expert fact-checker.\n"
            f"Claim: {claim_text}\n"
            f"Evidence:\n{combined_evidence}\n\n"
            "Does the evidence completely support, contradict, or fail to provide enough information for the claim?\n"
            "Respond ONLY with a valid JSON object containing 'reasoning' (string explaining why), 'status' (string, MUST be exactly 'verified', 'unverified', or 'contradicted'), and 'confidence' (float between 0.0 and 1.0)."
        )
        
        try:
             from .state import stream_status
             job_id = state.get("document_id", "tmp")
             if job_id not in stream_status:
                 stream_status[job_id] = {}
                 
             full_reasoning_text = ""
             for chunk in llm.stream([HumanMessage(content=prompt)]):
                 full_reasoning_text += chunk.content
                 
                 # Sync the reasoning text natively into the claim object array
                 if "claims" not in stream_status[job_id] or not stream_status[job_id]["claims"]:
                     stream_status[job_id]["claims"] = list(state.get("claims", []))
                     
                 for idx, stream_claim in enumerate(stream_status[job_id]["claims"]):
                     if stream_claim["id"] == c_id:
                         stream_status[job_id]["claims"][idx]["reasoning"] = full_reasoning_text
                         
                         # Dynamically guess the status for UI coloring while streaming
                         lower_rt = full_reasoning_text.lower()
                         if '"verified"' in lower_rt or 'status": "verified"' in lower_rt:
                             stream_status[job_id]["claims"][idx]["status"] = "verified"
                         elif '"contradicted"' in lower_rt or 'status": "contradicted"' in lower_rt:
                             stream_status[job_id]["claims"][idx]["status"] = "contradicted"
                         break
                 
             parsed = _parse_llm_json(full_reasoning_text)
             status = parsed.get("status", "unverified").lower()
             if status not in ["verified", "unverified", "contradicted"]:
                 status = "unverified"
             reasoning = parsed.get("reasoning", "LLM reasoning parsing failed.")
             confidence = float(parsed.get("confidence", 0.0))
        except Exception as e:
             print(f"Error during verification: {e}")
             status = "unverified"
             reasoning = "Error during LLM verification."
             confidence = 0.0
             
        results[c_id] = {"status": status, "reasoning": reasoning, "confidence": confidence}
        
    return {**state, "verification_results": results}

def score_confidence(state: AuditState) -> AuditState:
    print("Scoring confidence...")
    claims = list(state.get("claims", []))
    verification_results = state.get("verification_results", {})
    evidence_dict = state.get("retrieved_evidence", {})
    
    for i, claim in enumerate(claims):
        res = verification_results.get(claim["id"])
        c_evidence = evidence_dict.get(claim["id"], [])
        if res:
            claims[i]["status"] = res["status"]
            
            # Use dynamic confidence if available, else derive from status
            dyn_conf = res.get("confidence", 0.0)
            if float(dyn_conf) > 0.0:
                claims[i]["confidence"] = float(dyn_conf)
            else:
                if res["status"] == "verified":
                    claims[i]["confidence"] = 0.90
                elif res["status"] == "contradicted":
                    claims[i]["confidence"] = 0.85
                else:
                    claims[i]["confidence"] = 0.40
            
            if res["status"] == "verified":
                claims[i]["human_review_needed"] = False
            elif res["status"] == "contradicted":
                claims[i]["human_review_needed"] = False
            else:
                claims[i]["human_review_needed"] = True
                
            # Keep reasoning in state/claims if we want to show it in UI
            claims[i]["reasoning"] = res["reasoning"]
            claims[i]["evidence"] = c_evidence
            
    return {**state, "claims": claims}

def route_ambiguous_cases_to_human_review(state: AuditState) -> AuditState:
    print("Routing ambiguous cases...")
    needs_review = [c for c in state.get("claims", []) if c.get("human_review_needed")]
    print(f"Routed {len(needs_review)} claims to manual review.")
    return state

def aggregate_final_audit_report(state: AuditState) -> AuditState:
    print("Aggregating final report...")
    claims = state.get("claims", [])
    report = {
        "document_id": state.get("document_id"),
        "total_claims": len(claims),
        "verified": sum(1 for c in claims if c.get("status") == "verified"),
        "contradicted": sum(1 for c in claims if c.get("status") == "contradicted"),
        "needs_review": sum(1 for c in claims if c.get("human_review_needed")),
        "details": claims,
        "document_text": state.get("document_text", "")
    }
    return {**state, "final_report": report}
