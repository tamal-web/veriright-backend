from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
import uuid
import sqlite3
import json
from datetime import datetime

# Initialize FastAPI
app = FastAPI(title="VeriRight Backend Logic", description="API to ingest docs and start LangGraph agent")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Typically restricted in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    conn = sqlite3.connect("audits.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS audits (
            job_id TEXT PRIMARY KEY,
            filename TEXT,
            status TEXT,
            result TEXT,
            claims TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Dummy in-memory store for active job status polling
jobs: Dict[str, Dict[str, Any]] = {}

def process_document(job_id: str, filename: str, document_text: str):
    from app.agent.graph import app_graph
    
    # Initialize state
    initial_state = {
        "document_id": job_id,
        "document_text": document_text,
        "document_metadata": {},
        "claims": [],
        "current_claim_index": 0,
        "queries": {},
        "retrieved_evidence": {},
        "verification_results": {},
        "final_report": None
    }
    
    conn = get_db()
    
    try:
        # Run graph iteratively
        print(f"Starting job {job_id}")
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["document_text"] = document_text
        jobs[job_id]["filename"] = filename
        jobs[job_id]["claims"] = []
        
        conn.execute("UPDATE audits SET status = ? WHERE job_id = ?", ("processing", job_id))
        conn.commit()
        
        final_state = initial_state
        
        for output in app_graph.stream(initial_state):
             for node_name, node_state in output.items():
                  jobs[job_id]["status"] = f"running_{node_name}"
                  if "claims" in node_state:
                       jobs[job_id]["claims"] = node_state["claims"]
                  if "document_text" in node_state:
                       jobs[job_id]["document_text"] = node_state["document_text"]
                  if "extracted_citations" in node_state:
                       jobs[job_id]["citations"] = node_state["extracted_citations"]
                  
                  # Persist mid-flight claims temporarily so disconnected reloads don't lose them
                  conn.execute("UPDATE audits SET claims = ? WHERE job_id = ?", (json.dumps(node_state.get("claims", [])), job_id))
                  conn.commit()
                  
                  final_state = node_state
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = final_state.get("final_report", {})
        jobs[job_id]["claims"] = final_state.get("claims", [])
        
        conn.execute(
            "UPDATE audits SET status = ?, result = ?, claims = ? WHERE job_id = ?",
            ("completed", json.dumps(final_state.get("final_report", {})), json.dumps(final_state.get("claims", [])), job_id)
        )
        conn.commit()
        
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        
        conn.execute(
            "UPDATE audits SET status = ?, result = ? WHERE job_id = ?",
            ("failed", json.dumps({"error": str(e)}), job_id)
        )
        conn.commit()
    finally:
        conn.close()

@app.post("/api/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    filename = file.filename or "unknown.txt"
    jobs[job_id] = {"status": "queued"}
    
    conn = get_db()
    conn.execute(
        "INSERT INTO audits (job_id, filename, status) VALUES (?, ?, ?)",
        (job_id, filename, "queued")
    )
    conn.commit()
    conn.close()
    
    # Parse text based on file type
    content = await file.read()
    
    text = ""
    if filename.lower().endswith(".pdf"):
        try:
            import fitz # PyMuPDF
            doc = fitz.open(stream=content, filetype="pdf")
            for page in doc:
                text += page.get_text()
        except Exception as e:
            print(f"Error parsing PDF: {e}")
            text = content.decode('utf-8', errors="ignore")[:3000]
    else:
        text = content.decode('utf-8', errors="ignore")[:5000]
    
    background_tasks.add_task(process_document, job_id, filename, text)
    
    return {"job_id": job_id, "message": "Document uploaded successfully"}

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    base_data = None
    if job_id in jobs:
        base_data = dict(jobs[job_id])
    else:
        # Check db
        conn = get_db()
        row = conn.execute("SELECT status, result, claims, filename FROM audits WHERE job_id = ?", (job_id,)).fetchone()
        conn.close()
        
        if not row:
            return {"status": "not_found"}
            
        base_data = {
            "status": row["status"],
            "result": json.loads(row["result"]) if row["result"] else None,
            "claims": json.loads(row["claims"]) if row["claims"] else None,
            "citations": json.loads(row["result"]).get("extracted_citations", []) if row["result"] else [],
            "filename": row["filename"],
            "source": "db"
        }
        
    from app.agent.state import stream_status
    if job_id in stream_status and "claims" in stream_status[job_id]:
        if base_data.get("status") not in ["completed", "failed"]:
            base_data["claims"] = stream_status[job_id]["claims"]
            
    return base_data

@app.get("/api/audits")
def list_audits():
    conn = get_db()
    # Fetch latest 20 audits
    rows = conn.execute("SELECT job_id, filename, status, result, created_at FROM audits ORDER BY created_at DESC LIMIT 20").fetchall()
    conn.close()
    
    audits = []
    for r in rows:
        result_data = {}
        try:
            if r["result"]:
                result_data = json.loads(r["result"])
        except:
             pass
             
        # determine score/status derived UI variables
        ui_score = "Pending"
        ui_status = "review" # review, success, warning, error
        
        if r["status"] == "completed":
            total = result_data.get("total_claims", 1)
            verified = result_data.get("verified", 0)
            contradicted = result_data.get("contradicted", 0)
            
            if total > 0:
                verified_pct = (verified / total) * 100
                if verified_pct >= 80:
                    ui_score = "Verified"
                    ui_status = "success"
                elif contradicted > 0:
                     ui_score = "High Risk"
                     ui_status = "warning"
                else:
                    ui_score = "Plausible"
                    ui_status = "review"
            else:
                 ui_score = "No Claims"
                 ui_status = "warning"
        elif r["status"] == "failed":
            ui_score = "Failed"
            ui_status = "error"
            
        audits.append({
            "id": r["filename"],
            "job_id": r["job_id"],
            "score": ui_score,
            "time": r["created_at"],
            "status": ui_status
        })
        
    return {"audits": audits}
