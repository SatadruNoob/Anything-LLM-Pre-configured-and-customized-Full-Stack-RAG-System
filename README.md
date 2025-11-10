# 

# Internal RAG Pipeline with Distributed Colab Preprocessing and AnythingLLM

A production-lean deployment that ingests internal PDFs, denoises them at scale in Google Colab, and automatically embeds them into AnythingLLM (Docker) using Mistral 7B, LanceDB, and Nomic Embed Text v1. The pipeline supports GUI and REST-based ingestion, fault routing for large files, and is designed for continuous embedding automation. [1][2][3]

## Architecture overview

- Source PDFs are uploaded from client machines to Google Drive via a PowerShell uploader.  
- Cleaning and OCR-safe denoising run in parallel across 4 Colab notebooks to accelerate 7k+ document throughput.  
- Cleaned PDFs are auto-dumped to a designated client folder, then pushed to AnythingLLM via REST for vectorization using LanceDB and Nomic Embed Text v1. [3][4][5]
- Large-file failures are routed to a local “manual-review” folder for GUI-based ingestion.  
- Core RAG served by AnythingLLM (Docker) with Mistral 7B as the LLM. [2][6]

## Key components

- AnythingLLM (Docker) for RAG and workspace management. [2][1]
- Mistral 7B for responses; LanceDB as the vector database; Nomic Embed Text v1 for embeddings. [6][4][5]
- PowerShell uploader for Drive; Google Colab for distributed cleaning. [3]

## Images

- System diagram: Place a high-level architecture diagram image here (e.g., /docs/architecture.png).  
- Colab worker: Place a screenshot illustrating the batched Colab runtime (e.g., /docs/colab_batch.png).  
- AnythingLLM workspace: Place a screenshot of the workspace config (e.g., /docs/anythingllm_workspace.png). [3]

## Features

- RAG for internal employees with GUI and REST ingestion. [3]
- Distributed preprocessing across 4 Colab notebooks to accelerate >7k docs.  
- Automated REST push to AnythingLLM; fallback manual embedding for large files. [3]
- Dockerized deployment for easy backup, updates, and multi-platform support. [2][1]

## Prerequisites

- Docker and Docker Compose for AnythingLLM. [2]
- Google account with Drive access for staging PDFs.  
- Google Colab for preprocessing notebooks.  
- API key and base URL for AnythingLLM REST API. [7][3]
- Optional: Ollama or remote inference endpoint if running Mistral 7B locally. [8][6]

***

## RAG system setup

AnythingLLM was selected for its all-in-one RAG capabilities, flexible LLM/vector integrations, Agents, and straightforward Docker deployment for organizations. It supports workspace management, API access, and GUI upload out of the box. [1][3]

- Docker quickstart (example docker-compose.yml):
```yaml
version: '3.8'
services:
  anythingllm:
    image: mintplexlabs/anythingllm:latest
    container_name: anythingllm
    ports:
      - "3001:3001"
    environment:
      - JWT_SECRET=change_me
      - STORAGE_DIR=/app/storage
      - VECTOR_DB=lance
      - EMBEDDING_MODEL=nomic-embed-text-v1
      - LLM_PROVIDER=ollama
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - DEFAULT_LLM=mistral
    volumes:
      - ./storage:/app/storage
      - ./data:/app/data
    restart: unless-stopped
```
This sample shows LanceDB as the vector store, Nomic-Embed-Text for embeddings, and Mistral (via Ollama) as the LLM. Adjust to your infra. [2][4][5][8][3]

***

## PowerShell uploader to Google Drive

A PowerShell script uploads source PDFs from a local source folder to a Drive folder, enabling Colab access. This approach minimizes friction for non-technical employees.

- Requirements:
  - A Drive folder with a known Folder ID.
  - OAuth or Service Account flow via Google APIs (or rclone for simplicity).

- Example using rclone backend:
```powershell
param(
  [Parameter(Mandatory=$true)][string]$SourcePath,
  [Parameter(Mandatory=$true)][string]$RemoteName,       # e.g., "gdrive:"
  [Parameter(Mandatory=$true)][string]$RemoteFolderPath  # e.g., "company-ingest/raw"
)

# Ensure rclone is installed and 'RemoteName' is configured.
# Upload PDFs only
Get-ChildItem -Path $SourcePath -Filter *.pdf -Recurse | ForEach-Object {
    $relative = $_.FullName.Substring($SourcePath.Length).TrimStart('\')
    $dest = Join-Path $RemoteFolderPath $relative
    $dest = $dest -replace '\\','/'

    Write-Host "Uploading $($_.FullName) -> $RemoteName$dest"
    rclone copy $_.FullName "$RemoteName$dest" --create-empty-src-dirs --progress --transfers=8 --checkers=16 --drive-chunk-size=128M
}
```
This script pushes PDFs into a consistent Drive hierarchy for Colab workers. [3]

***

## PDF denoising and cleaning in Colab

Cleaning runs in Google Colab using a denoising pipeline (e.g., pikepdf for structural fixes, PyMuPDF for page ops, OCR with Tesseract if needed, and noise filtering via OpenCV for scanned images). Processing 7k+ documents is distributed across 4 notebooks for throughput. [3]

- Example Colab cell (Python) for page-level cleanup:
```python
!pip -q install pikepdf pymupdf opencv-python pytesseract pdf2image

import os, fitz, cv2, tempfile
from pdf2image import convert_from_path
import pytesseract
from pikepdf import Pdf, PdfError

SOURCE_DIR = "/content/drive/MyDrive/company-ingest/raw"
CLEAN_DIR = "/content/drive/MyDrive/company-ingest/clean"

os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_pdf(src_path, dst_path, ocr=True, dpi=300):
    try:
        # Structural repair
        with Pdf.open(src_path) as pdf:
            pdf.save(dst_path)
    except PdfError:
        dst_path = dst_path  # proceed with re-render

    # Re-render pages and apply denoise
    images = convert_from_path(src_path, dpi=dpi)
    cleaned_images = []
    for img in images:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name)
            mat = cv2.imread(tmp.name, cv2.IMREAD_GRAYSCALE)
            mat = cv2.fastNlMeansDenoising(mat, h=10, templateWindowSize=7, searchWindowSize=21)
            mat = cv2.threshold(mat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            cv2.imwrite(tmp.name, mat)
            cleaned_images.append(tmp.name)

    # Rebuild PDF (with optional OCR text layer)
    doc = fitz.open()
    for p in cleaned_images:
        pix = fitz.Pixmap(p)
        rect = fitz.Rect(0, 0, pix.width, pix.height)
        page = doc.new_page(width=pix.width, height=pix.height)
        page.insert_image(rect, filename=p)
        if ocr:
            text = pytesseract.image_to_string(p)
            page.insert_textbox(rect, text, fontsize=1)  # lightweight text layer
    doc.save(dst_path)
    doc.close()

for root, _, files in os.walk(SOURCE_DIR):
    for f in files:
        if f.lower().endswith(".pdf"):
            src = os.path.join(root, f)
            rel = os.path.relpath(src, SOURCE_DIR)
            out = os.path.join(CLEAN_DIR, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            try:
                clean_pdf(src, out)
                print("Cleaned:", rel)
            except Exception as e:
                print("Failed:", rel, e)
```

- Suggested denoising heuristics:
  - Fast non-local means for background speckle removal.
  - Otsu thresholding for binarization.
  - Optional OCR pass for scanned pages to ensure retrievability.

***

## Decentralized processing with 4 Colab notebooks

To accelerate processing of 7k+ PDFs, documents are partitioned into 4 batches and assigned to 4 notebooks across 3 Colab platforms, improving throughput substantially. Each notebook targets a distinct subfolder in Drive. [3]

- Example batch assignment snippet:
```python
import os, shutil

RAW_DIR = "/content/drive/MyDrive/company-ingest/raw"
BATCH_ROOT = "/content/drive/MyDrive/company-ingest/batches"
os.makedirs(BATCH_ROOT, exist_ok=True)

all_pdfs = []
for root, _, files in os.walk(RAW_DIR):
    for f in files:
        if f.lower().endswith(".pdf"):
            all_pdfs.append(os.path.join(root, f))

all_pdfs.sort()
batches = 4
for idx, path in enumerate(all_pdfs):
    b = idx % batches
    target_dir = os.path.join(BATCH_ROOT, f"batch-{b}")
    os.makedirs(target_dir, exist_ok=True)
    rel = os.path.relpath(path, RAW_DIR)
    dst = os.path.join(target_dir, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(path, dst)
```
Each Colab notebook points its SOURCE_DIR to a specific batch-n folder, and writes cleaned PDFs to a shared CLEAN_DIR.

***

## Automated dumping to client folder

Once cleaned, PDFs are synchronized to a designated client folder (network share or synced local path), enabling downstream ingestion. This can be done with rclone sync from Drive to the client. [3]

- Example periodic sync (PowerShell, scheduled task):
```powershell
param(
  [Parameter(Mandatory=$true)][string]$RemoteName,       # e.g., "gdrive:"
  [Parameter(Mandatory=$true)][string]$RemoteCleanPath,  # e.g., "company-ingest/clean"
  [Parameter(Mandatory=$true)][string]$LocalCleanPath    # e.g., "D:\RAG\clean"
)

rclone sync "$RemoteName$RemoteCleanPath" "$LocalCleanPath" --progress --transfers=16 --checkers=32 --drive-chunk-size=128M --delete-excluded
```

***

## AnythingLLM configuration (Docker)

The Docker edition was chosen for multi-platform integration, easier backups, and streamlined updates. It also aligns with enterprise deployment practices while preserving the desktop GUI option. [2][1]

- Core capabilities relevant here:
  - Multiple LLM providers and vector DBs.  
  - Workspaces for doc scoping and permissions.  
  - Full REST API for programmatic embedding and chat. [3][7]

Place a screenshot of the workspace and API key setup here (e.g., /docs/anythingllm_api.png). [3]

***

## Model and vector settings

- LLM: Mistral 7B for efficient, high-quality responses with strong benchmark performance and Apache 2.0 licensing. [6][9][10]
- Vector DB: LanceDB for local, high-performance vector storage compatible with modern embedding flows. [4]
- Embeddings: Nomic Embed Text v1 (and v1.5 option) supporting search_document and long-text handling parameters. [5][11][12]

If running via Ollama, ensure mistral and nomic-embed-text models are available to the embedding and inference layers. [8][4]

***

## REST ingestion to AnythingLLM

Clean PDFs in the client folder are tunneled to AnythingLLM via REST for vector embedding, automating ingestion from the client machine into the target workspace. AnythingLLM exposes an API for managing workspaces and documents programmatically. [3][7]

- Example Python ingestion client:
```python
import os
import time
import requests

ANY_BASE = os.getenv("ANY_BASE_URL", "http://localhost:3001/api")
ANY_KEY  = os.getenv("ANY_API_KEY", "replace_me")
WORKSPACE_ID = os.getenv("ANY_WORKSPACE_ID", "internal-knowledge")
CLEAN_DIR = os.getenv("CLEAN_DIR", r"D:\RAG\clean")
HEADERS = {"Authorization": f"Bearer {ANY_KEY}"}

def upload_doc(path):
    url = f"{ANY_BASE}/workspaces/{WORKSPACE_ID}/documents"
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, "application/pdf")}
        resp = requests.post(url, headers=HEADERS, files=files, timeout=120)
    if resp.status_code == 200:
        return True, resp.json()
    return False, resp.text

def watch_and_ingest():
    for root, _, files in os.walk(CLEAN_DIR):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                full = os.path.join(root, fn)
                ok, res = upload_doc(full)
                print("OK" if ok else "FAIL", fn, res)
                time.sleep(0.5)

if __name__ == "__main__":
    watch_and_ingest()
```
Refer to AnythingLLM’s API docs for exact endpoints and auth flows. [7][3]

***

## Handling large-file failures

Very large PDFs may fail in automated embedding. Those are routed to a designated local folder for manual upload through the AnythingLLM GUI. This hybrid flow ensures no document is blocked by API constraints. [3]

- Example Python router:
```python
import os, shutil

FAILED_DIR = r"D:\RAG\failed_manual"

def route_failure(path):
    os.makedirs(FAILED_DIR, exist_ok=True)
    dst = os.path.join(FAILED_DIR, os.path.basename(path))
    shutil.move(path, dst)
    print("Routed to manual:", dst)
```

- Manual GUI upload:
  - Open AnythingLLM, select the workspace, and use the Documents UI to add the large PDF directly. [3]

***

## Continuous embedding agent (next step)

The next step is to build an Agent that monitors cloud storage for new files and triggers automatic embedding into AnythingLLM. This can be realized with a lightweight scheduler plus the AnythingLLM API. [3][7]

- Example Python watcher (Drive-to-AnythingLLM):
```python
import time
from datetime import datetime, timedelta

POLL_SECONDS = 60

def list_new_drive_files(since_dt):
    # TODO: Implement Drive API list with modifiedTime > since_dt
    return []

def embed_new_files(file_list):
    for f in file_list:
        # Download to local staging, then call upload_doc(...)
        pass

def run_agent():
    last_check = datetime.utcnow() - timedelta(minutes=5)
    while True:
        new_files = list_new_drive_files(last_check)
        embed_new_files(new_files)
        last_check = datetime.utcnow()
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    run_agent()
```
Tie this into webhooks or Cloud Functions for near-real-time triggers as your governance allows. [3]

***

## Local LanceDB embedding example

If you maintain a parallel LanceDB store (for analytics or validation), this snippet shows how to generate embeddings with Nomic Embed Text via Ollama + LanceDB’s registry. [4]

```python
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

db = lancedb.connect("C:/rag/lance")
func = get_registry().get("ollama").create(name="nomic-embed-text")  # default model name
class Doc(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()

table = db.create_table("docs", schema=Doc, mode="create")
table.add([{"text": "Example cleaned content"}])
res = table.search("query about example").limit(1).to_pydantic(Doc)[0]
print(res.text)
```
Adjust to your infra and indexing strategy. [4]

***

## Operational tips

- Use 4 Colab notebooks for balanced throughput; pin each to a batch-n folder.  
- Keep a rolling log of ingestion outcomes; route oversized failures immediately for manual GUI ingestion. [3]
- Back up AnythingLLM storage and LanceDB directories regularly; prefer bind mounts in Docker. [2]

***

