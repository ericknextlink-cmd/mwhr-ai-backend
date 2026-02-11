from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Any, Dict
from app.services.pdf_analysis_service import pdf_analysis_service
from app.services.pdf_extract_local import extract_text_from_pdf_url
from app.services.chat_service import chat_service
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

class AnalyzeDocumentRequest(BaseModel):
    document_url: HttpUrl
    document_type: str
    strategy: Optional[str] = "hi_res"
    use_ocr: Optional[bool] = True
    extract_tables: Optional[bool] = True
    extract_forms: Optional[bool] = False
    languages: Optional[List[str]] = ["eng"]
    application_company_name: Optional[str] = None

class ExtractDocumentRequest(BaseModel):
    document_url: HttpUrl
    use_ocr: Optional[bool] = True

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.SERVICE_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_document(request: AnalyzeDocumentRequest):
    try:
        result = await pdf_analysis_service.analyze_document(
            document_url=str(request.document_url),
            document_type=request.document_type,
            strategy=request.strategy,
            use_ocr=request.use_ocr,
            extract_tables=request.extract_tables,
            extract_forms=request.extract_forms,
            languages=request.languages,
            application_company_name=request.application_company_name,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract", dependencies=[Depends(verify_api_key)])
async def extract_document(request: ExtractDocumentRequest):
    try:
        text = await extract_text_from_pdf_url(
            document_url=str(request.document_url),
            use_ocr=request.use_ocr
        )
        return {"extracted_text": text, "success": bool(text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    try:
        response = await chat_service.generate_response(request.message, request.history)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))