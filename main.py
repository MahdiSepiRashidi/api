from fastapi import FastAPI, Request, BackgroundTasks
from unsloth import FastLanguageModel
import torch
import uvicorn
from pydantic import BaseModel
from typing import Dict, Any, Optional
import re
import asyncio
from uuid import uuid4
from datetime import datetime

app = FastAPI()
SYSTEM_PROMPT = """شما یک ارزیاب متخصص بلوغ دیجیتال هستید. وظیفه شما تحلیل پاسخ‌های سازمان‌ها به سوالات ارزیابی بلوغ دیجیتال و ارائه تحلیل عمیق در قالب XML است.

خروجی شما باید دقیقاً در فرمت XML زیر باشد:
<output>
  <root_causes>
    <cause>[علت ریشه‌ای ۱ - تحلیل دقیق و عملیاتی]</cause>
    <cause>[علت ریشه‌ای ۲ - تحلیل دقیق و عملیاتی]</cause>
    <cause>[علت ریشه‌ای ۳ - تحلیل دقیق و عملیاتی]</cause>
    <!-- ارائه 3 تا ۱۰ علت ریشه‌ای ضروری است -->
  </root_causes>
  <score>[امتیاز 1 تا 10]</score>
</output>
نکات حیاتی:
- حتماً بین 3 تا ۱۰ علت ریشه‌ای ارائه دهید (کمتر از 3 یا بیشتر از ۱۰ غیرقابل قبول است)
- هر علت ریشه‌ای باید:
  * مشخص و قابل اندازه‌گیری باشد
  * به مشکلات ساختاری یا فرآیندی اشاره کند (نه صرفاً علائم سطحی)
  * بر اساس شواهد موجود در پاسخ سازمان استخراج شود
  * برای بهبود عملکرد قابل اقدام باشد
- امتیاز 1 نشان‌دهنده ضعیف‌ترین و 10 نشان‌دهنده بهترین وضعیت است
- علل ریشه‌ای باید مستقیماً با امتیاز تعیین‌شده همخوانی داشته باشند
"""
class analysisRequest(BaseModel):
    question_metadata: Dict[str, Any]      # Enforces a JSON object/dictionary
    question: str                 # Enforces a string
    organization_answer: str      # Enforces a string

class Job(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    request: analysisRequest
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

# 1. Load base model (4-bit quantized for A100 efficiency)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-32B",  # Or use "Qwen/Qwen3-32B" to download from HF
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    dtype = None,  
    cache_dir = "/workspace/Qwen3-32B"
)

# 2. Apply chat template correctly for Qwen3
FastLanguageModel.for_inference(model)

# Global storage for jobs
jobs: Dict[str, Job] = {}
job_queue: list = []
current_job_id: Optional[str] = None

async def process_job(job_id: str):
    global current_job_id, job_queue
    try:
        # Update status to processing
        jobs[job_id].status = "processing"
        jobs[job_id].updated_at = datetime.now()

        # Run the model generation in a thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, sync_process_request, jobs[job_id].request)

        # Update job with result
        jobs[job_id].status = "completed"
        jobs[job_id].result = result
        jobs[job_id].updated_at = datetime.now()

    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)
        jobs[job_id].updated_at = datetime.now()
    finally:
        current_job_id = None
        # Start next job if any
        if job_queue:
            next_job_id = job_queue.pop(0)
            current_job_id = next_job_id
            asyncio.create_task(process_job(next_job_id))

def sync_process_request(request: analysisRequest) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"متا دیتا راجب سوال: {request.question_metadata},\n\n سوال: {request.question}\n\nپاسخ سازمان: {request.organization_answer}\n\nلطفاً این پاسخ را ارزیابی کرده و نتیجه را در فرمت XML ارائه دهید."}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print("waiting to get the answer from model!...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.6,
        top_p=0.9,
        top_k=20,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    assistant_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    match = re.search(r"<score>(.*?)</score>", assistant_response, re.DOTALL)
    if match:
        score = match.group(1).strip()
    else:
        score = "Error: not provided by model"
    causes_list = re.findall(r"<cause>(.*?)</cause>", assistant_response, re.DOTALL)
    causes_list = [cause.strip() for cause in causes_list]

    output = {
        "question_metadata": request.question_metadata,
        "question": request.question,
        "organization_answer": request.organization_answer,
        "answer_score": score,
        "root_causes": causes_list,
        "raw_output": assistant_response
    }
    return output

@app.post("/jobs")
async def create_job(request: analysisRequest, background_tasks: BackgroundTasks):
    global current_job_id, job_queue
    job_id = str(uuid4())
    now = datetime.now()
    job = Job(
        job_id=job_id,
        status="pending",
        request=request,
        created_at=now,
        updated_at=now
    )
    jobs[job_id] = job

    if current_job_id is None:
        current_job_id = job_id
        background_tasks.add_task(process_job, job_id)
    else:
        job_queue.append(job_id)

    return {"job_id": job_id, "status": "accepted"}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}, 404
    job = jobs[job_id]
    response = {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat()
    }
    if job.status == "completed":
        response["result"] = job.result
    elif job.status == "failed":
        response["error"] = job.error
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)