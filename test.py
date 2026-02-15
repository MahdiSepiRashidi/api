from fastapi import FastAPI, Request, BackgroundTasks
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
    assistant_response = """vided a response from an organization regarding their investment model's flexibility in financial support. The question was about how much their investment model allows for quick and dynamic financial adjustments.

First, I need to compare the organization's answer to the best and worst answers given. The best answer mentions initial investments in incubation periods, moving towards revenue-generating areas, evaluating at milestones, and having easy access to additional funding if needed. The worst answer relies on annual sales forecasts, annual reassessment, and only funds low-risk projects with guaranteed returns.

The organization's response states that funding is mainly based on periodic budgeting and sales forecasts but allows for mid-term reviews and resource reallocation for prioritized projects. However, the evaluation and continuation of investments aren't uniformly milestone-based across all projects.

Now, identifying root causes. The first point is reliance on periodic budgeting and sales forecasts. This suggests a lack of real-time adaptability. The second is inconsistent milestone-based evaluation, meaning some projects might not be assessed properly. Third, limited flexibility in reallocating resources, implying that even though there's some reallocation, it's not comprehensive or systematic.

The score should reflect these issues. Since the organization has some flexibility but lacks uniform processes and real-time adjustments, a score around 5-6 makes sense. Considering the best answer has a robust system and the worst is rigid, the organization is in the middle but leaning towards moderate. So, a score of 5 seems appropriate.
</think>

<output>
  <root_causes>
    <cause>اعتماد زیاد به بودجه‌ریزی دوره‌ای و پیش‌بینی فروش به جای سیستم‌های پویا برای واکنش به تغییرات بازار</cause>
    <cause>عدم اعمال یکنواخت ارزیابی مایلستون در تمام پروژه‌ها که منجر به عدم همگونی در مدیریت سرمایه می‌شود</cause>
    <cause>امکان بازنگری بین‌دوره‌ای و جابه‌جایی منابع فقط در برخی موارد و بدون چارچوب سیستماتیک</cause>
    <cause>عدم تامین مالی مجدد آسان در صورت نیاز به دلیل عدم وجود مکانیزم‌های پویا برای جذب سرمایه</cause>
  </root_causes>
  <score>5</score>
</output>"""
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