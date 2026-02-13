from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel
from typing import Dict, Any
import re

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



@app.post("/question_analysis")
async def chat_completion(request: analysisRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"متا دیتا راجب سوال: {request.question_metadata},\n\n سوال: {request.question}\n\nپاسخ سازمان: {request.organization_answer}\n\nلطفاً این پاسخ را ارزیابی کرده و نتیجه را در فرمت XML ارائه دهید."}
    ]


    full_output = """provided a response from an organization regarding their digital vision and strategy communication. The task is to analyze the root causes and assign a score between 1-10.

First, I need to understand the context. The question is about the clarity and communication of the digital vision and strategy. The best answer mentions clear internal and external communication where stakeholders can describe it well. The worst answer is the opposite. The organization's response states that the vision is partially defined but communication is inconsistent across departments, leading to misaligned priorities.

Now, identifying root causes. The response points out that communication isn't unified, stakeholders don't share a common understanding, and there's misalignment in translating strategy into priorities. These are structural issues. 

First root cause: Inconsistent internal communication. The answer mentions that different departments aren't aligned, so the communication strategy isn't cohesive. This leads to varying interpretations.

Second root cause: Lack of stakeholder engagement. If stakeholders don't have a shared understanding, it's likely because they weren't involved in the strategy development or communication process.

Third root cause: Poor translation into actionable priorities. Even if the vision exists, if it's not effectively translated into departmental goals, execution will be inconsistent.

Fourth root cause: Absence of feedback mechanisms. Without feedback loops, the organization can't identify misalignments or adjust the strategy accordingly.

Fifth root cause: Inadequate training or resources. If departments aren't equipped to implement the strategy, it leads to misalignment.

Now, scoring. The best answer is a 10, worst is 1. The organization's answer is between the two. They have a partial vision but significant issues in communication and alignment. Considering the root causes, this seems like a mid-level score. Maybe a 4 or 5. Let's go with 5 because they have a partial vision but major gaps in execution and communication.
</think>

<output>
  <root_causes>
    <cause>عدم یکپارچگی اطلاع‌رسانی چشم‌انداز دیجیتال در بین بخش‌های سازمانی منجر به تفاوت در برداشت اهداف می‌شود</cause>
    <cause>عدم مشارکت فعال ذی‌نفعان در فرآیند تعریف و انتقال استراتژی دیجیتال</cause>
    <cause>عدم ترجمه واضح چشم‌انداز به اولویت‌های عملیاتی قابل اجرا در تمام واحدها</cause>
    <cause>عدم وجود مکانیزم‌های بازخورد برای هماهنگی بین استراتژی و عملکردهای واقعی</cause>
    <cause>عدم تطبیق استراتژی دیجیتال با ساختار سازمانی فعلی و فرهنگ کاری</cause>
  </root_causes>
  <score>5</score>
</output>"""
    
    # Decode only the new tokens (skip input prompt)
    match = re.search(r"<score>(.*?)</score>", full_output, re.DOTALL)
    if match:
        score = match.group(1).strip()
    else:
        score = "Error: not provided by model"
    causes_list = re.findall(r"<cause>(.*?)</cause>", full_output, re.DOTALL)

    # Clean up whitespace (optional but recommended)
    causes_list = [cause.strip() for cause in causes_list]
    
    return {
        "question_metadata": request.question_metadata,
        "question": request.question,
        "organization_answer": request.organization_answer,
        "answer_score": score,
        "root_causes": causes_list
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)