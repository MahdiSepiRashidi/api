from fastapi import FastAPI, Request
from unsloth import FastLanguageModel
import torch
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

# 1. Load base model (4-bit quantized for A100 efficiency)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-32B",  # Or use "Qwen/Qwen3-32B" to download from HF
    max_seq_length = 4096,
    load_in_4bit = True,
    load_in_8bit = False,
    dtype = None,  
    cache_dir = "/workspace/Qwen3-32B"
)

# 2. Apply chat template correctly for Qwen3
FastLanguageModel.for_inference(model)

@app.post("/question_analysis")
async def chat_completion(request: analysisRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"متا دیتا راجب سوال: {request.question_metadata},\n\n سوال: {request.question}\n\nپاسخ سازمان: {request.organization_answer}\n\nلطفاً این پاسخ را ارزیابی کرده و نتیجه را در فرمت XML ارائه دهید."}
    ]
    
    # Use Unsloth's built-in chat template handling
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Critical: adds assistant start token
        enable_thinking=True
        
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print("waiting to get the answer from model!...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.6,    # Qwen3 recommendation for reasoning
        top_p=0.9,
        top_k=20,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
# Decode only the new (assistant) tokens, excluding the input prompt
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]  # Slice from input length
    assistant_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    match = re.search(r"<score>(.*?)</score>", assistant_response, re.DOTALL)
    if match:
        score = match.group(1).strip()
    else:
        score = "Error: not provided by model"
    causes_list = re.findall(r"<cause>(.*?)</cause>", assistant_response, re.DOTALL)

    # Clean up whitespace (optional but recommended)
    causes_list = [cause.strip() for cause in causes_list]
    output = {
        "question_metadata": request.question_metadata,
        "question": request.question,
        "organization_answer": request.organization_answer,
        "answer_score": score,
        "root_causes": causes_list,
        "raw_output" : assistant_response
    }

    print(output)
    
    return output

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)