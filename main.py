from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import random
import os
import torch
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration

app = FastAPI()

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    print("Warning: 'static' folder not found. Logos will not load.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

questions_db = []
af3_model = None
af3_processor = None

af3_model_id = "nvidia/audio-flamingo-3-hf"
target_device = "cuda:0" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_system():
    global questions_db, af3_model, af3_processor
    
    if os.path.exists("audio.json"):
        with open("audio.json", "r") as f:
            questions_db = json.load(f)
    else:
        print("Warning: questions.json not found.")
        
    print(f"Loading {af3_model_id} onto {target_device}...")
    try:
        af3_processor = AutoProcessor.from_pretrained(af3_model_id)
        af3_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            af3_model_id, 
            device_map=target_device
        )
        print(f"Model loaded successfully to {target_device}. Ready to play!")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.get("/start_game")
def start_game():
    if not questions_db:
        raise HTTPException(status_code=500, detail="Questions database is empty.")
    sample_size = min(3, len(questions_db))
    return random.sample(questions_db, sample_size)


class InferRequest(BaseModel):
    audio_id: str
    question: str
    choices: list[str]

def infer_af3(question, choices, audio_file):
    if af3_model is None or af3_processor is None:
        return "Model not loaded properly."

    options_text = "\n".join([f"- {opt}" for opt in choices])
    prompt = (
        f"Question: {question}\n"
        f"Options:\n{options_text}\n\n"
        "Instruction: You must answer strictly by copying the exact text of the correct option. "
        "Do NOT answer with letters like A, B, C, or D. Output absolutely nothing else but the exact option text."
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": audio_file},
            ],
        }
    ]
    
    inputs = af3_processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(af3_model.device)

    outputs = af3_model.generate(**inputs, max_new_tokens=100)
    decoded_outputs = af3_processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    return decoded_outputs[0]

@app.post("/infer")
def infer(request: InferRequest):
    """API Endpoint to handle inference requests."""
    audio_file = request.audio_id
    
    print("\n" + "="*50)
    print(f"INFERENCE REQUEST RECEIVED")
    print(f"Audio File: {audio_file}")
    print(f"Question: {request.question}")
    print(f"Choices: {request.choices}")
    print("="*50)
    
    if not os.path.exists(audio_file):
        print(f"ERROR: Audio file not found at {audio_file}")
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_file}")
    
    print("Running Audio Flamingo 3...")
    
    # Run the model
    raw_af3_answer = infer_af3(request.question, request.choices, audio_file)
    
    print(f"RAW AF3 OUTPUT: '{raw_af3_answer}'")
    
    af3_answer = raw_af3_answer.strip()
    
    matched = False
    for choice in request.choices:
        if choice.lower() in af3_answer.lower():
            af3_answer = choice
            matched = True
            break
            
    if matched:
        print(f"MATCHED WITH CHOICE: '{af3_answer}'")
    else:
        print(f"WARNING: AF3 output did not perfectly match any choice.")
        
    print(f"📤 SENDING TO FRONTEND: '{af3_answer}'")
    print("="*50 + "\n")
            
    return {"af3_answer": af3_answer}

@app.get("/audio/{file_path:path}")
def get_audio(file_path: str):
    clean_path = "/" + file_path.lstrip("/")
    
    if not os.path.exists(clean_path):
        raise HTTPException(status_code=404, detail=f"Audio not found at: {clean_path}")
        
    return FileResponse(clean_path)

@app.get("/")
def serve_index():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html not found!")
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)