from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fog Node Explanation Service", version="1.0")

# Model configuration
MODEL_NAME = "distilgpt2"  # ~82M parameters, suitable for Jetson Nano
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model on device: {DEVICE}")

# Load model with quantization if possible
try:
    # Try 8-bit quantization (requires bitsandbytes)
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    logger.info("Model loaded with 8-bit quantization")
except (ImportError, RuntimeError) as e:
    # Fallback to float16
    logger.warning(f"8-bit quantization failed: {e}. Using float16.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    ).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

class DeviationItem(BaseModel):
    feature: str
    deviation: float

class ExplanationRequest(BaseModel):
    trust_score: float
    deviations: List[DeviationItem]

class ExplanationResponse(BaseModel):
    explanation: str
    model: str = MODEL_NAME

@app.post("/explain", response_model=ExplanationResponse)
async def explain(request: ExplanationRequest):
    """
    Generate a natural language explanation for detected anomalies.
    
    The input includes trust_score and a list of feature deviations.
    Only the top-k deviations are sent to preserve privacy.
    """
    try:
        # Build prompt
        prompt = f"""Trust score: {request.trust_score:.2f}. The following sensor features show unusual deviations:

"""
        for d in request.deviations:
            prompt += f"- {d.feature}: {d.deviation:.4f}\n"
        prompt += """
Provide a brief explanation of the potential issue (1-2 sentences):"""

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode and clean up
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = full_output[len(prompt):].strip()
        
        # Fallback if explanation is empty or too short
        if not explanation or len(explanation) < 10:
            explanation = f"Unusual pattern detected: {len(request.deviations)} sensor features show abnormal behavior."
        
        return ExplanationResponse(explanation=explanation)
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return ExplanationResponse(
            explanation="Anomaly detected but explanation generation failed. Please check system logs."
        )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/")
async def root():
    return {"message": "Fog Node Explanation Service", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )