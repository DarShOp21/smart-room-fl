from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fog Node Explanation Service", version="1.0")

# Model configuration
MODEL_NAME = "distilgpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model on device: {DEVICE}")

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    
    logger.info(f"Model loaded successfully: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    logger.info("Falling back to rule-based explanations")
    USE_LLM = False
else:
    USE_LLM = True

class DeviationItem(BaseModel):
    feature: str
    deviation: float

class ExplanationRequest(BaseModel):
    trust_score: float
    deviations: List[DeviationItem]

class ExplanationResponse(BaseModel):
    explanation: str
    trust_score: float
    severity: str
    model: str = MODEL_NAME
    llm_used: bool = USE_LLM

# Feature descriptions for fallback and context
FEATURE_DESCRIPTIONS = {
    "motion_rate": "motion detection frequency",
    "motion_variance": "irregularity in motion patterns",
    "entry_time": "entry timing",
    "entry_deviation": "deviation from normal schedule",
    "session_duration": "occupancy duration",
    "device_usage": "device interaction pattern",
    "sequence_similarity": "device usage sequence consistency"
}

def get_severity(trust_score: float) -> tuple:
    """Determine severity level based on trust score"""
    if trust_score < 0.2:
        return "CRITICAL", "highly concerning"
    elif trust_score < 0.5:
        return "WARNING", "moderately concerning"
    elif trust_score < 0.8:
        return "CAUTION", "mildly concerning"
    else:
        return "NORMAL", "normal"

def generate_llm_explanation(trust_score: float, deviations: List[DeviationItem]) -> str:
    """Generate explanation using language model"""
    # Sort deviations by magnitude
    sorted_deviations = sorted(deviations, key=lambda x: x.deviation, reverse=True)
    
    # Build prompt
    prompt = f"Trust score: {trust_score:.2f}. "
    
    if sorted_deviations:
        prompt += "Unusual sensor readings detected: "
        deviation_descriptions = []
        for dev in sorted_deviations[:3]:  # Top 3 deviations
            desc = FEATURE_DESCRIPTIONS.get(dev.feature, dev.feature)
            deviation_descriptions.append(f"{desc} (deviation: {dev.deviation:.3f})")
        prompt += ", ".join(deviation_descriptions) + ". "
    
    prompt += "Provide a brief security assessment and recommendation:"
    
    # Generate with the model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200).to(DEVICE)
    
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
    
    # Clean up common issues
    explanation = explanation.replace("-------------\n", "")
    explanation = explanation.replace("...", ".")
    
    # If explanation is too long, truncate
    if len(explanation) > 300:
        explanation = explanation[:300] + "..."
    
    return explanation if explanation else "Anomaly detected. Investigation recommended."

def generate_rule_explanation(trust_score: float, deviations: List[DeviationItem]) -> str:
    """Generate explanation using rules (fallback)"""
    severity, severity_text = get_severity(trust_score)
    
    if not deviations:
        return f"System behavior is {severity_text}. No significant deviations detected."
    
    # Sort deviations
    sorted_deviations = sorted(deviations, key=lambda x: x.deviation, reverse=True)
    
    parts = []
    
    # List top deviations
    top_features = []
    for dev in sorted_deviations[:3]:
        desc = FEATURE_DESCRIPTIONS.get(dev.feature, dev.feature)
        top_features.append(f"{desc} (deviation: {dev.deviation:.3f})")
    
    if top_features:
        parts.append(f"Unusual patterns detected in: {', '.join(top_features)}.")
        
        # Specific recommendations
        if any(d.feature in ["motion_rate", "motion_variance"] for d in sorted_deviations[:2]):
            parts.append("Abnormal movement patterns detected.")
        
        if any(d.feature in ["entry_time", "entry_deviation"] for d in sorted_deviations[:2]):
            parts.append("Entry timing deviates from normal schedule.")
        
        if any(d.feature in ["session_duration", "device_usage"] for d in sorted_deviations[:2]):
            parts.append("Irregular occupancy and device interaction patterns.")
        
        if any(d.feature == "sequence_similarity" for d in sorted_deviations[:2]):
            parts.append("Device usage sequence is unusual.")
        
        # Recommendations
        if severity == "CRITICAL":
            parts.append("Immediate investigation required!")
        elif severity == "WARNING":
            parts.append("Monitor the situation closely.")
        elif severity == "CAUTION":
            parts.append("Routine check recommended.")
    
    return " ".join(parts)

@app.post("/explain", response_model=ExplanationResponse)
async def explain(request: ExplanationRequest):
    """
    Generate a natural language explanation for detected anomalies.
    Uses LLM if available, otherwise falls back to rule-based.
    """
    try:
        severity, _ = get_severity(request.trust_score)
        
        # Generate explanation
        if USE_LLM:
            try:
                explanation = generate_llm_explanation(request.trust_score, request.deviations)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                explanation = generate_rule_explanation(request.trust_score, request.deviations)
        else:
            explanation = generate_rule_explanation(request.trust_score, request.deviations)
        
        return ExplanationResponse(
            explanation=explanation,
            trust_score=request.trust_score,
            severity=severity
        )
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return ExplanationResponse(
            explanation=f"Anomaly detected (trust score: {request.trust_score:.2f}). System recommends investigation.",
            trust_score=request.trust_score,
            severity="UNKNOWN"
        )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "llm_available": USE_LLM,
        "cuda_available": torch.cuda.is_available(),
        "features": list(FEATURE_DESCRIPTIONS.keys())
    }

@app.get("/")
async def root():
    return {
        "message": "Fog Node Explanation Service",
        "version": "1.0",
        "llm_enabled": USE_LLM,
        "model": MODEL_NAME if USE_LLM else "rule-based"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )