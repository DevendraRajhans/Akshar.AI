
import numpy as np
import tensorflow as tf
import json
from PIL import Image

# def load_tflite_model(model_path, labels_path, idx_to_class_path):
#     """Load TFLite model and label files"""
#     with open(model_path, "rb") as f:
#         model_content = f.read()

#     interpreter = tf.lite.Interpreter(model_content=model_content)
#     interpreter.allocate_tensors()
    
#     with open(labels_path, "r", encoding="utf-8") as f:
#         modi_labels = json.load(f)
#     with open(idx_to_class_path, "r") as f:
#         idx_to_class = json.load(f)
    
#     return interpreter, modi_labels, idx_to_class

def load_tflite_model(model_path, labels_path, idx_to_class_path):
    import tensorflow as tf

    # 🔥 FIX: load model as bytes (important for Streamlit)
    with open(model_path, "rb") as f:
        model_content = f.read()

    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    
    import json
    with open(labels_path, "r", encoding="utf-8") as f:
        modi_labels = json.load(f)
    with open(idx_to_class_path, "r") as f:
        idx_to_class = json.load(f)
    
    return interpreter, modi_labels, idx_to_class

def compute_entropy(probabilities):
    """
    Higher entropy = model is confused = likely not a valid character.
    Max entropy for 57 classes = log(57) ≈ 4.04
    """
    probs = np.clip(probabilities, 1e-10, 1.0)
    return float(-np.sum(probs * np.log(probs)))

def is_valid_input(probabilities, 
                   confidence_threshold=0.30,
                   entropy_threshold=3.0):
    """
    Returns True if the image is likely a real Modi character.
    Returns False if it looks like noise/random strokes.
    
    Two conditions BOTH must pass:
    1. Top confidence must be above threshold
    2. Entropy must be below threshold (model is not confused)
    """
    top_confidence = float(np.max(probabilities))
    entropy        = compute_entropy(probabilities)
    
    passes_confidence = top_confidence >= confidence_threshold
    passes_entropy    = entropy <= entropy_threshold
    
    return passes_confidence and passes_entropy, {
        'top_confidence': top_confidence,
        'entropy': entropy,
        'max_entropy': np.log(57),
        'confidence_ok': passes_confidence,
        'entropy_ok': passes_entropy
    }

def get_rejection_reason(diagnostics):
    if not diagnostics['confidence_ok'] and not diagnostics['entropy_ok']:
        return "Image does not appear to be a Modi character — confidence too low and pattern too ambiguous."
    elif not diagnostics['confidence_ok']:
        return "Confidence too low. Try a cleaner, larger image of a single character."
    else:
        return "Pattern too ambiguous. The image may contain noise or multiple characters."

def predict(image, interpreter, modi_labels, idx_to_class, top_k=3):
    """
    Predict Modi character from image.
    image: PIL Image object
    Returns dict with validity check, list of top_k predictions and diagnostics
    """
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    # Infer
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    
    # Check if input is valid
    valid, diagnostics = is_valid_input(preds)
    
    if not valid:
        return {
            'valid': False,
            'reason': get_rejection_reason(diagnostics),
            'diagnostics': diagnostics,
            'results': []
        }
    
    # Top-k results
    top_indices = np.argsort(preds)[::-1][:top_k]
    results = []
    for idx in top_indices:
        cls  = idx_to_class[str(idx)]
        conf = float(preds[idx])
        info = modi_labels.get(cls, {})
        results.append({
            "class_name":     cls,
            "confidence":     conf,
            "confidence_pct": f"{conf*100:.1f}%",
            "devanagari":     info.get("devanagari", "?"),
            "english_name":   info.get("english_name", cls),
            "pronunciation":  info.get("pronunciation", ""),
            "example_word":   info.get("example_word", ""),
            "historical_note":info.get("historical_note", ""),
            "character_type": info.get("character_type", "")
        })
    
    return {
        'valid': True,
        'results': results,
        'diagnostics': diagnostics
    }
