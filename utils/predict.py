
import numpy as np
import tensorflow as tf
import json
from PIL import Image

def load_tflite_model(model_path, labels_path, idx_to_class_path):
    """Load TFLite model and label files"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    with open(labels_path, "r", encoding="utf-8") as f:
        modi_labels = json.load(f)
    with open(idx_to_class_path, "r") as f:
        idx_to_class = json.load(f)
    
    return interpreter, modi_labels, idx_to_class

def predict(image, interpreter, modi_labels, idx_to_class, top_k=3):
    """
    Predict Modi character from image.
    image: PIL Image object
    Returns list of top_k predictions with metadata
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
    return results
