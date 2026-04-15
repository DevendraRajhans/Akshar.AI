import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("model/aksharai_final.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 🔥 IMPORTANT FIX
converter.target_spec.supported_types = [tf.float32]

tflite_model = converter.convert()

# Save new model
with open("model/aksharai_model_tf.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted successfully!")