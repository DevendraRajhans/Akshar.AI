import tensorflow as tf

# 🔥 Load WITHOUT compilation
model = tf.keras.models.load_model(
    "model/aksharai_final.keras",
    compile=False
)

# 🔥 Rebuild model cleanly (IMPORTANT)
inputs = tf.keras.Input(shape=model.input_shape[1:])
outputs = model(inputs)
clean_model = tf.keras.Model(inputs, outputs)

# 🔥 Save in compatible format
clean_model.save("model/aksharai_fixed.h5")

print("✅ Grad-CAM model fixed and saved!")