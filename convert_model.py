import tensorflow as tf

model = tf.keras.models.load_model("model/aksharai_final.keras")

# 🔥 RE-SAVE CLEAN MODEL
model.save("model/aksharai_fixed.keras", save_format="keras")