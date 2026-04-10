# #
# # import tensorflow as tf
# # import numpy as np
# # import cv2
# # from PIL import Image
# #
# # def get_last_conv_layer(model):
# #     base = model.layers[0]
# #     for layer in reversed(base.layers):
# #         if "conv" in layer.name.lower():
# #             return layer.name
# #     return None
# #
# # # def generate_gradcam(image_pil, model, last_conv_layer_name, alpha=0.4):
# # #     img_array = np.expand_dims(
# # #         np.array(image_pil.convert("RGB").resize((224, 224))) / 255.0,
# # #         axis=0
# # #     ).astype(np.float32)
# # #
# # #     base = model.layers[0]
# # #     conv_layer = base.get_layer(last_conv_layer_name)
# # #     feature_extractor = tf.keras.Model(
# # #         inputs=base.input,
# # #         outputs=conv_layer.output
# # #     )
# # #
# # #     with tf.GradientTape() as tape:
# # #         conv_output = feature_extractor(img_array, training=False)
# # #         tape.watch(conv_output)
# # #         x = conv_output
# # #         for layer in model.layers[1:]:
# # #             x = layer(x, training=False)
# # #         predictions = x
# # #         pred_idx = tf.argmax(predictions[0])
# # #         score = predictions[:, pred_idx]
# # #
# # #     grads = tape.gradient(score, conv_output)
# # #     pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
# # #     heatmap = conv_output[0] @ pooled[..., tf.newaxis]
# # #     heatmap = tf.squeeze(heatmap)
# # #     heatmap = tf.maximum(heatmap, 0)
# # #     heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
# # #     heatmap = heatmap.numpy()
# # #
# # #     heatmap_resized = cv2.resize(heatmap, (224, 224))
# # #     colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
# # #     orig = np.array(image_pil.convert("RGB").resize((224, 224)))
# # #     orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
# # #     blended = cv2.addWeighted(orig_bgr, 1-alpha, colored, alpha, 0)
# # #     return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
# #
# #
# #
# # def generate_gradcam(image_pil, model, last_conv_layer_name, alpha=0.4):
# #     img_array = np.expand_dims(
# #         np.array(image_pil.convert("RGB").resize((224, 224))) / 255.0,
# #         axis=0
# #     ).astype(np.float32)
# #
# #     # Get base model (MobileNetV2)
# #     base_model = model.layers[0]
# #
# #     # Get last conv layer
# #     last_conv_layer = base_model.get_layer(last_conv_layer_name)
# #
# #     # Create Grad-CAM model
# #     grad_model = tf.keras.models.Model(
# #         inputs=model.input,
# #         outputs=[last_conv_layer.output, model.output]
# #     )
# #
# #     with tf.GradientTape() as tape:
# #         conv_outputs, predictions = grad_model(img_array)
# #         pred_index = tf.argmax(predictions[0])
# #         loss = predictions[:, pred_index]
# #
# #     # Compute gradients
# #     grads = tape.gradient(loss, conv_outputs)
# #
# #     # Global average pooling
# #     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
# #
# #     # Weight feature maps
# #     conv_outputs = conv_outputs[0]
# #     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
# #     heatmap = tf.squeeze(heatmap)
# #
# #     # Normalize
# #     heatmap = tf.maximum(heatmap, 0)
# #     heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
# #     heatmap = heatmap.numpy()
# #
# #     # Resize + overlay
# #     heatmap_resized = cv2.resize(heatmap, (224, 224))
# #     colored = cv2.applyColorMap(
# #         np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
# #     )
# #
# #     original = np.array(image_pil.convert("RGB").resize((224, 224)))
# #     original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
# #
# #     blended = cv2.addWeighted(original_bgr, 1 - alpha, colored, alpha, 0)
# #
# #     return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
#
# import tensorflow as tf
# import numpy as np
# import cv2
# from PIL import Image
#
#
# def get_last_conv_layer(model):
#     for layer in reversed(model.layers[0].layers):
#         if "conv" in layer.name.lower():
#             return layer.name
#     return None
#
#
# def generate_gradcam(image_pil, model, last_conv_layer_name, alpha=0.4):
#     """
#     Generate Grad-CAM overlay for a PIL image.
#     Returns: PIL Image with heatmap overlaid
#     """
#     img_array = np.expand_dims(
#         np.array(image_pil.convert("RGB").resize((224, 224))) / 255.0,
#         axis=0
#     ).astype(np.float32)
#
#     grad_model = tf.keras.models.Model(
#         inputs=model.inputs,
#         outputs=[
#             model.get_layer(model.layers[0].name)
#             .get_layer(last_conv_layer_name).output,
#             model.output
#         ]
#     )
#
#     with tf.GradientTape() as tape:
#         conv_out, preds = grad_model(img_array)
#         pred_idx = tf.argmax(preds[0])
#         score = preds[:, pred_idx]
#
#     grads = tape.gradient(score, conv_out)
#     pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
#     heatmap = conv_out[0] @ pooled[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0)
#     heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
#     heatmap = heatmap.numpy()
#
#     # Resize and colorize
#     heatmap_resized = cv2.resize(heatmap, (224, 224))
#     colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
#
#     # Blend with original
#     orig = np.array(image_pil.convert("RGB").resize((224, 224)))
#     orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
#     blended = cv2.addWeighted(orig_bgr, 1 - alpha, colored, alpha, 0)
#     blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
#
#     return Image.fromarray(blended_rgb)
# import tensorflow as tf
# import numpy as np
# import cv2
# from PIL import Image
#
#
# def get_last_conv_layer(model):
#     """Find the last convolutional layer in the base model."""
#     try:
#         base = model.layers[0]  # MobileNetV2 is the first layer
#         for layer in reversed(base.layers):
#             if 'conv' in layer.name.lower():
#                 return layer.name
#     except Exception:
#         pass
#     return None
#
#
# def generate_gradcam(image_pil, model, last_conv_layer_name, alpha=0.4):
#     """
#     Generate Grad-CAM heatmap overlay on a PIL image.
#
#     Uses a two-model approach compatible with Keras 3 Sequential models:
#     - feature_extractor: input → last conv layer output
#     - classifier_head:   last conv layer output → predictions
#
#     Returns PIL Image with heatmap overlaid, or original image if it fails.
#     """
#     try:
#         # Preprocess image
#         img_resized = image_pil.convert('RGB').resize((224, 224))
#         img_array = np.expand_dims(
#             np.array(img_resized) / 255.0, axis=0
#         ).astype(np.float32)
#
#         base_model = model.layers[0]
#
#         # Model 1: Input → last conv layer output
#         feature_extractor = tf.keras.Model(
#             inputs=base_model.input,
#             outputs=base_model.get_layer(last_conv_layer_name).output
#         )
#
#         # Compute gradients
#         with tf.GradientTape() as tape:
#             # Get conv features and watch them
#             conv_output = feature_extractor(img_array, training=False)
#             tape.watch(conv_output)
#
#             # Pass through remaining layers manually
#             # model.layers[0] = base, [1] = GAP, [2] = Dense, [3] = Dropout, [4] = Dense(57)
#             x = conv_output
#             for layer in model.layers[1:]:
#                 x = layer(x, training=False)
#
#             predictions = x
#             pred_class_idx = tf.argmax(predictions[0])
#             class_score = predictions[:, pred_class_idx]
#
#         # Gradient of class score w.r.t. conv output
#         grads = tape.gradient(class_score, conv_output)
#
#         if grads is None:
#             return image_pil.convert('RGB').resize((224, 224))
#
#         # Average gradients over spatial dimensions: (H, W, C) → (C,)
#         pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#
#         # Weight each feature map by its gradient
#         # conv_output[0]: (H, W, C), pooled_grads: (C,)
#         heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
#         heatmap = tf.squeeze(heatmap)
#
#         # ReLU and normalize to [0, 1]
#         heatmap = tf.maximum(heatmap, 0)
#         max_val = tf.math.reduce_max(heatmap)
#         if max_val == 0:
#             return img_resized
#         heatmap = heatmap / max_val
#         heatmap = heatmap.numpy()
#
#         # Resize heatmap to image size
#         heatmap_resized = cv2.resize(heatmap, (224, 224))
#
#         # Apply JET colormap: blue=low influence, red=high influence
#         colored = cv2.applyColorMap(
#             np.uint8(255 * heatmap_resized),
#             cv2.COLORMAP_JET
#         )
#
#         # Blend with original image
#         orig_rgb = np.array(img_resized)
#         orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
#         blended  = cv2.addWeighted(orig_bgr, 1 - alpha, colored, alpha, 0)
#
#         return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
#
#     except Exception as e:
#         print(f"Grad-CAM failed: {e}")
#         # Return original image resized — app still works
#         return image_pil.convert('RGB').resize((224, 224))

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


def get_last_conv_layer(model):
    """
    Find the last convolutional layer in the entire model.
    Works with Functional API and nested models.
    """
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            for inner_layer in reversed(layer.layers):
                if "conv" in inner_layer.name.lower():
                    return inner_layer.name
        if "conv" in layer.name.lower():
            return layer.name
    return None


def generate_gradcam(image_pil, model, last_conv_layer_name, alpha=0.4):
    """
    Generate Grad-CAM heatmap overlay on a PIL image.
    Uses TensorFlow GradientTape for true gradient-based Grad-CAM.
    Compatible with Keras models.
    """
    try:
        # Preprocess image
        img_resized = image_pil.convert("RGB").resize((224, 224))
        img_array = np.expand_dims(
            np.array(img_resized) / 255.0, axis=0
        ).astype(np.float32)

        # Locate the conv layer (could be nested inside a base model)
        conv_layer = None
        target_model = model
        
        for layer in model.layers:
            if layer.name == last_conv_layer_name:
                conv_layer = layer
                break
            if hasattr(layer, 'layers'):
                for inner_layer in layer.layers:
                    if inner_layer.name == last_conv_layer_name:
                        conv_layer = inner_layer
                        target_model = layer
                        break
        
        if conv_layer is None:
            # Fallback to direct get_layer if above fails
            conv_layer = model.get_layer(last_conv_layer_name)

        # Create Grad-CAM model based on where the conv layer is
        if target_model == model:
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[conv_layer.output, model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                pred_index = tf.argmax(predictions[0])
                class_score = predictions[:, pred_index]

            grads = tape.gradient(class_score, conv_outputs)
            
        else:
            # Two-model approach for nested base models (e.g., Sequential -> MobileNetV2 -> layers)
            feature_extractor = tf.keras.models.Model(
                inputs=target_model.inputs,
                outputs=conv_layer.output
            )
            
            with tf.GradientTape() as tape:
                conv_outputs = feature_extractor(img_array, training=False)
                tape.watch(conv_outputs)
                
                # Pass through the rest of the layers
                x = conv_outputs
                for layer in target_model.layers:
                    # simplistic logic to skip layers before the conv layer
                    # actually it's easier to use gradient wrt base model output
                    pass
                
                # A better approach for nested models without duplicating rest of the network:
                base_outputs = target_model(img_array, training=False)
                # Since we don't have a direct path from conv_outputs to model.output easily
                # We'll just fallback to the original grad_model if possible, but catching errors
            raise ValueError("Nested model Grad-CAM requires specific sub-model wiring.")
            
    except ValueError:
        # Fallback simplistic grad_model definition targeting the inner convolutions
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(target_model.name).get_layer(last_conv_layer_name).output if target_model != model else model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]

        grads = tape.gradient(class_score, conv_outputs)
        
    except Exception as e:
        print(f"⚠️ Initial Grad-CAM model creation failed, retrying simple approach: {e}")
        # One last try standard approach
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]

        grads = tape.gradient(class_score, conv_outputs)

    try:
        if grads is None:
            print("⚠️ Gradients are None")
            return img_resized

        # Compute gradients and global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize safely (avoid division by zero)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        heatmap = heatmap / (max_val + 1e-8)
        heatmap = heatmap.numpy()

        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap, (224, 224))

        # Apply color map
        colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            cv2.COLORMAP_JET
        )

        # Overlay on original image (using proper OpenCV logic)
        original = np.array(img_resized)
        original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

        blended = cv2.addWeighted(
            original_bgr, 1 - alpha,
            colored, alpha,
            0
        )

        # Convert back to RGB PIL
        return Image.fromarray(
            cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        )

    except Exception as e:
        print(f"❌ Grad-CAM failed during gradient computation: {e}")
        return image_pil.convert("RGB").resize((224, 224))