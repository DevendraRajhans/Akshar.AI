import numpy as np
from PIL import Image, ImageOps, ImageFilter
from skimage.metrics import structural_similarity as ssim

def _preprocess_for_comparison(img_bin):
    """
    Crops to bounding box and pads to a square to ensure scale invariance.
    """
    inv = ImageOps.invert(img_bin)
    bbox = inv.getbbox()
    if bbox:
        cropped = img_bin.crop(bbox)
        side = max(cropped.size)
        
        # 15% standard padding
        pad = int(side * 0.15)
        padded_side = side + 2 * pad
        
        padded = Image.new("L", (padded_side, padded_side), 255)
        offset = (pad + (side - cropped.size[0]) // 2, pad + (side - cropped.size[1]) // 2)
        padded.paste(cropped, offset)
        return padded
    return img_bin

def compare_images(drawn_img, reference_path):
    """
    Compares drawn canvas with a reference character image.
    Returns a similarity score between 0.0 and 1.0.
    """
    # 1. Load Reference Image
    ref_img = Image.open(reference_path).convert("L")
    ref_arr = np.array(ref_img)
    ref_bin = (ref_arr > 128).astype(np.uint8) * 255
    ref_img_bin = Image.fromarray(ref_bin)
    
    # 2. Process Drawn Image
    if drawn_img.mode in ('RGBA', 'LA') or (drawn_img.mode == 'P' and 'transparency' in drawn_img.info):
        alpha = drawn_img.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", drawn_img.size, (255,255,255,255))
        bg.paste(drawn_img, mask=alpha)
        drawn_img = bg.convert("RGB")
        
    drawn_img = drawn_img.convert("L")
    drawn_arr = np.array(drawn_img)
    
    # Binarize with threshold ~200 (since background is white, strokes are black/gray)
    drawn_bin_arr = (drawn_arr > 200).astype(np.uint8) * 255
    drawn_img_bin = Image.fromarray(drawn_bin_arr)
    
    # 3. Normalize both images (Bounding box + Square padding)
    norm_ref = _preprocess_for_comparison(ref_img_bin)
    norm_drawn = _preprocess_for_comparison(drawn_img_bin)
    
    # 4. Resize drawn to exactly match reference
    target_size = norm_ref.size
    final_drawn = norm_drawn.resize(target_size, Image.Resampling.LANCZOS)
    
    # 5. Apply subtle blur to handle minor stroke variations
    final_drawn = final_drawn.filter(ImageFilter.GaussianBlur(radius=3))
    final_ref = norm_ref.filter(ImageFilter.GaussianBlur(radius=3))
    
    # 6. SSIM Comparison
    score, _ = ssim(np.array(final_ref), np.array(final_drawn), full=True, data_range=255)
    
    return max(0.0, float(score))
