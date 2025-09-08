# fixed_gradio_app.py (Correctly processes sketchpad layers)

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
from model import MnistFullyCNN
from config import model_path, device

# Load model
model = MnistFullyCNN()
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def process_sketchpad_correctly(sketchpad_data):
    """Correctly process sketchpad data by combining layers"""
    
    if not isinstance(sketchpad_data, dict):
        return None
    
    print(f"Processing sketchpad with keys: {list(sketchpad_data.keys())}")
    
    # Method 1: Use composite if available (this should have the final result)
    if 'composite' in sketchpad_data and sketchpad_data['composite'] is not None:
        composite = sketchpad_data['composite']
        print(f"Using composite: {composite.shape}, min/max: {composite.min()}/{composite.max()}")
        
        # Check if composite has actual drawing (not just background)
        if composite.max() - composite.min() > 0:
            return Image.fromarray(composite.astype(np.uint8))
    
    # Method 2: Process layers (where the actual drawing is)
    if 'layers' in sketchpad_data and sketchpad_data['layers'] is not None:
        layers = sketchpad_data['layers']
        print(f"Processing {len(layers)} layers")
        
        # Get background
        background = None
        if 'background' in sketchpad_data and sketchpad_data['background'] is not None:
            background = sketchpad_data['background'].copy()
        else:
            # Create white background if none exists
            if len(layers) > 0:
                background = np.full_like(layers[0], 255, dtype=np.uint8)
        
        # Combine layers onto background
        combined = background.copy() if background is not None else None
        
        for i, layer in enumerate(layers):
            if layer is not None:
                print(f"Layer {i}: shape={layer.shape}, min/max={layer.min()}/{layer.max()}")
                
                if combined is None:
                    combined = layer.copy()
                else:
                    # Handle alpha blending if layer has alpha channel
                    if layer.shape[-1] == 4:  # RGBA
                        alpha = layer[:, :, 3:4] / 255.0
                        rgb = layer[:, :, :3]
                        
                        # Blend with background
                        if combined.shape[-1] == 4:
                            combined_rgb = combined[:, :, :3]
                            combined[:, :, :3] = (alpha * rgb + (1 - alpha) * combined_rgb).astype(np.uint8)
                        else:
                            combined = (alpha * rgb + (1 - alpha) * combined).astype(np.uint8)
                    else:
                        # Simple overlay for RGB layers
                        mask = np.any(layer != [0, 0, 0], axis=2) if layer.shape[-1] == 3 else layer != 0
                        combined[mask] = layer[mask]
        
        if combined is not None:
            print(f"Combined result: shape={combined.shape}, min/max={combined.min()}/{combined.max()}")
            return Image.fromarray(combined.astype(np.uint8))
    
    return None

def predict_drawing(sketchpad_input):
    """Predict from sketchpad input"""
    
    if sketchpad_input is None:
        return "Please draw something", {}
    
    try:
        print("=== PROCESSING SKETCHPAD ===")
        
        # Process the sketchpad correctly
        image = process_sketchpad_correctly(sketchpad_input)
        
        if image is None:
            return "Could not extract drawing from canvas", {}
        
        print(f"Extracted image: {image.mode}, {image.size}")
        
        # Convert to grayscale
        if image.mode != 'L':
            if image.mode in ('RGBA', 'LA'):
                # Create white background for transparency
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])
                image = background
            image = image.convert('L')
        
        # Get image statistics
        img_array = np.array(image)
        contrast = img_array.max() - img_array.min()
        mean_val = img_array.mean()
        
        print(f"Final image stats: min={img_array.min()}, max={img_array.max()}, mean={mean_val:.1f}, contrast={contrast}")
        
        # Check if there's actual content
        if contrast < 5:
            return f"Please draw a more visible digit (contrast: {contrast})", {}
        
        # Auto-invert if needed (dark drawing on light background)
        if mean_val > 127:
            image = ImageOps.invert(image)
            print("Image inverted")
        
        # Make prediction
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(outputs).item()
            conf = probs[0][pred].item()
        
        result = f"🎯 Predicted: {pred}\n📊 Confidence: {conf:.2%}\n🔍 Contrast: {contrast}"
        prob_dict = {str(i): float(probs[0][i]) for i in range(10)}
        
        return result, prob_dict
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, {}

def predict_upload(image):
    """Predict from uploaded image"""
    if image is None:
        return "Please upload an image", {}
    
    try:
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        img_array = np.array(image)
        
        # Auto-invert if needed
        if img_array.mean() > 127:
            image = ImageOps.invert(image)
        
        # Make prediction
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(outputs).item()
            conf = probs[0][pred].item()
        
        result = f"🎯 Predicted: {pred}\n📊 Confidence: {conf:.2%}"
        prob_dict = {str(i): float(probs[0][i]) for i in range(10)}
        
        return result, prob_dict
        
    except Exception as e:
        return f"Error: {str(e)}", {}

# Create interface
with gr.Blocks(title="MNIST Classifier") as app:
    gr.Markdown("# 🔢 MNIST Digit Classifier")
    
    with gr.Tab("✏️ Draw"):
        gr.Markdown("### Draw a digit (0-9)")
        sketchpad = gr.Sketchpad(label="Draw here")
        draw_btn = gr.Button("🔍 Predict Drawing", variant="primary")
        draw_result = gr.Textbox(label="Result", lines=4)
        draw_probs = gr.Label(label="Probabilities")
        
        draw_btn.click(predict_drawing, sketchpad, [draw_result, draw_probs])
    
    with gr.Tab("📁 Upload"):
        gr.Markdown("### Upload an image")
        upload = gr.Image(type="pil", label="Upload image")
        upload_btn = gr.Button("🔍 Predict Upload", variant="primary")
        upload_result = gr.Textbox(label="Result", lines=4)
        upload_probs = gr.Label(label="Probabilities")
        
        upload_btn.click(predict_upload, upload, [upload_result, upload_probs])
        upload.change(predict_upload, upload, [upload_result, upload_probs])

if __name__ == "__main__":
    app.launch(debug=True)