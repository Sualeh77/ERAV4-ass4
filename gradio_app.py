# gradio_app.py (Updated with fixed sketchpad processing)

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import io
from pathlib import Path
from torchvision import transforms

# Import your model
from model import MnistFullyCNN
from config import model_path, device

class MNISTGradioPredictor:
    def __init__(self):
        self.device = device
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = MnistFullyCNN()
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def process_sketchpad_input(self, sketchpad_data):
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
    
    def preprocess_image(self, image):
        """Preprocess the uploaded image"""
        if image is None:
            return None, None
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to grayscale and invert if needed
        image = image.convert('L')
        
        # Check if we need to invert (assuming dark digit on light background)
        img_array = np.array(image)
        if img_array.mean() > 127:  # Light background
            image = ImageOps.invert(image)
        
        # Apply transforms
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device), image
    
    def predict(self, image_input):
        """Make prediction and return results"""
        if image_input is None:
            return None, None, {}, "Please upload or draw an image first!"
        
        try:
            image = None
            input_type = "unknown"
            
            # Handle different input types
            if isinstance(image_input, dict):
                # From Sketchpad
                input_type = "sketchpad"
                image = self.process_sketchpad_input(image_input)
                if image is None:
                    return None, None, {}, "Please draw something on the canvas"
            elif isinstance(image_input, np.ndarray):
                input_type = "numpy"
                image = Image.fromarray(image_input)
            elif isinstance(image_input, Image.Image):
                input_type = "pil"
                image = image_input
            else:
                return None, None, {}, f"Unsupported input type: {type(image_input)}"
            
            if image is None:
                return None, None, {}, "Could not process the image"
            
            # Convert to RGB first, then to grayscale (handles RGBA)
            if image.mode in ('RGBA', 'LA'):
                # Create white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                else:
                    background.paste(image, mask=image.split()[-1])
                image = background
            
            image = image.convert('L')
            
            # Get image statistics
            img_array = np.array(image)
            img_min, img_max = img_array.min(), img_array.max()
            img_mean = img_array.mean()
            contrast = img_max - img_min
            
            print(f"Debug - Input type: {input_type}")
            print(f"Debug - Image stats: min={img_min}, max={img_max}, mean={img_mean:.1f}, contrast={contrast}")
            
            # Check contrast
            if contrast < 5:
                return None, None, {}, f"Please draw a clearer digit (contrast: {contrast})"
            
            # Invert if needed (dark digit on light background)
            if img_mean > 127:
                image = ImageOps.invert(image)
                print("Debug - Image inverted")
            
            # Preprocess image
            tensor, processed_img = self.preprocess_image(image)
            if tensor is None:
                return None, None, {}, "Failed to process image"
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Create probability chart
            prob_dict = {str(i): float(probabilities[0][i]) for i in range(10)}
            
             # Create confidence bar chart - HORIZONTAL
            fig, ax = plt.subplots(figsize=(10, 8))
            digits = list(range(10))
            probs = probabilities[0].cpu().numpy()
            
            # Use barh for horizontal bars
            bars = ax.barh(digits, probs, color=['#ff6b6b' if i == predicted_class else '#4ecdc4' for i in digits])
            ax.set_ylabel('Digit', fontsize=12)
            ax.set_xlabel('Probability', fontsize=12)
            ax.set_title(f'Prediction Probabilities (Predicted: {predicted_class})', fontsize=14)
            ax.set_yticks(digits)
            ax.set_xlim(0, 1)
            
            # Add probability values on bars (adjusted for horizontal)
            for bar, prob in zip(bars, probs):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                       f'{prob:.3f}', ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            
            # Create result text
            result_text = f"""
🎯 **Predicted Digit: {predicted_class}**
📊 **Confidence: {confidence:.2%}**
🔍 **Contrast: {contrast:.1f}**

📈 **Top 3 Predictions:**
"""
            
            # Get top 3 predictions
            top3_indices = torch.topk(probabilities[0], 3).indices
            for i, idx in enumerate(top3_indices):
                digit = idx.item()
                prob = probabilities[0][digit].item()
                result_text += f"\n{i+1}. Digit {digit}: {prob:.2%}"
            
            return fig, processed_img, prob_dict, result_text
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None, {}, f"❌ Error during prediction: {str(e)}"

# Initialize predictor
try:
    predictor = MNISTGradioPredictor()
    model_status = "✅ Model loaded successfully!"
except Exception as e:
    predictor = None
    model_status = f"❌ Model loading failed: {str(e)}"

def predict_digit(image):
    """Main prediction function for Gradio"""
    if predictor is None:
        return None, None, {}, "❌ Model not loaded. Please check the model file."
    
    return predictor.predict(image)

def clear_all():
    """Clear all outputs"""
    return None, None, None, None, {}, ""

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif !important;
}

.main-header {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.prediction-box {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    border-left: 4px solid #007bff;
    color: #333333 !important;
}

.model-status {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
    color: #155724 !important;
}

/* Fix for light background sections */
.html-container {
    color: #333333 !important;
}

/* Ensure dark text on light backgrounds */
div[style*="background: #f8f9fa"], 
div[style*="background: #e3f2fd"],
div[style*="background: #d4edda"] {
    color: #333333 !important;
}

/* Override any white text on light backgrounds */
.svelte-phx28p.padding {
    color: #333333 !important;
}

/* Footer section fix */
div[style*="background: #f8f9fa; border-radius: 8px"] {
    color: #333333 !important;
}

div[style*="background: #f8f9fa; border-radius: 8px"] p {
    color: #333333 !important;
}

/* Instructions section fix */
div[style*="background: #e3f2fd"] {
    color: #333333 !important;
}

div[style*="background: #e3f2fd"] h4,
div[style*="background: #e3f2fd"] li,
div[style*="background: #e3f2fd"] ul {
    color: #333333 !important;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="MNIST Digit Classifier", theme=gr.themes.Soft()) as app:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🔢 MNIST Digit Classifier</h1>
        <p>Upload an image or draw a handwritten digit (0-9) and get AI-powered predictions!</p>
        <p><strong>Fully Convolutional Neural Network</strong> | <strong>&lt;25k Parameters</strong></p>
    </div>
    """)
    
    # Model status
    gr.HTML(f'<div class="model-status"><strong>Model Status:</strong> {model_status}</div>')
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.HTML("<h3>📤 Input Options</h3>")
            
            with gr.Tab("Upload Image"):
                upload_image = gr.Image(
                    label="Upload an Image",
                    sources=["upload", "webcam", "clipboard"],
                    type="pil",
                    height=300
                )
                upload_btn = gr.Button("🔍 Predict from Upload", variant="primary", size="lg")
            
            with gr.Tab("Draw Digit"):
                draw_image = gr.Sketchpad(
                    label="Draw a Digit"
                )
                draw_btn = gr.Button("🔍 Predict from Drawing", variant="primary", size="lg")
            
            clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
            
            # Instructions
            gr.HTML("""
            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin-top: 15px; color: #333333;">
                <h4 style="color: #333333;">📋 Instructions:</h4>
                <ul style="color: #333333;">
                    <li style="color: #333333;">✏️ <strong style="color: #333333;">Draw:</strong> Use the drawing pad to draw a digit</li>
                    <li style="color: #333333;">📁 <strong style="color: #333333;">Upload:</strong> Upload an image file</li>
                    <li style="color: #333333;">📷 <strong style="color: #333333;">Webcam:</strong> Take a photo with your camera</li>
                    <li style="color: #333333;">🎨 <strong style="color: #333333;">Tips:</strong> Dark digits on light background work best!</li>
                </ul>
            </div>
            """)
        
        with gr.Column(scale=2):
            # Results section
            gr.HTML("<h3>📊 Prediction Results</h3>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    processed_image = gr.Image(
                        label="Processed Image (28x28)",
                        type="pil",
                        height=200
                    )
                
                with gr.Column(scale=2):
                    result_text = gr.Markdown(
                        label="Prediction Results",
                        value="Upload or draw an image to see predictions..."
                    )
            
            # Probability visualization
            probability_plot = gr.Plot(
                label="Probability Distribution"
            )
            
            # Probability values (for API access)
            probability_json = gr.JSON(
                label="Raw Probabilities",
                visible=False
            )
    
    # Event handlers
    upload_btn.click(
        fn=predict_digit,
        inputs=[upload_image],
        outputs=[probability_plot, processed_image, probability_json, result_text]
    )
    
    draw_btn.click(
        fn=predict_digit,
        inputs=[draw_image],
        outputs=[probability_plot, processed_image, probability_json, result_text]
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[upload_image, draw_image, probability_plot, processed_image, probability_json, result_text]
    )
    
    # Auto-predict on image upload
    upload_image.change(
        fn=predict_digit,
        inputs=[upload_image],
        outputs=[probability_plot, processed_image, probability_json, result_text]
    )
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 8px; color: #333333;">
        <p style="color: #333333;"><strong style="color: #333333;">🧠 Model Architecture:</strong> Fully Convolutional Neural Network</p>
        <p style="color: #333333;"><strong style="color: #333333;">📊 Parameters:</strong> ~18k parameters (under 25k limit)</p>
        <p style="color: #333333;"><strong style="color: #333333;">🎯 Accuracy:</strong> >99% on MNIST test set</p>
        <p style="color: #333333;"><strong style="color: #333333;">⚡ Framework:</strong> PyTorch | <strong style="color: #333333;">🚀 Interface:</strong> Gradio</p>
    </div>
    """)

# Launch configuration
if __name__ == "__main__":
    print("🚀 Starting MNIST Digit Classifier...")
    
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public link
        # debug=True,             # Enable debug mode
        show_error=True,        # Show errors in interface
        # inbrowser=True,         # Open in browser automatically
    )