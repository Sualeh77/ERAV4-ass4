import torch
import torch.nn.functional as F
from PIL import Image
import base64
import io
import json
from pathlib import Path
from torchvision import transforms
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your model
from model import MnistFullyCNN

class MNISTPredictor:
    def __init__(self, model_path=None, device=None):
        self.device = device or ('cpu')  # Lambda typically uses CPU
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            self.model = MnistFullyCNN()
            
            # Load state dict
            if isinstance(model_path, (str, Path)):
                state_dict = torch.load(model_path, map_location=self.device)
            else:
                # For Lambda, model might be loaded from memory
                state_dict = model_path
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_input):
        """Preprocess image for prediction"""
        try:
            if isinstance(image_input, str):
                # If it's a file path
                if image_input.startswith('data:image'):
                    # Base64 encoded image
                    header, encoded = image_input.split(',', 1)
                    image_data = base64.b64decode(encoded)
                    image = Image.open(io.BytesIO(image_data))
                else:
                    # File path
                    image = Image.open(image_input)
            elif isinstance(image_input, bytes):
                # Raw bytes
                image = Image.open(io.BytesIO(image_input))
            else:
                # Assume it's already a PIL Image
                image = image_input
            
            # Apply transforms
            tensor_ = self.transform(image) # CxHXW
            tensor_ = tensor_.unsqueeze(0)  # Add batch dimension - NxCxHxW
            
            return tensor_.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_input):
        """Make prediction on image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess image
            tensor_ = self.preprocess_image(image_input)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(tensor_)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return {
                'predicted_digit': predicted_class,
                'confidence': confidence,
                'all_probabilities': probabilities[0].tolist()
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

# Global predictor instance for Lambda (initialized once)
predictor = None

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    global predictor
    
    try:
        # Initialize predictor on first call (cold start)
        if predictor is None:
            # In Lambda, you'd load the model from S3 or include it in the deployment package
            model_path = "/opt/ml/model/mnist_fully_cnn.pth"  # Adjust path as needed
            predictor = MNISTPredictor(model_path)
        
        # Parse the request
        if 'body' in event:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
        else:
            body = event
        
        # Get image from request
        if 'image' in body:
            image_data = body['image']
        elif 'image_base64' in body:
            image_data = body['image_base64']
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image provided'})
            }
        
        # Make prediction
        result = predictor.predict(image_data)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'  # For CORS if needed
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Lambda handler error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def predict_from_path(image_path, model_path):
    """Local function for testing with file paths"""
    predictor = MNISTPredictor(model_path)
    result = predictor.predict(image_path)
    return result

# For local testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MNIST Digit Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, help="Path to model file")
    
    args = parser.parse_args()
    
    # Use default model path if not provided
    if not args.model:
        model_path = Path(__file__).parent / "models" / "mnist_fully_cnn.pth"
    else:
        model_path = args.model
    
    try:
        result = predict_from_path(args.image, model_path)
        print(f"Predicted digit: {result['predicted_digit']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"All probabilities: {[f'{p:.4f}' for p in result['all_probabilities']]}")
    except Exception as e:
        print(f"Error: {e}")