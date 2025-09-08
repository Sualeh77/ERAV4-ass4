# 🔢 MNIST Digit Classifier with AWS Lambda Deployment

A fully convolutional neural network for MNIST digit classification with under 25k parameters, deployed as a serverless web application using AWS Lambda and Gradio.

## 🎯 Project Overview

This project implements a lightweight CNN model that achieves >97% accuracy on MNIST digit classification with less than 25,000 parameters. The model is deployed as a serverless web application using AWS Lambda, providing a cost-effective and scalable solution for digit recognition.

## 🏗️ Architecture

### Model Architecture
- **Type**: Fully Convolutional Neural Network (CNN)
- **Parameters**: ~18k parameters (under 25k limit)
- **Accuracy**: >97% on MNIST test set
- **Framework**: PyTorch
- **Input**: 28x28 grayscale images
- **Output**: 10-class classification (digits 0-9)

### Deployment Architecture
- **Frontend**: Gradio web interface
- **Backend**: AWS Lambda Function
- **Container**: Docker image with PyTorch CPU
- **Infrastructure**: AWS CDK (Infrastructure as Code)
- **Web Adapter**: AWS Lambda Web Adapter for HTTP handling

## 📁 Project Structure

```
train_deploy_mnist_cnn/
├── model.py                 # CNN model definition
├── config.py               # Configuration settings
├── dataset.py              # Dataset handling
├── train.py                # Model training script
├── gradio_app.py           # Gradio web interface
├── Dockerfile              # Docker container definition
├── cdk.py                  # AWS CDK infrastructure
├── requirements.txt        # Python dependencies
├── models/
│   └── mnist_fully_cnn.pth # Trained model weights
├── data/                   # MNIST dataset
└── mnist_images/           # Processed image data
```

This comprehensive README covers:

✅ **Project Overview** - Clear description and goals  
✅ **Architecture Details** - Both model and deployment architecture  
✅ **Setup Instructions** - Local, Docker, and AWS deployment  
✅ **AWS Lambda Deployment** - Complete deployment guide  
✅ **Technical Challenges** - Solutions to Lambda-specific issues  
✅ **Performance Metrics** - Model and Lambda performance data  
✅ **Usage Examples** - How to use both upload and draw features  
✅ **Future Enhancements** - Roadmap for improvements  

You can copy this content and paste it into your README.md file. The documentation reflects the current state of your project with the working upload functionality and the Lambda-optimized Sketchpad configuration.

## 🚀 Features

### Web Interface
- **Upload**: Support for image files, webcam capture, and clipboard
- **Draw**: Interactive drawing pad for digit sketching
- **Real-time Prediction**: Instant classification results
- **Probability Visualization**: Horizontal bar chart showing confidence scores
- **Top-3 Predictions**: Display of most likely digits with probabilities

### Model Features
- **Lightweight**: Under 25k parameters for fast inference
- **High Accuracy**: >97% accuracy on MNIST test set
- **CPU Optimized**: Runs efficiently on Lambda's CPU-only environment
- **Preprocessing**: Automatic image normalization and contrast adjustment

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.12+
- Docker
- AWS CLI configured
- AWS CDK v2
- Node.js (for CDK)

### Local Development

1. **Clone the repository**
```bash
git clone <repository-url>
cd train_deploy_mnist_cnn/train_deploy_mnist_cnn
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run locally**
```bash
python gradio_app.py
```
Access at `http://localhost:8080`

### Docker Deployment

1. **Build Docker image**
```bash
docker build -t mnist-gradio-app .
```

2. **Run container**
```bash
docker run -p 8080:8080 mnist-gradio-app
```

## ☁️ AWS Lambda Deployment

### Prerequisites
- AWS CLI configured with appropriate permissions
- AWS CDK v2 installed
- Docker for building images

### Deployment Steps

1. **Install CDK dependencies**
```bash
npm install -g aws-cdk
```

2. **Bootstrap CDK (first time only)**
```bash
cdk bootstrap
```

3. **Deploy to AWS**
```bash
cdk deploy
```

4. **Access your application**
The deployment will output a function URL that you can access directly.

### Infrastructure Components

The CDK deployment creates:
- **Lambda Function**: 2GB memory, 15-minute timeout
- **Function URL**: Public HTTPS endpoint
- **Docker Image**: Stored in ECR automatically
- **IAM Roles**: Appropriate permissions for Lambda execution

### Configuration Details

```python
# Lambda Configuration
memory_size=2048,              # 2GB for ML model loading
timeout=Duration.minutes(15),  # Extended timeout for cold starts
architecture=Architecture.X86_64,
environment={
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "GRADIO_SERVER_NAME": "0.0.0.0",
    "GRADIO_SERVER_PORT": "8080",
    "GRADIO_TEMP_DIR": "/tmp/gradio",
    "TMPDIR": "/tmp",
}
```

## 🧠 Model Details

### Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: Adaptive scheduling
- **Batch Size**: 64
- **Epochs**: Trained until convergence
- **Data Augmentation**: Random rotation, normalization

### Model Performance
- **Test Accuracy**: >97%
- **Parameters**: ~18,000
- **Inference Time**: <100ms on Lambda
- **Model Size**: <1MB

## 🎨 Usage

### Web Interface

1. **Upload Method**:
   - Click "Upload Image" tab
   - Upload image file, use webcam, or paste from clipboard
   - Click "🔍 Predict from Upload"

2. **Drawing Method**:
   - Click "Draw Digit" tab
   - Draw a digit using the interactive pad
   - Click "🔍 Predict from Drawing"

3. **Results**:
   - View processed 28x28 image
   - See prediction confidence scores
   - Analyze probability distribution chart

### API Usage

The Lambda function URL accepts HTTP requests and can be integrated into other applications.

## 🔧 Technical Challenges & Solutions

### Lambda-Specific Optimizations

1. **File System Issues**
   - **Problem**: Gradio's temporary file handling in serverless environment
   - **Solution**: Configured Sketchpad with `type="numpy"` for in-memory processing

2. **Cold Start Optimization**
   - **Problem**: Model loading time on first request
   - **Solution**: Increased Lambda memory to 2GB and timeout to 15 minutes

3. **Dependencies**
   - **Problem**: Large PyTorch package size
   - **Solution**: CPU-only PyTorch installation with optimized requirements

### Model Optimizations

1. **Parameter Efficiency**
   - Fully convolutional architecture
   - Shared feature extraction layers
   - Minimal fully connected layers

2. **Inference Speed**
   - CPU-optimized operations
   - Efficient tensor operations
   - Minimal preprocessing overhead

## 📊 Performance Metrics

### Model Performance
- **Accuracy**: >97% on test set
- **Precision**: >97% average across all digits

### Lambda Performance
- **Cold Start**: ~9 seconds (first request)
- **Warm Requests**: 50-200ms
- **Memory Usage**: ~550MB peak
- **Cost**: Pay-per-request pricing
- **Consistency Issue**: Multiple inferences behaves weird, May because of AWS lambda without any storage is not able to handle files properly: (Suggestions to fix this are welcome!)

## 🛡️ Security & Best Practices

- **Model Security**: `weights_only=True` for safe model loading
- **Environment Variables**: Proper configuration for Lambda environment
- **Error Handling**: Graceful degradation for edge cases
- **Resource Management**: Efficient memory usage and cleanup

## 🚀 Future Enhancements

- [ ] Model quantization for smaller size
- [ ] Support for additional datasets
- [ ] Batch prediction API
- [ ] Model versioning and A/B testing
- [ ] Enhanced monitoring and logging
- [ ] Multi-region deployment

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For questions or issues, please open an issue in the repository or contact the development team.

---

**Built with ❤️ using PyTorch, Gradio, and AWS Lambda**
