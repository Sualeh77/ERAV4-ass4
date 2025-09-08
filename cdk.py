import os
from pathlib import Path
from constructs import Construct
from aws_cdk import App, Stack, Environment, Duration, CfnOutput
from aws_cdk.aws_lambda import (
    DockerImageFunction,
    DockerImageCode,
    Architecture,
    FunctionUrlAuthType,
)

my_environment = Environment(
    account=os.environ["CDK_DEFAULT_ACCOUNT"], region=os.environ["CDK_DEFAULT_REGION"]
)

class MnistInferenceGradioLambda(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create Lambda function
        lambda_fn = DockerImageFunction(
            self,
            "MnistClassifier",
            code=DockerImageCode.from_image_asset(
                str(Path.cwd()),
                file="Dockerfile"
            ),
            architecture=Architecture.X86_64,
            memory_size=2048,  # 2GB memory for ML model loading
            timeout=Duration.minutes(15),  # Increased timeout for cold starts
            environment={
                "MPLCONFIGDIR": "/tmp/matplotlib",  # Fix matplotlib config directory
                "GRADIO_SERVER_NAME": "0.0.0.0",   # Ensure Gradio binds to all interfaces
                "GRADIO_SERVER_PORT": "8080",      # Standard port for Lambda Web Adapter
            }
        )

        # Add HTTPS URL
        fn_url = lambda_fn.add_function_url(auth_type=FunctionUrlAuthType.NONE)

        CfnOutput(self, "functionUrl", value=fn_url.url)

app = App()
mnist_inference_gradio_lambda = MnistInferenceGradioLambda(app, "MnistInferenceGradioLambda", env=my_environment)
app.synth()