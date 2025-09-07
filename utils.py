from torchsummary import summary
from config import device

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def show_model_summary(model, input_size):
  summary(model, input_size, device=device)