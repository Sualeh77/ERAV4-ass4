from config import device, model_path
import torch
from tqdm import tqdm
from utils import GetCorrectPredCount
from torch import nn
import torch
from torch.utils.data import DataLoader


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    pbar = tqdm(dataloader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch, (image, label) in enumerate(dataloader):
        image, label = image.to(device), label.to(device)

        # Compute prediction error
        pred = model(image)
        loss = loss_fn(pred, label)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct += GetCorrectPredCount(pred, label)
        processed += len(image)
        pbar.set_description(desc= f'Train: Loss={loss.item():>7f} Batch_id={batch+1} Accuracy={100*correct/processed:0.2f}')

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)
            pred = model(image)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def trainer(epochs:int, train_loader:DataLoader, test_loader:DataLoader, model:nn.Module, loss_fn:nn.Module, optimizer:torch.optim.Optimizer):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)
    print("Done!")
    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch Model State to {model_path}")