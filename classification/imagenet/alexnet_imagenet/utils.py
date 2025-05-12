######
#Created on Tue Sep 21 12:10:58 2021
#
#@author: veronesi
######

import torch
from torch import nn
from torch import softmax


#################
## Test Network
#################
def test(dataloader, model, loss_fn=nn.CrossEntropyLoss(), verbose=True, complete=True, sample_size=1000, device='cpu'):
    model = model.to(device)
    model.eval()
    num_batches = len(dataloader)


    try:
        size = len(dataloader.dataset)
    except AttributeError:
        size = sum(len(batch) for batch, _ in dataloader)  # Safe fallback

    test_loss, correct = 0, 0

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            prob = softmax(pred, dim=1)  # Fixed softmax usage
            test_loss += loss_fn(prob, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if not complete and i == sample_size:
                break

    test_loss /= num_batches
    correct /= size

    if verbose:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct, test_loss

############
## Clipper
############
class WeightClipper:
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)
            module.weight.data=w

###############
## Train Step
###############
def train_step(dataloader, model, loss_fn, optimizer, device):

    model = model.to(device)
    size=len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        #Prediction error
        pred = model(X)
        prob = softmax(pred, 1)
        loss = loss_fn(prob, y)
        
        #BackPropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # #Clipper
        # clipper = WeightClipper()
        # model.apply(clipper)
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

################
## Train Model
################
def train_model(model, dataloader, epochs, loss_fn, optimizer, scheduler=None, device='cuda'):
    model = model.to(device)
    model.train()

    # Training
    print("\nFP32 Training Begin...")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_step(dataloader, model, loss_fn, optimizer, device)
        test(dataloader, model, loss_fn, device)
        if scheduler is not None:
            scheduler.step()

    print("FP32 Training Complete\n")
