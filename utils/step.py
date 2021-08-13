import torch

def __out__(model, x):
    out = model(x)
    return out

def __get_loss__(model, x, y, loss_func):
    out = __out__(model, x)
    loss = loss_func(out, y)
    return loss

def step(model, device, x, y, loss_func):
    x = x.type(torch.float32).to(device)
    y = y.type(torch.long).to(device)
    loss = __get_loss__(model, x, y, loss_func)
    return loss

def step_without_loss(model, device, x):
    x = x.type(torch.float32).to(device)
    with torch.no_grad():
        out = __out__(model, x)
    return out