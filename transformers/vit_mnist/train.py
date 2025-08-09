import torch
from torch import nn
from torch.optim import Adam
from typing import Tuple, Dict, List

@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: str = "cpu") -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total)

def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device: str = "cpu") -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total)

def fit(model: nn.Module, train_loader, test_loader, epochs: int, lr: float, weight_decay: float,
        device: str = "cpu", callbacks: Dict = None) -> Dict[str, List[float]]:
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss);  history["test_acc"].append(te_acc)

        if callbacks and "on_epoch_end" in callbacks:
            callbacks["on_epoch_end"](epoch, model, history)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
              f"test loss {te_loss:.4f} acc {te_acc:.2f}%")
    return history