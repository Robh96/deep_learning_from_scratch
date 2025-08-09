import torch
import matplotlib.pyplot as plt
from pathlib import Path

@torch.no_grad()
def visualize_predictions(model, images, true_labels, device, epoch, save_path: str):
    model.eval()
    images = images.to(device)
    outputs = model(images)
    preds = outputs.argmax(dim=1)
    probs = torch.softmax(outputs, dim=1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.ravel()
    for i in range(len(images)):
        img = images[i].detach().cpu().permute(1, 2, 0)
        img = (img + 1) / 2  # from [-1,1] to [0,1]
        axes[i].imshow(img)
        pred_label = preds[i].item()
        true_label = true_labels[i].item()
        conf = probs[i][pred_label].item()
        color = "green" if pred_label == true_label else "red"
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {conf:.2f}", color=color, fontsize=10)
        axes[i].axis("off")
    plt.suptitle(f"Epoch {epoch+1} - Fixed Test Sample Predictions", fontsize=16)
    plt.tight_layout()
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_path) / f"predictions_epoch_{epoch+1:02d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)  # ensure file is flushed and resources freed
    return preds, probs

def plot_loss_curves(train_losses, test_losses, train_accs, test_accs, save_path: str, tag: str, epoch: int):
    """
    Save loss/accuracy curves after each epoch with train/test overlaid.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax_loss.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=5)
    ax_loss.plot(epochs, test_losses,  'r-o', label='Test Loss',  linewidth=2, markersize=5)
    ax_loss.set_title('Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    # Accuracy
    ax_acc.plot(epochs, train_accs, 'b-o', label='Train Acc', linewidth=2, markersize=5)
    ax_acc.plot(epochs, test_accs,  'r-o', label='Test Acc',  linewidth=2, markersize=5)
    ax_acc.set_title('Accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    plt.suptitle(f'Training Progress (up to epoch {epoch+1})', fontsize=16)
    plt.tight_layout()
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_path) / f"training_curves_{tag}_epoch_{epoch+1:02d}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)