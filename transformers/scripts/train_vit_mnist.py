# Lightweight script to train from terminal
import time
import sys
from pathlib import Path
import torch
# Ensure the sibling package 'vit_mnist' is importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vit_mnist import Config, ViT, get_dataloaders, fit
from vit_mnist.viz import visualize_predictions, plot_loss_curves

def main():
    cfg = Config()
    device = torch.device(cfg.train.device if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu")
    train_loader, test_loader = get_dataloaders(
        root=cfg.data.root, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
        target_size=cfg.data.target_size, make_rgb=cfg.data.make_rgb
    )
    model = ViT(**cfg.model.__dict__)
    tag = time.strftime("%Y%m%d_%H%M%S")

    # Fixed preview set
    fixed_imgs, fixed_labels = next(iter(test_loader))
    fixed_imgs = fixed_imgs[:10]; fixed_labels = fixed_labels[:10]

    # Per-epoch callback: visualize predictions and save loss curves
    def on_epoch_end(epoch, model, history):
        visualize_predictions(model, fixed_imgs, fixed_labels, device=str(device), epoch=epoch, save_path=cfg.save_path.as_posix())
        plot_loss_curves(history["train_loss"], history["test_loss"], history["train_acc"], history["test_acc"], cfg.save_path.as_posix(), tag, epoch)

    # Train the model
    hist = fit(
        model, train_loader, test_loader,
        epochs=cfg.train.epochs, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        device=str(device), callbacks={"on_epoch_end": on_epoch_end}
    )

    # Save final model checkpoint
    ckpt_path = cfg.save_path / f"vit_mnist_{tag}_final.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_class": "ViT",
            "model_args": cfg.model.__dict__,
            "history": hist,
            "tag": tag,
        },
        ckpt_path,
    )
    print(f"Saved model checkpoint to: {ckpt_path}")

if __name__ == "__main__":
    main()