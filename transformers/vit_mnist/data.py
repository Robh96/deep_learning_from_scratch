import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F

def preprocess_mnist_for_vit(images: torch.Tensor, target_size=224, make_rgb=True):
    # images: [B, C, H, W] or [C, H, W]
    is_batched = images.dim() == 4
    if not is_batched:
        images = images.unsqueeze(0)
    if target_size != 28:
        images = F.interpolate(images, size=(target_size, target_size), mode="bilinear", align_corners=False)
    if make_rgb and images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return images if is_batched else images.squeeze(0)

class VitTransform:
    def __init__(self, target_size=224, make_rgb=True):
        self.target_size = int(target_size)
        self.make_rgb = bool(make_rgb)
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize((0.5,), (0.5,))

    def __call__(self, img):
        x = self.to_tensor(img)
        x = self.norm(x)
        x = preprocess_mnist_for_vit(x, target_size=self.target_size, make_rgb=self.make_rgb)
        return x

def vit_transform(target_size=224, make_rgb=True):
    # return a picklable callable instead of a local closure
    return VitTransform(target_size=target_size, make_rgb=make_rgb)

def get_dataloaders(root="../datasets", batch_size=64, num_workers=2, target_size=224, make_rgb=True):
    tf = vit_transform(target_size=target_size, make_rgb=make_rgb)
    train_ds = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=tf)
    test_ds  = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def get_fixed_test_samples(dataset, num_samples=10):
    idxs = list(range(min(num_samples, len(dataset))))
    imgs, labels = zip(*(dataset[i] for i in idxs))
    return torch.stack(imgs), torch.tensor(labels)