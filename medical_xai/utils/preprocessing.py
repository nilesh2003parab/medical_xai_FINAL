"""
Preprocessing utilities for chest X-ray images.
Standard pipeline: Resize → Center Crop → Normalize (ImageNet stats).
"""

from torchvision import transforms


# ImageNet mean / std — used for ResNet18 pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Chest X-ray specific mean / std (approx. from NIH ChestX-ray14)
XRAY_MEAN = [0.5056, 0.5056, 0.5056]
XRAY_STD  = [0.2521, 0.2521, 0.2521]


def get_transform(image_size: int = 224, augment: bool = False):
    """
    Returns a torchvision transform pipeline.

    Args:
        image_size: Target square size after resize/crop.
        augment:    If True, apply training-time augmentations.

    Returns:
        torchvision.transforms.Compose object
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: Normalized tensor [C, H, W] or [B, C, H, W]

    Returns:
        Denormalized tensor, values in [0, 1]
    """
    import torch
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std).view(3, 1, 1)
    if tensor.dim() == 4:
        mean_t = mean_t.unsqueeze(0)
        std_t  = std_t.unsqueeze(0)
    return (tensor * std_t + mean_t).clamp(0, 1)
