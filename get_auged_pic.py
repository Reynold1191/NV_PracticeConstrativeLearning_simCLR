import os
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import random
from main5 import get_simclr_transforms, GaussianBlur


def save_images(batch, save_dir, index):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    img, img1, img2 = batch

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))

    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")


    ax[1].imshow(T.ToPILImage()(img1))
    ax[1].set_title("Augmented 1")
    ax[1].axis("off")

    ax[2].imshow(T.ToPILImage()(img2))
    ax[2].set_title("Augmented 2")
    ax[2].axis("off")
    plt.savefig(os.path.join(save_dir, f"aug_imgs_{i+1}.jpg"))  # Save augmented images
    plt.show()

# Prepare DataLoader
# transform = get_simclr_transforms()
# train_ds = CIFAR10Pairs(root="./data", train=True, transform=transform, download=True)
# train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

dataset = datasets.CIFAR10(root="data", train=True, download=True)
for i in range(2):
    image, _ = dataset[random.randint(0, len(dataset)-1)]  # Randomly select an image
    transform = get_simclr_transforms()
    img1 = transform(image)
    img2 = transform(image)
    batch = (image, img1, img2)
    # Iterate over the data loader and save a few images
    save_dir = "./augmented_images"
    save_images(batch, save_dir, i)

