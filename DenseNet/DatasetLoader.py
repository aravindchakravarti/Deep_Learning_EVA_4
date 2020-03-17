from torchvision import datasets, transforms
import torch

from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, OneOf, Cutout, VerticalFlip, Rotate
from albumentations import ShiftScaleRotate
from albumentations.pytorch import ToTensor
from PIL import ImageFile, Image
import numpy as np

def getCifar10Data():
  train_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                      #  transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                       transforms.RandomAffine(degrees=10, shear = 10),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])

  # Test Phase transformations
  test_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                       ])

  print ('Now downloading and allocating dataset')

  # Do we have CUDA drivers for us?
  cuda = torch.cuda.is_available()
  print ("Cuda Available?", cuda)

  train = datasets.CIFAR10(root = './data', train=True, download=True, transform=train_transforms)
  test = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

  dataloader_args = dict(shuffle=True, batch_size=512, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

  print ('Now allocating Dataloaders')

  # Dataloaders
  train_loader = torch.utils.data.DataLoader(dataset=train, **dataloader_args)
  test_loader = torch.utils.data.DataLoader(dataset=test, **dataloader_args)

  return train_loader, test_loader
  
def cifar10WithAlbumentations(batch_size = 512):
    print ('Building up with Albumentations - 1v3')
    def strong_aug(p=.5):
        return Compose([
            HorizontalFlip(),
            VerticalFlip(),
            ShiftScaleRotate(rotate_limit=30), 
            # RandomGridShuffle(grid=(3, 3)),
            # RandomGridShuffle is unfortunately not getting imported
            Cutout(num_holes=2, max_h_size=8, max_w_size=8),
            Rotate(limit=30) 
            # Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            # I am doing Normalization and ToTensor using Torch default tansform module
            ], p=p)

    def augment(aug, image):
        return aug(image=image)['image']

    class MyTransform(object):
        def __call__(self, img):
            aug = strong_aug(p=0.9)
            return Image.fromarray(augment(aug, np.array(img)))

    train_transforms = transforms.Compose([
                                        MyTransform(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ])
    
    print ('Now downloading and allocating dataset')

    # Do we have CUDA drivers for us?
    cuda = torch.cuda.is_available()
    print ("Cuda Available?", cuda)

    train = datasets.CIFAR10(root = './data', train=True, download=True, transform=train_transforms)
    test = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    print ('Now allocating Dataloaders')

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(dataset=test, **dataloader_args)

    return train_loader, test_loader


def imshow(img):
  img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))


  # get some random training images
  dataiter = iter(train_loader)
  images, labels = dataiter.next()

  # show images <Enable below instructions if data visualization required>
  imshow(torchvision.utils.make_grid(images))
  # print labels
  print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
