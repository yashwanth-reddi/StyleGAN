import math
import matplotlib.pylab as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


DATASET = 'faces/allfaces'
BATCH_SIZES = [256,256,256,128,64,32,16]


def get_loader(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5 ],[0.5,0.5,0.5])
        ])
    
    index = int(math.log2(image_size/4))
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    loader = DataLoader(dataset,batch_size = BATCH_SIZES[index],shuffle=True)
    
    return loader, dataset


def check_loader():
    loader,_ = get_loader(256)
    face,_  = next(iter(loader))
    _,ax     = plt.subplots(4,4,figsize=(10,10))
    
    plt.suptitle('Some real samples with 256 X 256 size')
    index = 0
    for i in range(4):
        for j in range(4):
            ax[i][j].imshow((face[index].permute(1,2,0)+1)/2)
            index +=1
            
            