import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, ToTensor
def download_dataset():
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True)
    print('length of MNIST', len(mnist))
    id = 4
    img, label = mnist[id]
    print(img)
    print(label)

    # On computer with monitor
    # img.show()

    # img.save('work_dirs/tmp.jpg')
    # 每一张图片都是单通道图片（灰度图），颜色值的取值范围是0~1。
    tensor = ToTensor()(img)
    print(tensor.shape)
    print(tensor.max())
    print(tensor.min())

def get_dataloader(batch_size: int):
    # 由于DDPM会把图像和正态分布关联起来，我们更希望图像颜色值的取值范围是[-1, 1]。为此，我们可以对图像做一个线性变换，减0.5再乘2。
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(root='./data/mnist',
                                         transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
def get_img_shape():
    return (1, 28, 28)
    
if __name__ == '__main__':
    download_dataset()