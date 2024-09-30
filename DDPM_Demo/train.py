import os
import time

import torch
import torch.nn as nn
import cv2
import numpy as np
import einops

from dataset import get_dataloader, get_img_shape
from ddpm import DDPM
from network import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)


batch_size = 512
n_epochs = 100


def train(ddpm: DDPM, net, device='cuda', ckpt_path='model/model.pth'):
    print('batch size:', batch_size)
    # n_steps 就是公式里的 T
    # net 是某个继承自 torch.nn.Module 的神经网络
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)
    
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        
        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            # t的采样：指定生成的张量的形状为一维，长度为 current_batch_size。这意味着将生成 current_batch_size 个随机整数。
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)
            # 高斯噪声的采样： torch.randn_like(x)生成一个和训练图片x形状一样的符合标准正态分布的图像。
            eps = torch.randn_like(x).to(device)
            # x（512，1，28，28） t(512,1)  eps(512,1,28,28)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))
            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')

def sample_imgs(ddpm,
                net,
                output_path,
                n_sample=81,
                device='cuda',
                simple_var=True):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        # 星号运算符用于解包元组 get_img_shape() 的内容。
        shape = (n_sample, *get_img_shape())  # shape = 81, 1, 28, 28
        imgs = ddpm.sample_backward(shape,
                                    net,
                                    device=device,
                                    simple_var=simple_var).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        """
        clamp(min, max)：
            clamp 方法会将张量中的所有元素限制在给定的最小值（min）和最大值（max）之间。
            如果某个元素小于 min，它将被设置为 min；如果大于 max，则被设置为 max。
        """
        imgs = imgs.clamp(0, 255)
        # 81, 3, 28, 28 -> 252, 252, 3
        # 为了更好的展示图像，一次展示9x9个图像
        imgs = einops.rearrange(imgs,
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample**0.5))

        imgs = imgs.numpy().astype(np.uint8)

        cv2.imwrite(output_path, imgs)
        
configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]

if __name__ == '__main__':
    n_steps = 1000
    config_id = 4
    device = 'cuda'
    model_path = 'model/model_unet_res.pth'

    config = configs[config_id]
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)

    # train(ddpm, net, device=device, ckpt_path=model_path)

    net.load_state_dict(torch.load(model_path))
    sample_imgs(ddpm, net, 'diffusion.jpg', device=device)