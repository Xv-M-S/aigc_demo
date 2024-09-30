import torch

class DDPM():

    # n_steps 就是论文里的 T
    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        # 从 min_beta 到 max_beta 线性地生成 n_steps 个时刻的 beta 值
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        # torch.empty_like(alphas) 的作用是创建一个与 alphas 张量形状相同但未初始化的张量。
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        # ???
        alpha_prev = torch.empty_like(alpha_bars)
        alpha_prev[1:] = alpha_bars[0:n_steps - 1]
        alpha_prev[0] = 1
        self.coef1 = torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars)
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1 - alpha_bars)
        
    """
    前向过程
    输入：x（512，1，28，28） t(512,1)  eps(512,1,28,28)
    """
    def sample_forward(self, x, t, eps=None):
        # 将从 self.alpha_bars 中选择的 alpha_bar 值重塑为适合与输入张量 x 进行广播的形状。
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            # torch.randn_like(x) 是 PyTorch 中的一个函数，用于生成与给定张量 x 形状相同的张量，其中的元素是从标准正态分布（均值为 0，方差为 1）中随机抽取的。
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

    """
    反向过程
    """
    def sample_backward(self, img_shape, net, device, simple_var=True, clip_x0=True):
        # 这个函数用于生成一个张量，张量中的元素是从标准正态分布（均值为 0，标准差为 1）中随机抽取的。
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var, clip_x0)
        return x
    def sample_backward_step(self, x_t, t, net, simple_var=True, clip_x0=True):
        # 获取batch_size n
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)

        if t == 0:
            # t=0 时，没有方差项
            noise = 0
        else:
            # 方差项用到的方差有两种取值，效果差不多，我们用simple_var来控制选哪种取值方式。
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)
        
        if clip_x0:
            x_0 = (x_t - torch.sqrt(1 - self.alpha_bars[t]) *
                   eps) / torch.sqrt(self.alpha_bars[t])
            """
                将 x_0 的值限制在 -1 到 1 的范围内。任何小于 -1 的值会被设为 -1，任何大于 1 的值会被设为 1。
                这一步是为了确保 x_0 的数值稳定性，防止数值过大或过小导致的问题。
            """
            x_0 = torch.clip(x_0, -1, 1)
            mean = self.coef1[t] * x_t + self.coef2[t] * x_0
        else:
            mean = (x_t -
                    (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                    eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t
def visualize_forward():
    import cv2
    import einops
    import numpy as np

    from dataset import get_dataloader

    n_steps = 100
    device = 'cuda'
    dataloader = get_dataloader(5)
    x, _ = next(iter(dataloader))
    x = x.to(device)

    ddpm = DDPM(device, n_steps)
    xts = []
    percents = torch.linspace(0, 0.99, 10)
    for percent in percents:
        t = torch.tensor([int(n_steps * percent)])
        t = t.unsqueeze(1)
        x_t = ddpm.sample_forward(x, t)
        xts.append(x_t)
    res = torch.stack(xts, 0)
    res = einops.rearrange(res, 'n1 n2 c h w -> (n2 h) (n1 w) c')
    res = (res.clip(-1, 1) + 1) / 2 * 255
    res = res.cpu().numpy().astype(np.uint8)

    cv2.imwrite('work_dirs/diffusion_forward.jpg', res)

def main():
    visualize_forward()

if __name__ == '__main__':
    main()