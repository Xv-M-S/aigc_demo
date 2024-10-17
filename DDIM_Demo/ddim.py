import torch
from tqdm import tqdm

from ddpm import DDPM


class DDIM(DDPM):

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        super().__init__(device, n_steps, min_beta, max_beta)

    def sample_backward(self,
                        img_or_shape,
                        net,
                        device,
                        simple_var=True,
                        ddim_step=20, # 表示执行几轮去噪迭代
                        eta=1): # 表示DDPM和DDIM的插值系数
        # 在开始迭代前，要做一些预处理。根据论文的描述，如果使用了DDPM的那种简单方差，一定要令eta=1
        if simple_var:
            eta = 1
        # 我们会用到从self.n_steps到0等间距的ddim_step+1个时刻（self.n_steps是初始时刻，不在去噪迭代中）。
        # 比如总时刻self.n_steps=100，ddim_step=10，ts数组里的内容就是[100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]。
        ts = torch.linspace(self.n_steps, 0,
                            (ddim_step + 1)).to(device).to(torch.long)
        if isinstance(img_or_shape, torch.Tensor):
            x = img_or_shape
        else:
            # 如果 img_or_shape 不是张量（通常意味着它是一个形状规格，如元组），则生成一个随机张量。
                # torch.randn(img_or_shape)创建一个填充有来自标准正态分布（均值为 0，标准差为 1）的随机数的张量。
                # 张量的大小由 img_or_shape 指定。
            x = torch.randn(img_or_shape).to(device)
        batch_size = x.shape[0]
        net = net.to(device)
        for i in tqdm(range(1, ddim_step + 1),
                      f'DDIM sampling with eta {eta} simple_var {simple_var}'):
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1

            ab_cur = self.alpha_bars[cur_t]
            ab_prev = self.alpha_bars[prev_t] if prev_t >= 0 else 1
            # 给每个样本添加一个 cur_t 采样步数标签
            # 这部分代码创建一个一维张量，其元素为 cur_t，重复 batch_size 次。
            t_tensor = torch.tensor([cur_t] * batch_size,
                                    dtype=torch.long).to(device).unsqueeze(1)
            eps = net(x, t_tensor)
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(x)

            first_term = (ab_prev / ab_cur)**0.5 * x
            second_term = ((1 - ab_prev - var)**0.5 -
                           (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev)**0.5 * noise
            else:
                third_term = var**0.5 * noise
            x = first_term + second_term + third_term

        return x