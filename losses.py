# -*- coding: utf-8 -*-

# List of loss function that can be used here: MSE, L1, Smooth L1, Perceptual Loss, Adversarial Loss, SSIM, MS_SSIM, combination of several losses

class SSIMLoss(nn.Module):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)
        C1 = (self.k1 * 1) ** 2
        C2 = (self.k2 * 1) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()

# img1 = Variable(torch.rand(1, 1, 256, 256))
# img2 = Variable(torch.rand(1, 1, 256, 256))

# if torch.cuda.is_available():
#     img1 = img1.cuda()
#     img2 = img2.cuda()
# ssim_loss = SSIMLoss()
# print(ssim_loss(img1, img1)),print(ssim_loss(img1, img2))