import torch  # 入PyTorch深度学习框架的语句。
import torch.nn as nn  # 导入PyTorch深度学习库中的nn模块，其中包含了神经网络相关的类和函数。
import torch.nn.functional as F  # 导入torch库中的nn模块中的functional子模块，并将其命名为F。


class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        """
        gamma是幂指数，越大表示对难分样本的权重越大
        alpha是平衡因子，取值在[0,1]之间
        """
        super(FocalBCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets, inf=1e-9):
        """
        计算 Focal BCE 损失

        参数:
        - inputs: 模型的输出
        - targets: 真实标签
        - inf: 避免对数的小常数
        """

        # 检查输入维度是否正确
        # assert inputs.size(1) == 2, "The model should output logits for two classes."
        # # 使用 Softmax 将输出转换为概率
        # masks_pred = F.softmax(inputs, dim=1)  # B,C,H,W
        masks_pred = inputs
        # 获取正类别的概率
        # masks_pred = inputs_softmax[:, 1]
        true_masks = targets
        alpha = self.alpha

        loss = - alpha * (1 - masks_pred) ** self.gamma * true_masks * torch.log(abs(masks_pred)+inf) - \
            (1 - alpha) * masks_pred ** self.gamma * \
            (1 - true_masks) * torch.log(abs(1 - masks_pred)+inf)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss
