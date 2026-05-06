import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CFFExplainer(nn.Module):
    """
    反事实解释器：基于图级别的解释方法。
    该方法通过学习一个掩码矩阵，对图中边进行扰动，
    同时结合事实解释和反事实解释来平衡解释的充分性和必要性，
    其目标函数由 L1 正则（最小化解释复杂度）与对抗性损失（平衡解释强度）组成。

    参数：
        model: 预训练的漏洞检测模型（GNN），其 forward 方法需支持输入 edge_weight 参数
        epochs: 训练优化掩码的轮数（默认1000）
        lr: 优化学习率（默认0.05，后面采用自衰减策略调整lr）
        gam: 对抗性损失中的偏置参数（默认0.0）
        lam: 对抗性损失项的权重（默认1.0）
        alp: 用于平衡事实解释与反事实解释的比例（默认0.5，取值范围[0,1]）
    """

    def __init__(self, model, epochs=800, lr=0.05, gam=0.0, lam=1.0, alp=0.5, sparse_coeff=0.05, group_lambda=0.8):
        super(CFFExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.gam = gam
        self.lam = lam
        self.alp = alp
        self.sparse_coeff = sparse_coeff
        self.device = next(model.parameters()).device
        self.group_lambda = group_lambda
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def _initialize_mask(self, num_nodes):
        # 初始化一个 [num_nodes, num_nodes] 的掩码参数
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(2.0 / (num_nodes + num_nodes))
        self.adj_mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes).normal_(0.0, 0.1).to(self.device))
        # self.adj_mask = nn.Parameter(torch.rand(num_nodes, num_nodes).to(self.device) * 0.1)
        # self.adj_mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes).to(self.device))
        # nn.init.kaiming_normal_(self.adj_mask, mode='fan_out', nonlinearity='relu')


    def get_masked_edge_weights(self, data, mask):
        """
        根据输入图 data 的 edge_index 和原始边权（若不存在则默认为1），
        从密集矩阵 mask 中提取每条边对应的权值，并与原始边权相乘。
        """
        edge_index = data.edge_index  # [2, E]
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            orig_weights = data.edge_attr.view(-1)
        else:
            orig_weights = torch.ones(edge_index.size(1), device=self.device)
        m = mask[edge_index[0], edge_index[1]]
        new_weights = orig_weights * m
        return new_weights

    def compute_loss(self, mask, pred_factual, pred_counter):
        """
        计算目标函数：
          loss = L1_norm(mask_sym) + lam * [ alp * factual_loss + (1-alp) * counterfactual_loss ]
        其中：
          - factual_loss = ReLU(gam + 0.5 - pred_factual_vuln)
          - counterfactual_loss = ReLU(gam + pred_counter_vuln - 0.5)
        这里以 0.5 作为阈值，具体可根据实际情况调整。
        """
        L1_loss = torch.norm(mask, p=1)  * self.sparse_coeff# 新增系数
        vuln_pred_factual = pred_factual[:, 0]
        vuln_pred_counter = pred_counter[:, 0]
        # 在初始化时添加阈值参数

        # 修改对抗损失计算
        factual_loss = F.relu(self.gam + self.threshold - vuln_pred_factual)
        counter_loss = F.relu(self.gam + vuln_pred_counter - self.threshold)
        adv_loss = L1_loss + self.lam * (self.alp * factual_loss + (1 - self.alp) * counter_loss)

        current_lam = self.lam * (L1_loss.item() / (adv_loss.mean().item() + 1e-8))
        current_lam = max(min(current_lam, 2000), 1)  # 限制范围防止极端值


        # 新增Group Lasso正则（按行分组）
        group_lambda = 0.5  # 可调超参数
        # 动态调整group_lambda（前500轮强正则，后弱正则）
        # dynamic_group_lambda = self.group_lambda * max(0.0, 1.0 - epoch / 500.0)
        group_loss = torch.sum(torch.norm(mask, p=2, dim=1))  # 对每行求L2范数

        # 总损失
        loss = L1_loss + current_lam * adv_loss.mean() + group_lambda * group_loss


        return loss, L1_loss, adv_loss.mean(), current_lam



    def forward(self, data, target_label=None):
        num_nodes = data.x.size(0)
        self._initialize_mask(num_nodes)
        optimizer = torch.optim.Adam([self.adj_mask], lr=self.lr)

        # # 添加学习率衰减策略，这里使用 StepLR，每 100 个 epoch 学习率乘以 0.9
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1, eta_min=1e-4
        )


        # 构造原始图的密集邻接矩阵 A
        A = torch.zeros((num_nodes, num_nodes), device=self.device)
        edge_index = data.edge_index
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            orig_weights = data.edge_attr.view(-1)
        else:
            orig_weights = torch.ones(edge_index.size(1), device=self.device)
        A[edge_index[0], edge_index[1]] = orig_weights

        best_loss = float('inf')
        best_mask = None


        # 训练优化掩码
        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            # 修改后（非对称）：
            mask = torch.sigmoid(self.adj_mask)  # 直接使用非对称掩码
            # 计算扰动后的边权（事实解释）
            new_edge_weights = self.get_masked_edge_weights(data, mask)
            # 计算反事实边权：原始边权减去扰动部分
            comp_edge_weights = self.get_masked_edge_weights(data, (A - mask))

            # 对于单图情况，batch 全部为0
            batch = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
            # 分别计算事实解释和反事实解释下的预测结果
            pred_factual = self.model(data.x, data.edge_index, batch, edge_weight=new_edge_weights)
            pred_counter = self.model(data.x, data.edge_index, batch, edge_weight=comp_edge_weights)

            loss , L1_loss, adv_loss, current_lam= self.compute_loss(mask, pred_factual, pred_counter)


            loss.backward()

            optimizer.step()
            # 在每个 epoch 后更新学习率
            scheduler.step()

            # 记录当前最优结果
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_mask = mask.detach().clone()


            if epoch % 100 == 0:
                # print(f"Epoch {epoch}/{self.epochs}, Loss: {loss.item():.4f}, Best Loss: {best_loss:.4f}")

                print(f"Epoch {epoch}: L1_loss = {L1_loss.item():.4f}, "
                      f"adv_loss = {adv_loss.mean().item():.4f}, "
                      f"current_lam = {current_lam:.4f}, total_loss = {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")
                sparsity = (mask < 0.1).float().mean().item()  # 阈值可调
                print(f"Mask Sparsity: {sparsity:.4f}")



        # 如果训练过程中没有更新 best_mask，则使用最后的掩码
        if best_mask is None:
            best_mask = torch.sigmoid(self.adj_mask)
            best_mask = (best_mask + best_mask.t()) / 2

        final_edge_weights = self.get_masked_edge_weights(data, best_mask)
        return final_edge_weights

