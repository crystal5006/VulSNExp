import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing


class VulSNExp(nn.Module):
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

    def __init__(self, model, epochs=800, lr=0.05, gam=0.0, lam=1.0, alp=0.5, sparse_coeff=0.1, group_lambda=0.01,
                 inject_mode: str = "auto"):
        super(VulSNExp, self).__init__()
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
        # inject_mode: 'auto' | 'internal' | 'edge_weight'
        self.inject_mode = inject_mode

    def _initialize_mask(self, num_nodes):
        # 初始化一个 [num_nodes, num_nodes] 的掩码参数
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(2.0 / (num_nodes + num_nodes))
        self.adj_mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes).normal_(0.0, 0.1).to(self.device))
        # self.adj_mask = nn.Parameter(torch.rand(num_nodes, num_nodes).to(self.device) * 0.1)
        # self.adj_mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes).to(self.device))
        nn.init.kaiming_normal_(self.adj_mask, mode='fan_out', nonlinearity='relu')

    def get_masked_edge_weights(self, data, mask):
        """
        保留：如需回退到 edge_weight 方式可用。本实现采用内部掩码注入，不依赖该函数。
        """
        edge_index = data.edge_index
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            orig_weights = data.edge_attr.view(-1)
        else:
            orig_weights = torch.ones(edge_index.size(1), device=self.device)
        m = mask[edge_index[0], edge_index[1]]
        new_weights = orig_weights * m
        return new_weights

    def compute_loss(self, mask, pred_factual, pred_counter, class_idx: int = 1):
        """
        计算目标函数：
            loss = L1_norm(mask) + lam * [ alp * factual_loss + (1-alp) * counterfactual_loss ]
        其中：
            - factual_loss = ReLU(gam + threshold - pred_factual_vuln)
            - counterfactual_loss = ReLU(gam + pred_counter_vuln - threshold)
        """
        L1_loss = torch.norm(mask, p=1) * self.sparse_coeff  # 稀疏正则
        vuln_pred_factual = pred_factual[:, class_idx]
        vuln_pred_counter = pred_counter[:, class_idx]

        # 对抗损失（事实/反事实）
        factual_loss = F.relu(self.gam + self.threshold - vuln_pred_factual)
        counter_loss = F.relu(self.gam + vuln_pred_counter - self.threshold)
        adv_loss = L1_loss + self.lam * (self.alp * factual_loss + (1 - self.alp) * counter_loss)

        current_lam = self.lam * (L1_loss.item() / (adv_loss.mean().item() + 1e-8))
        current_lam = max(min(current_lam, 2000), 0.1)  # 限制范围防止极端值

        # Group Lasso 正则（按行分组）
        group_loss = torch.sum(torch.norm(mask, p=2, dim=1))  # 对每行求L2范数

        # 总损失
        loss = L1_loss + current_lam * adv_loss.mean() + self.group_lambda * group_loss

        return loss, L1_loss, adv_loss.mean(), current_lam

    def _apply_edge_mask(self, edge_mask_vec, edge_index):
        """将边掩码注入到所有 MessagePassing 模块，启用 explain 模式。"""
        loop_mask = edge_index[0] != edge_index[1]
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = True
                module._edge_mask = edge_mask_vec
                module._loop_mask = loop_mask
                module._apply_sigmoid = False  # 已经是 [0,1] 概率

    def _clear_edge_mask(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = None
                module._apply_sigmoid = True

    def _supports_edge_weight(self):
        """Return True if any GNN layer likely supports edge_weight (GCN/GraphConv)."""
        # supported = {"GCNConv", "GraphConv"}
        supported = {"GCNConv"}
        for module in self.model.modules():
            name = module.__class__.__name__
            if name in supported:
                return True
        return False

    def _contains_layer(self, names: set):
        """Check whether the backbone contains any layer class names in `names`."""
        for module in self.model.modules():
            if module.__class__.__name__ in names:
                return True
        return False

    def forward(self, data, target_label=None):
        num_nodes = data.x.size(0)
        self._initialize_mask(num_nodes)
        # 将 threshold 纳入优化，使阈值可学习
        optimizer = torch.optim.Adam([self.adj_mask, self.threshold], lr=self.lr)

        # # 添加学习率衰减策略，这里使用 StepLR，每 100 个 epoch 学习率乘以 0.9
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1, eta_min=1e-4
        )

        # 构造原始图的密集邻接矩阵 A（用于构建反事实掩码的补集）
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
        # 决定注入策略（GIN 一律仅用内部注入，避免与自环/自项交互引发的干扰）
        gin_like = self._contains_layer({"GINConv"})
        if gin_like:
            use_edge_weight = False
        elif self.inject_mode == "edge_weight":
            use_edge_weight = True
        elif self.inject_mode == "internal":
            use_edge_weight = False
        else:  # auto
            use_edge_weight = self._supports_edge_weight()

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            # 非对称掩码（N x N）
            mask = torch.sigmoid(self.adj_mask)
            # 提取与边对齐的一维掩码
            edge_mask_vec = mask[edge_index[0], edge_index[1]].clamp(0.0, 1.0)

            # 对于单图情况，batch 全部为0
            batch = torch.zeros(num_nodes, dtype=torch.long, device=self.device)

            if use_edge_weight:
                # 边权模式：将掩码投影为权重
                new_edge_weights = edge_mask_vec
                comp_edge_weights = (1.0 - edge_mask_vec)
                pred_factual = self.model(data.x, data.edge_index, batch, edge_weight=new_edge_weights)
                pred_counter = self.model(data.x, data.edge_index, batch, edge_weight=comp_edge_weights)
            else:
                # 内部注入：注入 m 与 (1-m)
                mask_applied = self._apply_edge_mask(edge_mask_vec, edge_index)
                pred_factual = self.model(data.x, data.edge_index, batch)
                comp_edge_mask_vec = (1.0 - edge_mask_vec).clamp(0.0, 1.0)
                self._apply_edge_mask(comp_edge_mask_vec, edge_index)
                pred_counter = self.model(data.x, data.edge_index, batch)

            class_idx = int(target_label) if target_label is not None else 1
            loss, L1_loss, adv_loss, current_lam = self.compute_loss(mask, pred_factual, pred_counter,
                                                                     class_idx=class_idx)

            loss.backward()
            optimizer.step()
            # 清理掩码，避免影响下一次前向
            if not use_edge_weight:
                self._clear_edge_mask()
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

        # 导出边级解释：使用最佳掩码投影到边（与评估接口兼容）
        best_edge_weights = best_mask[edge_index[0], edge_index[1]].detach().clone()
        return best_edge_weights

