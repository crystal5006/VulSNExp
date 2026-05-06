import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GraphConv, GATConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import *
import torch_scatter

class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()


class GlobalMeanPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_mean_pool(x, batch)


class GlobalAddPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_add_pool(x, batch)


class GlobalMaxPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return torch_scatter.segment_csr(x,
                                         self.cumsum(global_add_pool(torch.ones_like(batch).to(batch.device), batch)))

    def cumsum(self, value):
        out = value.new_empty((value.size(0) + 1,) + value.size()[1:])
        out[0] = 0
        torch.cumsum(value, 0, out=out[1:])
        return out


class Detector(nn.Module):
    def __init__(self, args, **kwargs):
        super(Detector, self).__init__()
        self.args = args

        self.linear = nn.Sequential(
            Linear(args.gnn_feature_dim_size, args.gnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
        )

        self.gnn_layers = torch.nn.ModuleList()
        for i in range(args.num_gnn_layers):
            if args.gnn_model == "GCNConv":
                # gnn_layer = GCNConv(args.gnn_hidden_size, args.gnn_hidden_size, normalize=False)
                gnn_layer = GCNConv(args.gnn_hidden_size, args.gnn_hidden_size)
            elif args.gnn_model == "GatedGraphConv":
                gnn_layer = GatedGraphConv(args.gnn_hidden_size, args.num_ggnn_steps, args.ggnn_aggr)
            elif args.gnn_model == "GINConv":
                # Revert to vanilla GINConv since CFFExplainer now uses internal mask injection
                mlp = Linear(args.gnn_hidden_size, args.gnn_hidden_size)
                gnn_layer = GINConv(mlp, args.gin_eps, args.gin_train_eps)
            elif args.gnn_model == "GraphConv":
                gnn_layer = GraphConv(args.gnn_hidden_size, args.gnn_hidden_size, args.gconv_aggr)
            elif args.gnn_model == "GATConv":
                # Keep hidden size consistent across layers. If concat is True, split channels across heads; otherwise keep as-is.
                if getattr(args, 'gat_concat', False):
                    if args.gnn_hidden_size % args.gat_heads != 0:
                        raise ValueError(
                            f"gnn_hidden_size ({args.gnn_hidden_size}) must be divisible by gat_heads ({args.gat_heads}) when gat_concat=True"
                        )
                    out_channels = args.gnn_hidden_size // args.gat_heads
                else:
                    out_channels = args.gnn_hidden_size
                gnn_layer = GATConv(
                    in_channels=args.gnn_hidden_size,
                    out_channels=out_channels,
                    heads=args.gat_heads,
                    concat=args.gat_concat,
                    dropout=args.gat_dropout,
                    add_self_loops=False,
                )
            elif args.gnn_model == "SAGEConv":
                # GraphSAGE layer with configurable aggregator; default to 'mean' if not provided
                gnn_layer = SAGEConv(
                    in_channels=args.gnn_hidden_size,
                    out_channels=args.gnn_hidden_size,
                    aggr=getattr(args, 'sage_aggr', 'mean'),  # options: 'mean', 'max', 'add', 'lstm'
                )
            else:
                raise ValueError(f"Unsupported gnn_model: {args.gnn_model}")
            self.gnn_layers.append(gnn_layer)
        self.relu = ReLU()
        self.dropout = Dropout(args.dropout_rate)
        self.relu_layers_index = range(args.num_gnn_layers)
        self.dropout_layers_index = range(args.num_gnn_layers)

        if args.graph_pooling == "sum":
            self.pool = GlobalAddPool()
        elif args.graph_pooling == "mean":
            self.pool = GlobalMeanPool()
        elif args.graph_pooling == "max":
            self.pool = GlobalMaxPool()
        else:
            raise ValueError("Invalid graph pooling type.")

        self.classifier = nn.Sequential(
            nn.Linear(args.gnn_hidden_size, args.gnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.gnn_hidden_size, args.num_classes),
            nn.Softmax(dim=-1)
        )

    # def forward(self, x, edge_index, batch=None, **kwargs):
    #     output = self.get_emb(x, edge_index, **kwargs)
    #
    #     pooled_output = self.pool(output, batch)
    #     logits = self.classifier(pooled_output)
    #
    #     return logits

    def forward(self, x, edge_index, batch, edge_weight=None):
        # 处理输入特征
        x = self.linear(x)

        # 遍历 GNN 层
        for gnn_layer in self.gnn_layers:
            # 针对不同层的 edge_weight 兼容处理
            if isinstance(gnn_layer, GATConv):
                # GATConv 默认不使用 edge_weight
                x = gnn_layer(x, edge_index)
            else:
                # 尝试传入 edge_weight；若该层/版本不支持，则回退为不传
                if edge_weight is not None:
                    try:
                        x = gnn_layer(x, edge_index, edge_weight=edge_weight)
                    except TypeError:
                        x = gnn_layer(x, edge_index)
                else:
                    x = gnn_layer(x, edge_index)

            x = F.relu(x)
            x = self.dropout(x)

        # 图池化
        x = self.pool(x, batch)

        # 分类器
        x = self.classifier(x)
        return x

    def get_emb(self, x, edge_index, **kwargs):
        output = self.linear(x)
        for gnn_layer_index in range(len(self.gnn_layers)):
            if self.args.residual:
                output = output + self.gnn_layers[gnn_layer_index](output, edge_index)
            else:
                output = self.gnn_layers[gnn_layer_index](output, edge_index)
            if gnn_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if gnn_layer_index in self.dropout_layers_index:
                output = self.dropout(output)

        return output
