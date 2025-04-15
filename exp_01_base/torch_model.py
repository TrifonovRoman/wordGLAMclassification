import torch
from torch.nn import Linear, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, GELU
from torch.nn.functional import relu
from torch_geometric.nn import BatchNorm, TAGConv
from typing import List

class NodeGLAM(torch.nn.Module):
    def __init__(self, input_, h, output_):
        super(NodeGLAM, self).__init__()
        self.activation = GELU()
        self.batch_norm1 = BatchNorm(input_)
        self.linear1 = Linear(input_, h[0])
        self.tag1 = TAGConv(h[0], h[1])
        self.linear2 = Linear(h[1], h[2])
        self.tag2 = TAGConv(h[2], h[3])

        self.linear3 = Linear(h[3] + input_, h[4])
        self.linear4 = Linear(h[4], output_)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [num_nodes, input_dim]
        x_norm = self.batch_norm1(x)
        h = self.linear1(x_norm)
        h = self.activation(h)
        h = self.tag1(h, edge_index)
        h = self.activation(h)
        h = self.linear2(h)
        h = self.activation(h)
        h = self.tag2(h, edge_index)
        h = self.activation(h)
        a = torch.cat([x, h], dim=1)
        a = self.linear3(a)
        a = self.activation(a)
        node_emb = self.linear4(a)
        return node_emb

class EdgeGLAM(torch.nn.Module):
    def __init__(self, input_, h, output_):
        super(EdgeGLAM, self).__init__()
        self.activation = GELU()
        self.batch_norm2 = BatchNorm(input_, output_)
        self.linear1 = Linear(input_, h[0])
        self.linear2 = Linear(h[0], output_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm2(x)
        h = self.linear1(x)
        h = self.activation(h)
        edge_emb = self.linear2(h)
        return edge_emb


class EdgeCategoryClass(torch.nn.Module):
    def __init__(self, input_, h, output_):
        super(EdgeCategoryClass, self).__init__()
        self.activation = GELU()
        self.linear1 = Linear(input_, h[0])
        self.linear2 = Linear(h[0], h[1])
        self.classifer = Linear(h[1], output_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        h = self.activation(h)
        h = self.linear2(h)
        cl = self.classifer(self.activation(h))
        return cl

class EdgeBinaryClass(torch.nn.Module):
    def __init__(self, input_, h, output_):
        super(EdgeBinaryClass, self).__init__()
        self.activation = GELU()
        self.linear1 = Linear(input_, h[0])
        self.linear2 = Linear(h[0], output_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        h = self.activation(h)
        h = self.linear2(h)
        return torch.squeeze(h, 1)

class TorchModel(torch.nn.Module):
    def __init__(self,
                 node_input_dim, node_hidden_dims, node_emb_dim,
                 edge_raw_dim, edge_hidden_dims, edge_emb_dim,
                 cat_hidden_dims, num_edge_classes,
                 bin_hidden_dims):
        '''
        node_input_dim: размер фич узлов
        node_hidden_dims: список скрытых размерностей для NodeGLAM
        node_emb_dim: размер узлового эмбеддинга
        edge_input_dim: размер признаков ребра (например, угол, длина и т.д.)
        edge_hidden_dims: список скрытых размерностей для EdgeGLAM
        edge_emb_dim: размер эмбеддинга ребра
        cat_hidden_dims: список скрытых размерностей для многоклассового классификатора ребер
        num_edge_classes: число классов ребра
        bin_hidden_dims: список скрытых размерностей для бинарного классификатора ребер
        '''
        super(TorchModel, self).__init__()
        self.node_model = NodeGLAM(node_input_dim, node_hidden_dims, node_emb_dim)

        edge_glam_input_dim = 2 * node_emb_dim + edge_raw_dim
        self.edge_model = EdgeGLAM(edge_glam_input_dim, edge_hidden_dims, edge_emb_dim)

        multi_cat_input_dim = 2 * node_emb_dim + edge_emb_dim + edge_raw_dim
        self.edge_cat_model = EdgeCategoryClass(multi_cat_input_dim, cat_hidden_dims, num_edge_classes)

        self.edge_bin_model = EdgeBinaryClass(edge_emb_dim, bin_hidden_dims, 1)

    def forward(self, node_x, edge_raw, edge_index):

        node_emb = self.node_model(node_x, edge_index)

        src, dst = edge_index[0], edge_index[1]
        node_emb_src = node_emb[src]
        node_emb_dst = node_emb[dst]

        edge_glam_input = torch.cat([node_emb_src, node_emb_dst, edge_raw], dim=1)
        edge_emb = self.edge_model(edge_glam_input)

        multi_cat_input = torch.cat([node_emb_src, node_emb_dst, edge_emb, edge_raw], dim=1)
        edge_multi_class = self.edge_cat_model(multi_cat_input)

        edge_bin_input = edge_emb
        edge_bin_class = self.edge_bin_model(edge_bin_input)

        return {
            "node_emb": node_emb,
            "edge_emb": edge_emb,
            "edge_multi_class": edge_multi_class,
            "edge_bin_class": edge_bin_class
        }

class CustomLoss(torch.nn.Module):
    def __init__(self, params):
        super(CustomLoss, self).__init__()
                    #BCELoss
        self.bce = BCEWithLogitsLoss(pos_weight=torch.tensor(params['edge_imbalance']))
        self.ce = CrossEntropyLoss(weight=torch.tensor(params['publaynet_imbalance']))
        self.edge_coef:float = params['edge_coef']

    def forward(self, n_pred, n_true, e_pred, e_true):
        loss = self.ce(n_pred, n_true) + self.edge_coef*self.bce(e_pred, e_true)
        return loss