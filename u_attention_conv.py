import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

from torch.nn import Sequential, ReLU, BatchNorm1d

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax


    
class LinearBlock(torch.nn.Module):
    """A linear block with optional activation, consisting of Linear + BatchNorm + ReLU."""
    def __init__(self, in_node_num, out_node_num, activation=True):
        super().__init__()
        
        self.activation = activation
        
        self.linear = Linear(in_node_num, out_node_num, weight_initializer='kaiming_uniform')
        self.bn = BatchNorm1d(out_node_num)
        if self.activation:
            self.a = ReLU()
    
        
    def forward(self, x):
        """Forward pass through the linear block.
        
        Args:
            x (Tensor): Input tensor of shape (N, in_node_num).
        
        Returns:
            Tensor: Output tensor of shape (N, out_node_num).
        """
        x = self.linear(x)
        x = self.bn(x)
        if self.activation:
            x = self.a(x)
        
        return x


class UNet(torch.nn.Module):
    """A U-Net style network with an encoder and two decoder branches for generating key and value embeddings."""
    def __init__(self, in_node_num, out_node_num, latent_node_num):
        super().__init__()
        
        # self.encoder_layer1 = LinearBlock(in_node_num, int(latent_node_num//4))
        # self.encoder_layer2 = LinearBlock(int(latent_node_num//4), int(latent_node_num//2))
        # self.encoder_layer3 = LinearBlock(int(latent_node_num//2), latent_node_num)
        
        # self.decoder1_layer3 = LinearBlock(latent_node_num, int(in_node_num//2))
        # self.decoder1_layer2 = LinearBlock(int(in_node_num//2)+int(latent_node_num//2), int(in_node_num//4))
        # self.decoder1_layer1 = LinearBlock(int(in_node_num//4)+int(latent_node_num//4), in_node_num, activation=False)
        
        # self.decoder2_layer3 = LinearBlock(latent_node_num, int(out_node_num//2))
        # self.decoder2_layer2 = LinearBlock(int(out_node_num//2)+int(latent_node_num//2), int(out_node_num//4))
        # self.decoder2_layer1 = LinearBlock(int(out_node_num//4)+int(latent_node_num//4), out_node_num, activation=False)
        
        
        self.encoder_layer1 = LinearBlock(in_node_num, int(latent_node_num//2))
        self.encoder_layer2 = LinearBlock(int(latent_node_num//2), latent_node_num)
        
        self.decoder1_layer2 = LinearBlock(latent_node_num, int(in_node_num//2))
        self.decoder1_layer1 = LinearBlock(int(in_node_num//2)+int(latent_node_num//2), in_node_num, activation=False)
        
        self.decoder2_layer2 = LinearBlock(latent_node_num, int(out_node_num//2))
        self.decoder2_layer1 = LinearBlock(int(out_node_num//2)+int(latent_node_num//2), out_node_num, activation=False)
    
        
    def forward(self, x):
        """Forward pass through the U-Net.
        
        Args:
            x (Tensor): Input tensor of shape (N, in_node_num).
        
        Returns:
            Tuple[Tensor, Tensor]: Two output tensors for the key and value paths,
            each of shape (N, in_node_num) and (N, out_node_num) respectively.
        """
        # x1_e = self.encoder_layer1(x)
        # x2_e = self.encoder_layer2(x1_e)
        # x3_e = self.encoder_layer3(x2_e)
      
        # x3_d1 = self.decoder1_layer3(x3_e)
        # x3_d1 = torch.cat([x3_d1,x2_e], dim=-1)
        # x2_d1 = self.decoder1_layer2(x3_d1)        
        # x2_d1 = torch.cat([x2_d1,x1_e], dim=-1)
        # x1_d1 = self.decoder1_layer1(x2_d1)
        
        # x3_d2 = self.decoder2_layer3(x3_e)
        # x3_d2 = torch.cat([x3_d2,x2_e], dim=-1)
        # x2_d2 = self.decoder2_layer2(x3_d2)
        # x2_d2 = torch.cat([x2_d2,x1_e], dim=-1)
        # x1_d2 = self.decoder2_layer1(x2_d2)
        
        x1_e = self.encoder_layer1(x)
        x2_e = self.encoder_layer2(x1_e)
      
        x2_d1 = self.decoder1_layer2(x2_e)        
        x2_d1 = torch.cat([x2_d1,x1_e], dim=-1)
        x1_d1 = self.decoder1_layer1(x2_d1)
        
        x2_d2 = self.decoder2_layer2(x2_e)
        x2_d2 = torch.cat([x2_d2,x1_e], dim=-1)
        x1_d2 = self.decoder2_layer1(x2_d2)
        
        return x1_d1, x1_d2


class MyTransformerConv(MessagePassing):
    """Custom Transformer-style graph convolution layer with U-Net based attention mechanism."""

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        key_query_len: int = None,
        beta: bool = False,
        dropout: float = 0.,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        """Initialize the MyTransformerConv layer.
        
        Args:
            in_channels (int or Tuple[int, int]): Input feature dimension(s).
            out_channels (int): Output feature dimension.
            key_query_len (int, optional): Dimension of key/query for attention (default: out_channels).
            beta (bool): Whether to use a gated residual connection (default: False).
            dropout (float): Dropout probability for attention weights (default: 0.0).
            bias (bool): Whether to use bias in linear layers (default: True).
            root_weight (bool): Whether to add a residual connection from the source node (default: True).
        """
        kwargs.setdefault('aggr', 'add')
        super(MyTransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.dropout = dropout
        self._alpha = None
        
        if isinstance(key_query_len, int) & key_query_len > 0:
            self.key_query_len = key_query_len
        else:
            self.key_query_len = out_channels
            
        self.unet = UNet(in_node_num=2*in_channels, out_node_num=out_channels, latent_node_num=self.key_query_len)
        self.query_bn = BatchNorm1d(2*in_channels)
        
        # self.unet = UNet(in_node_num=in_channels, out_node_num=out_channels, latent_node_num=self.key_query_len)
        # self.query_bn = BatchNorm1d(in_channels)

        # self.key = Linear(2*in_channels, self.key_query_len, weight_initializer='kaiming_uniform')
        # self.query = Linear(2*in_channels, self.key_query_len, weight_initializer='kaiming_uniform')
        # self.value = Linear(2*in_channels, out_channels, weight_initializer='kaiming_uniform')
        
        # self.encoder_key = Sequential(LinearBlock(2*in_channels, int(self.key_query_len//2)),
        #                               LinearBlock(int(self.key_query_len//2), self.key_query_len),)
        # self.decoder_query = Sequential(LinearBlock(self.key_query_len, int(self.key_query_len//2)),
        #                                 LinearBlock(int(self.key_query_len//2), 2*in_channels, activation=False),)
        # self.decoder_value = Sequential(LinearBlock(self.key_query_len, int(self.key_query_len//2)),
        #                                 LinearBlock(int(self.key_query_len//2), out_channels, activation=False))
        
        self.lin_skip = Linear(in_channels, out_channels, bias=bias)
        if self.beta:
            self.lin_beta = Linear(3 * out_channels, 1, bias=False)
        else:
            self.lin_beta = self.register_parameter('lin_beta', None)


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                return_attention_weights=None):
        """Forward pass of the Transformer convolution.
        
        Args:
            x (Tensor or PairTensor): Node features. If Tensor, treated as (x, x).
            edge_index (Adj): Graph edge indices (2 x E).
            return_attention_weights (bool, optional): If True, also return attention weights.
        
        Returns:
            Tensor or Tuple[Tensor, (Tensor, Tensor)]: Output features, optionally with attention weights.
        """
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index=edge_index, x=x, size=None)

        alpha = self._alpha
        # self._alpha = None

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]):
        """Construct messages from source to target nodes with attention.
        
        Args:
            x_i (Tensor): Features of target nodes (incoming).
            x_j (Tensor): Features of source nodes (outgoing).
            index (Tensor): Target node indices for each edge.
            ptr (OptTensor): Pointer for sorted edge indices (for softmax).
            size_i (Optional[int]): Number of target nodes.
        
        Returns:
            Tensor: Attention-weighted value messages.
        """
        x = torch.cat([x_i, x_j - x_i], dim=-1)
        query = self.query_bn(x)
        key, value = self.unet(x)
        
        # query = self.query_bn(x_i)
        # key, value = self.unet(x_j)
        
        # query = self.query(x)
        # key = self.key(x)
        # value = self.value(x)
        
        # latent_z = self.key(x)
        # key = self.decoder_query(latent_z)
        # value = self.decoder_value(latent_z)

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.key_query_len)
        self._alpha_logits = alpha.clone()
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value

        out *= alpha.view(-1, 1)
        return out

    def __repr__(self):
        """String representation of the layer."""
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
