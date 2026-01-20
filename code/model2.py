import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, NNConv, GINConv, GATConv, Set2Set

class GNNm_DNABERT(nn.Module):
    def __init__(self, n_features, hidden_dim, n_classes, n_conv_layers=3, dropout=0.1,
                 conv_type1="GIN", conv_type2="GIN", batch_norm=True, batch_size=128,
                 dnabert_path="DNABERT-2-117M", freeze_bert=True):
        super(GNNm_DNABERT, self).__init__()

        self.batch_size = batch_size
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.batch_norms1 = nn.ModuleList()
        self.batch_norms2 = nn.ModuleList()

        '''GNN1 - 暂时注释不用'''
        # self.convs1.append(self.get_conv_layer(n_features, hidden_dim, conv_type=conv_type1))
        # self.batch_norms1.append(nn.BatchNorm1d(hidden_dim))

        '''GNN2'''
        self.convs2.append(self.get_conv_layer(n_features, hidden_dim, conv_type=conv_type2))
        self.batch_norms2.append(nn.BatchNorm1d(hidden_dim))

        # 后续层: hidden_dim → hidden_dim
        # for i in range(1, n_conv_layers):
        #     self.convs2.append(self.get_conv_layer(hidden_dim, hidden_dim, conv_type=conv_type2))
        #     self.batch_norms2.append(nn.BatchNorm1d(hidden_dim))

        '''MLP - 暂时注释不用'''
        # self.mlp_cksnap = nn.Sequential(nn.Linear(96, 80, bias=True), 
        #                                 nn.Linear(80, 50, bias=True),
        #                                 nn.Linear(50, 128, bias=True) 
        #                                 )
        # self.mlp_kmer = nn.Sequential(nn.Linear(1364, 800, bias=True), 
        #                                 nn.Linear(800, 300, bias=True),
        #                                 nn.Linear(300, 128, bias=True) 
        #                                 )
        
        # self.cross_attention1 = nn.MultiheadAttention(hidden_dim*2, 4)
        # self.cross_attention2 = nn.MultiheadAttention(hidden_dim*2, 4)

        '''DNABERT-2分支'''
        from dnabert2_encoder import DNABERT2Encoder
        self.dnabert = DNABERT2Encoder(model_path=dnabert_path, 
                                       freeze_bert=freeze_bert, 
                                       pooling='mean')
        
        # GNN2升维到768
        self.fc_gnn2 = nn.Linear(hidden_dim*2, 768)  # 128 → 768
        
        # 交叉注意力融合（768维，8个头）
        self.cross_attention_fusion = nn.MultiheadAttention(768, num_heads=8)

        # 分类层
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, n_classes)

        # self.pooling1 = Set2Set(hidden_dim, processing_steps=10)  # 暂时不用
        self.pooling2 = Set2Set(hidden_dim, processing_steps=10)

        self.dropout = nn.Dropout(dropout)
        self.conv_type1 = conv_type1
        self.conv_type2 = conv_type2
        self.batch_norm = batch_norm


    def forward(self, data, data1, sequences): 

        # g1, adj1, edge_attr1, batch1 = data.x, data.edge_index, data.edge_attr, data.batch
        g2, adj2, edge_attr2, batch2 = data1.x, data1.edge_index, data1.edge_attr, data1.batch
        # x_cksnap = data.cksnap  # 暂时不用
        # x_kmer = data.kmer  # 暂时不用
        
        # GNN 1 - 暂时注释
        # for i, con in enumerate(self.convs1):
        #     g1 = self.apply_conv_layer(con, g1, adj1, edge_attr1, conv_type=self.conv_type1)
        #     g1 = self.dropout(g1)
        # g1 = self.pooling1(g1, batch1)
        # g1 = self.dropout(g1)

        # GNN 2
        for i, con in enumerate(self.convs2):
            g2 = self.apply_conv_layer(con, g2, adj2, edge_attr2, conv_type=self.conv_type2)
            g2 = self.dropout(g2)

        g2 = self.pooling2(g2, batch2)
        g2 = self.dropout(g2)
        # GNN 2 - 使用多层 + BatchNorm
        # for i, conv in enumerate(self.convs2):
        #     g2 = self.apply_conv_layer(conv, g2, adj2, edge_attr2, conv_type=self.conv_type2)
        #     if self.batch_norm:
        #         g2 = self.batch_norms2[i](g2)  # 添加 BatchNorm
        #     g2 = torch.relu(g2)  # 添加激活函数
        #     g2 = self.dropout(g2)

        # g2 = self.pooling2(g2, batch2)  # [num_nodes, 128] → [batch_size, 256]
        # g2 = self.dropout(g2)
        # GNN2升维到768
        g2 = self.fc_gnn2(g2)  # [batch, 128] → [batch, 768]

        # DNABERT-2分支
        dna_emb = self.dnabert(sequences)  # [batch, 768]

        # 准备交叉注意力输入
        g2_attn = g2.unsqueeze(dim=1).permute(1, 0, 2)  # [1, batch, 768]
        dna_attn = dna_emb.unsqueeze(dim=1).permute(1, 0, 2)  # [1, batch, 768]

        # 交叉注意力融合（GNN2作为query，DNABERT作为key和value）
        fused, _ = self.cross_attention_fusion(g2_attn, dna_attn, dna_attn)
        fused = fused.permute(1, 0, 2).squeeze(dim=1)  # [batch, 768]

        # CKSNAP分支 - 暂时注释
        # ss = self.batch_size
        # x1 = x_cksnap.reshape((ss, -1))
        # x1 = self.mlp_cksnap(x1) 
        # x1 = x1.unsqueeze(dim=1) 
        # x1 = x1.permute(1, 0, 2)
        # x1, _ = self.cross_attention1(x1, x1, x1) 
        # x1 = x1.permute(1, 0, 2)
        # x1 = x1.squeeze(dim=1)

        # K-mer分支 - 暂时注释
        # x2 = x_kmer.reshape((ss, -1))
        # x2 = self.mlp_kmer(x2)
        # x2 = x2.unsqueeze(dim=1)
        # x2 = x2.permute(1, 0, 2)
        # x2, _ = self.cross_attention2(x2, x2, x2)
        # x2 = x2.permute(1, 0, 2)
        # x2 = x2.squeeze(dim=1)

        # x = torch.cat((g1, g2, x1, x2), dim=1)  # 原来的4路融合
        x = fused  # 现在只用融合后的特征
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # output = torch.sigmoid(x)
        # return output
        return x


    @staticmethod
    def get_conv_layer(n_input_features, n_output_features, conv_type="GCN"):
        if conv_type == "GCN":
            return GCNConv(n_input_features, n_output_features)
        elif conv_type == "GAT":
            return GATConv(n_input_features, n_output_features)
        elif conv_type == "MPNN":
            net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, n_input_features *
                                                                      n_output_features))
            return NNConv(n_input_features, n_output_features, net)
        elif conv_type == "GIN":
            net = nn.Sequential(nn.Linear(n_input_features, n_output_features), nn.ReLU(),
                                nn.Linear(n_output_features, n_output_features))
            return GINConv(net)
        else:
            raise Exception("{} convolutional layer is not supported.".format(conv_type))

    @staticmethod
    def apply_conv_layer(conv, x, adj, edge_attr, conv_type="GCN"):
        if conv_type in ["GCN", "GAT", "GIN"]:
            return conv(x, adj)
        elif conv_type in ["MPNN"]:
            return conv(x, adj, edge_attr)
        else:
            raise Exception("{} convolutional layer is not supported.".format(conv_type))