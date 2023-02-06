import torch
from torch_geometric.nn import DenseGCNConv

class GCNEncoder(torch.nn.Module):
    def __init__(self, *,
                 num_vertex, edges, hidden_dim=128, out_dim):
        super().__init__()
        self.num_vertex = num_vertex
        self.register_buffer('edges', edges.detach().clone()) # Ex2
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.conv1 = DenseGCNConv(1, hidden_dim)
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        self.conv3 = DenseGCNConv(hidden_dim, hidden_dim)
        self.conv4 = DenseGCNConv(hidden_dim, out_dim // 2)

        self.activation = torch.nn.ELU()

        self.latent_ffn = torch.nn.Sequential(
            torch.nn.Linear(out_dim // 2, hidden_dim),
            self.activation,
            torch.nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            torch.nn.Linear(hidden_dim, out_dim // 2),
        )

    def forward(self, edge_weights):
        # edge_weights: BxE
        B, E = edge_weights.shape
        adj = torch.zeros([self.num_vertex, self.num_vertex],
                          dtype=edge_weights.dtype,
                          device=edge_weights.device) # VxV
        adj = adj.unsqueeze(0).expand(B, -1, -1).clone() # BxVxV

        index_0 = torch.arange(B).unsqueeze(-1).expand(-1, E).reshape(-1) # BE
        index_1 = self.edges[:, 0].unsqueeze(0).expand(B, -1).reshape(-1) # BE
        index_2 = self.edges[:, 1].unsqueeze(0).expand(B, -1).reshape(-1) # BE
        adj[[index_0, index_1, index_2]] = edge_weights.reshape(-1)
        adj[[index_0, index_2, index_1]] = edge_weights.reshape(-1)

        x = adj.mean(-1).unsqueeze(-1) # BxVx1 TODO
        x = self.activation(self.conv1(x, adj))
        x = self.activation(self.conv2(x, adj))
        x = self.activation(self.conv3(x, adj))
        x = self.activation(self.conv4(x, adj)) # BxVxO/2

        y = self.latent_ffn(x) # BxVxO/2

        global_feature = y.max(-2)[0] # BxO/2
        global_feature = global_feature.unsqueeze(-2).expand(
            B, self.num_vertex, -1) # BxVxO/2

        vertex_feature = torch.cat([global_feature, x], -1) # BxVxO

        return global_feature, vertex_feature


class SpectralEncoder(torch.nn.Module):
    def __init__(self, *,
                 num_vertex, edges, hidden_dim=128, out_dim,
                 kind='adj'):
        super().__init__()
        self.num_vertex = num_vertex
        self.register_buffer('edges', edges.detach().clone()) # Ex2
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.activation = torch.nn.ReLU()
        self.kind = kind

        if kind == 'adj' or kind == 'spectral':
            self.latent_ffn = torch.nn.Sequential(
                torch.nn.Linear(num_vertex*num_vertex if use_adj
                                else 2*num_vertex*num_vertex, hidden_dim),
                self.activation,
                torch.nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                torch.nn.Linear(hidden_dim, out_dim),
            )
        else:
            assert(kind == 'spectral_shared')
            self.latent_ffn = torch.nn.Sequential(
                torch.nn.Linear(2*num_vertex, 4*num_vertex),
                self.activation,
                torch.nn.Linear(4*num_vertex, 8*num_vertex),
                self.activation,
                torch.nn.Linear(8*num_vertex, out_dim // num_vertex),
            )

    def forward(self, edge_weights):
        # edge_weights: BxE
        B, E = edge_weights.shape
        V = self.num_vertex
        adj = torch.zeros([V, V],
                          dtype=edge_weights.dtype,
                          device=edge_weights.device) # VxV
        adj = adj.unsqueeze(0).expand(B, -1, -1).clone() # BxVxV

        index_0 = torch.arange(B).unsqueeze(-1).expand(-1, E).reshape(-1) # BE
        index_1 = self.edges[:, 0].unsqueeze(0).expand(B, -1).reshape(-1) # BE
        index_2 = self.edges[:, 1].unsqueeze(0).expand(B, -1).reshape(-1) # BE
        adj[[index_0, index_1, index_2]] = edge_weights.reshape(-1)
        adj[[index_0, index_2, index_1]] = edge_weights.reshape(-1) # BxVxV

        if self.kind != 'adj':
            eig_vals, eig_vecs = torch.linalg.eigh(adj)
            # eig_vals: BxV
            # eig_vecs: BxVxV
            # BxVxV, now eig_vecs[:, :, k] is the kth eigenvector
            eig_vecs = eig_vecs / (eig_vecs.norm(dim=-2, keepdim=True) + 1e-8) # BxVxV
            x = torch.cat([eig_vecs,
                           eig_vals.unsqueeze(-2).expand(-1, V, -1)], -1) # BxVx2V
            if self.kind == 'spectral':
                x = x.reshape(B, -1) # Bx2V^2
                y = self.latent_ffn(x) # BxO
            else:
                assert(self.kind == 'spectral_shared')
                y = self.latent_ffn(x) # BxVxO/V
                y = y.reshape(B, -1)
        else:
            x = adj.reshape(B, -1) # BxV^2
            y = self.latent_ffn(x) # BxO
        return y
