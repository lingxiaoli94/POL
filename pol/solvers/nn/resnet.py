import torch
import torchvision

class ResNetEncoder(torch.nn.Module):
    def __init__(self, *,
                 hidden_dim=256,
                 pretrained=True,
                 pe_kind='sine', # ['sine', 'rand', 'none']
                 learn_pe=False,
                 max_compressed_size=100,
                 freeze_bn=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pe_kind = pe_kind
        self.learn_pe = learn_pe
        self.max_compressed_size = max_compressed_size

        self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        del self.backbone.fc

        self.should_freeze_bn = freeze_bn
        if self.should_freeze_bn:
            self.freeze_bn_layers()

        self.conv = torch.nn.Conv2d(2048, hidden_dim, 1)
        self.create_positional_encoding()

        self.latent_ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim+self.pe_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, hidden_dim))
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))

    def create_positional_encoding(self):
        if self.pe_kind == 'none':
            self.pe_dim = 0
            return

        if self.pe_kind == 'rand':
            row_embed = torch.rand(self.max_compressed_size,
                                   self.hidden_dim // 2) # MxD/2
            col_embed = torch.rand(self.max_compressed_size,
                                   self.hidden_dim // 2) # MxD/2
            self.pe_dim = self.hidden_dim
        else:
            assert(self.pe_kind == 'sine')
            tmp = torch.arange(self.max_compressed_size).float() # M
            freq = torch.arange(self.hidden_dim // 4).float() # D/4
            freq = tmp.unsqueeze(-1) / torch.pow(10000, 4 * freq.unsqueeze(0) / self.hidden_dim) # MxD/4
            row_embed = torch.cat([freq.sin(), freq.cos()], -1) # MxD/2
            col_embed = row_embed
            self.pe_dim = self.hidden_dim

        if self.learn_pe:
            self.row_embed = torch.nn.Parameter(row_embed.detach().clone())
            self.col_embed = torch.nn.Parameter(col_embed.detach().clone())
        else:
            self.register_buffer('row_embed', row_embed.detach().clone())
            self.register_buffer('col_embed', col_embed.detach().clone())


    def freeze_bn_layers(self):
        for module in self.backbone.modules():
            is_bn = (isinstance(module, torch.nn.BatchNorm1d) or
                     isinstance(module, torch.nn.BatchNorm2d) or
                     isinstance(module, torch.nn.BatchNorm3d))
            if is_bn:
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.should_freeze_bn:
            self.freeze_bn_layers()

    def forward(self, inputs):
        '''
        Args:
            inputs: BxCxHxW, batched images

        Returns:
            BxD, embedded latent codes, or BxKx4, boxes
        '''
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # x is Bx2048xH'xW'
        h = self.conv(x) # BxDxH'xW'
        B, _, H, W = h.shape

        h = h.permute(0, 2, 3, 1) # BxH'xW'xD
        if self.pe_kind != 'none':
            pos = torch.cat([
                self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)
            ], -1) # H'xW'xD
            pos = pos.unsqueeze(0).repeat(B, 1, 1, 1) # BxH'xW'xD
            h = torch.cat([h, pos], -1) # BxH'xW'x2D
        h = self.latent_ffn(h) # BxH'xW'xD
        h = h.permute(0, 3, 1, 2) # BxDxH'xW'

        h = self.pooling(h) # BxDx1x1
        return h.squeeze(-1).squeeze(-1)
