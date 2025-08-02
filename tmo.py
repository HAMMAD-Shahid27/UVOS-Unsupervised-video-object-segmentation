import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from transformers import SegformerModel

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
    
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        
        logits = torch.matmul(z_i, z_j.T) / self.temperature
        batch_size = z_i.size(0)

        
        labels = torch.arange(batch_size).cuda()

        
        loss = F.cross_entropy(logits, labels)
        return loss


class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(ConvRelu(c, c, 1, 1, 0), Conv(c, c, 1, 1, 0))
        self.conv3 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        c = torch.sigmoid(self.conv2(F.adaptive_avg_pool2d(x, output_size=(1, 1))) +
                          self.conv2(F.adaptive_max_pool2d(x, output_size=(1, 1))))
        x = x * c
        s = torch.sigmoid(self.conv3(torch.cat([torch.mean(x, dim=1, keepdim=True),
                                                torch.max(x, dim=1, keepdim=True)[0]], dim=1)))
        x = x * s
        return x


class Encoder(nn.Module):
    def __init__(self, ver):
        super().__init__()
        self.ver = ver

        if ver == 'rn101':
            backbone = tv.models.resnet101(pretrained=True)
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4

            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            self.embedding_layer_s32 = nn.Linear(2048, 256)  
            self.embedding_layer_s16 = nn.Linear(1024, 256) 

        if ver == 'mitb2':
            self.backbone = SegformerModel.from_pretrained('nvidia/mit-b2')
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            self.embedding_layer_s32 = nn.Linear(512, 256)
            self.embedding_layer_s16 = nn.Linear(320, 256)

    def forward(self, img):
        x = (img - self.mean) / self.std
        if self.ver == 'rn101':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            s4 = x
            x = self.layer2(x)
            s8 = x
            x = self.layer3(x)
            s16 = x
            x = self.layer4(x)
            s32 = x

            embedding_s32 = self.embedding_layer_s32(torch.flatten(F.adaptive_avg_pool2d(s32, (1, 1)), 1))
            embedding_s16 = self.embedding_layer_s16(torch.flatten(F.adaptive_avg_pool2d(s16, (1, 1)), 1))

        elif self.ver == 'mitb2':
            x = self.backbone(x, output_hidden_states=True).hidden_states
            s4, s8, s16, s32 = x[0], x[1], x[2], x[3]

            embedding_s32 = self.embedding_layer_s32(torch.flatten(F.adaptive_avg_pool2d(s32, (1, 1)), 1))
            embedding_s16 = self.embedding_layer_s16(torch.flatten(F.adaptive_avg_pool2d(s16, (1, 1)), 1))
        return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}, embedding_s16, embedding_s32


class Decoder(nn.Module):
    def __init__(self, ver):
        super().__init__()
        if ver == 'rn101':
            self.conv1 = ConvRelu(2048, 256, 1, 1, 0)
            self.blend1 = ConvRelu(256, 256, 3, 1, 1)
            self.cbam1 = CBAM(256)
            self.conv2 = ConvRelu(1024, 256, 1, 1, 0)
            self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam2 = CBAM(256)
            self.conv3 = ConvRelu(512, 256, 1, 1, 0)
            self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam3 = CBAM(256)
            self.conv4 = ConvRelu(256, 256, 1, 1, 0)
            self.blend4 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam4 = CBAM(256)
            self.predictor = Conv(256, 2, 3, 1, 1)

        if ver == 'mitb2':
            self.conv1 = ConvRelu(512, 256, 1, 1, 0)
            self.blend1 = ConvRelu(256, 256, 3, 1, 1)
            self.cbam1 = CBAM(256)
            self.conv2 = ConvRelu(320, 256, 1, 1, 0)
            self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam2 = CBAM(256)
            self.conv3 = ConvRelu(128, 256, 1, 1, 0)
            self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam3 = CBAM(256)
            self.conv4 = ConvRelu(64, 256, 1, 1, 0)
            self.blend4 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam4 = CBAM(256)
            self.predictor = Conv(256, 2, 3, 1, 1)

    def forward(self, app_feats, mo_feats):
        x = self.conv1(app_feats['s32'] + mo_feats['s32'])
        x = self.cbam1(self.blend1(x))
        s16 = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = torch.cat([self.conv2(app_feats['s16'] + mo_feats['s16']), s16], dim=1)
        x = self.cbam2(self.blend2(x))
        s8 = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = torch.cat([self.conv3(app_feats['s8'] + mo_feats['s8']), s8], dim=1)
        x = self.cbam3(self.blend3(x))
        s4 = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = torch.cat([self.conv4(app_feats['s4'] + mo_feats['s4']), s4], dim=1)
        x = self.predictor(self.cbam4(self.blend4(x)))
        score = F.interpolate(x, scale_factor=4, mode='bicubic')
        return score


class VOS(nn.Module):
    def __init__(self, ver):
        super().__init__()
        self.app_encoder = Encoder(ver)
        self.mo_encoder = Encoder(ver)
        self.decoder = Decoder(ver)


class MetaWeightingNet(nn.Module):
    def __init__(self):
        super(MetaWeightingNet, self).__init__()
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def forward(self, losses):
        w = F.softmax(self.weights, dim=0)
        return (losses * w).sum(), w


class TMO(nn.Module):
    def __init__(self, ver, init_temp=0.07):
        super().__init__()
        self.vos = VOS(ver)
        self.contrastive_loss_s32 = ContrastiveLoss(init_temp)
        self.contrastive_loss_s16 = ContrastiveLoss(init_temp)
        self.meta_weighter = MetaWeightingNet()

    @property
    def temperature_s32(self):
        return self.contrastive_loss_s32.temperature

    @property
    def temperature_s16(self):
        return self.contrastive_loss_s16.temperature

    def forward(self, imgs, flows):
        B, L, _, H1, W1 = imgs.size()
        _, _, _, H2, W2 = flows.size()

        s = 512
        imgs = F.interpolate(imgs.view(B * L, -1, H1, W1), size=(s, s), mode='bicubic').view(B, L, -1, s, s)
        flows = F.interpolate(flows.view(B * L, -1, H2, W2), size=(s, s), mode='bicubic').view(B, L, -1, s, s)

        score_lst = []
        mask_lst = []
        contrastive_loss_lst_s32 = []
        contrastive_loss_lst_s16 = []

        for i in range(L):
            app_feats, app_embedding_s16, app_embedding_s32 = self.vos.app_encoder(imgs[:, i])
            mo_feats, mo_embedding_s16, mo_embedding_s32 = self.vos.mo_encoder(flows[:, i])
            score = self.vos.decoder(app_feats, mo_feats)
            score = F.interpolate(score, size=(H1, W1), mode='bicubic')
            score_lst.append(score)

            contrastive_loss_s32 = self.contrastive_loss_s32(app_embedding_s32, mo_embedding_s32)
            contrastive_loss_s16 = self.contrastive_loss_s16(app_embedding_s16, mo_embedding_s16)

            contrastive_loss_lst_s32.append(contrastive_loss_s32)
            contrastive_loss_lst_s16.append(contrastive_loss_s16)

            pred_seg = torch.softmax(score, dim=1)
            pred_mask = torch.max(pred_seg, dim=1, keepdim=True)[1]
            mask_lst.append(pred_mask)

        total_contrastive_loss_s32 = torch.mean(torch.stack(contrastive_loss_lst_s32))
        total_contrastive_loss_s16 = torch.mean(torch.stack(contrastive_loss_lst_s16))

        losses = torch.stack([total_contrastive_loss_s16, total_contrastive_loss_s32])
        total_contrastive_loss, weights = self.meta_weighter(losses)

        return {
            'scores': torch.stack(score_lst, dim=1),
            'masks': torch.stack(mask_lst, dim=1),
            'contrastive_loss': total_contrastive_loss,
            'meta_weights': weights.detach(),
            'contrastive_loss_s32': total_contrastive_loss_s32,
            'contrastive_loss_s16': total_contrastive_loss_s16
        }