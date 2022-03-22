#

from torch import nn
from torch.nn import functional as F


"""
    SimpleSegmentationModel
    A simple encoder-decoder based segmentation model. 
"""

class SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, decoder):
        super(SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

class BackBoneModel(nn.Module):
    def __init__(self, backbone):
        super(BackBoneModel, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        # token = 0
        return x, 1


class ContrastivePredictionModel(nn.Module):
    def __init__(self, backbone, decoder, head='linear', upsample=True, use_classification_head=False, freeze_batchnorm='none'):
        super(ContrastivePredictionModel, self).__init__()
        self.backbone = backbone
        self.upsample = upsample
        self.use_classification_head = use_classification_head
        
        if head == 'linear': 
            # Head is linear.
            # We can just use regular decoder since final conv is 1 x 1.
            self.head = decoder[-1]
            decoder[-1] = nn.Identity()
            self.decoder = decoder

        else:
            raise NotImplementedError('Head {} is currently not supported'.format(head))


    def forward(self, x):
        # Standard model
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        embedding = self.decoder(x)  ## this embedding is used for contrastive learning
        # Head
        x = self.head(embedding) ## self.head is the classification head, output dim = num_classes
        # Upsample to input resolution
        if self.upsample: 
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        
        return x, embedding


