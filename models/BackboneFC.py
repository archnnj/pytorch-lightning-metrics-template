import pytorch_lightning as pl
from models.ResnetBackbone import ResnetBackbone
from models.FC import FC

class BackboneFC(pl.LightningModule):
    def __init__(self, in_ch, fc_hidden_dims, out_dim, fc_activation='relu', backbone='resnet18', finetuning=False):
        super().__init__()
        self.save_hyperparameters('in_ch', 'fc_hidden_dims', 'out_dim', 'fc_activation', 'backbone', 'finetuning')
        self._init_backbone(backbone)

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        return self.classifier(representations)

    def freeze_backbone(self):
        self.feature_extractor.freeze()

    def unfreeze_backbone(self):
        self.feature_extractor.unfreeze()

    def _init_backbone(self, backbone_type):
        # init a pretrained resnet
        if backbone_type.startswith('resnet'):
            self.feature_extractor = ResnetBackbone(in_ch=1, resnet_type=backbone_type, pretrained=True)
        else:
            raise ValueError(f"{self.__class__.__name__}: backbone '{backbone_type}' not available.")
        if not self.hparams.finetuning:
            self.freeze_backbone()
        # classifier head
        self.classifier = FC(in_dim=self.feature_extractor.num_features_out,
                             hidden_dims=self.hparams.fc_hidden_dims,
                             out_dim=self.hparams.out_dim,
                             activation=self.hparams.fc_activation)
