# Modified based on https://github.com/facebookresearch/simsiam

import copy

import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        # output embedding of avgpool
        self.encoder.avgpool.register_forward_hook(self._get_avg_output())
        self.embedding = None

    def _get_avg_output(self):
        def hook(model, input, output):
            self.embedding = output.detach()
        return hook

    def forward(self, x1, x2=None):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        if self.training:
            # compute features for one view
            z1 = self.encoder(x1) # NxC
            z2 = self.encoder(x2) # NxC

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC

            return p1, p2, z1.detach(), z2.detach()
        else:
            _ = self.encoder(x1)
            return self.embedding.squeeze()


class SimSiamGSG(nn.Module):
    """
    Build a SimSiamGSG model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiamGSG, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        # output embedding of avgpool
        self.encoder.avgpool.register_forward_hook(self._get_avg_output())
        self.embedding = None

    def _get_avg_output(self):
        def hook(model, input, output):
            self.embedding = output.detach()
        return hook

    def forward(self, x11, x12=None, x21=None, x22=None):
        """
        Input:
            x11: first views of first images
            x12: second views of first images
            x21: first views of second images
            x22: second views of second images
        Output:
            p11, p12, z11, z12, p21, p22, z21, z22: predictors and targets of the network
        """
        if self.training:
            # compute features for one view
            z11 = self.encoder(x11) # NxC
            z12 = self.encoder(x12) # NxC

            p11 = self.predictor(z11) # NxC
            p12 = self.predictor(z12) # NxC

            z21 = self.encoder(x21) # NxC
            z22 = self.encoder(x22) # NxC

            p21 = self.predictor(z21) # NxC
            p22 = self.predictor(z22) # NxC

            return p11, p12, z11.detach(), z12.detach(), p21, p22, z21.detach(), z22.detach()
        else:
            _ = self.encoder(x11)
            return self.embedding.squeeze()


class BYOL(nn.Module):
    """
    Build a BYOL model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(BYOL, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.encoder_momentum = copy.deepcopy(self.encoder)
        for param in self.encoder_momentum.parameters():
            param.requires_grad = False

        # output embedding of avgpool
        self.encoder.avgpool.register_forward_hook(self._get_avg_output())
        self.embedding = None

    def _get_avg_output(self):
        def hook(model, input, output):
            self.embedding = output.detach()
        return hook

    def forward(self, x1, x2=None):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        if self.training:
            # compute features for one view
            z1 = self.encoder(x1) # NxC
            z2 = self.encoder(x2) # NxC

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC

            for model_ema, model in zip(self.encoder_momentum.parameters(), self.encoder.parameters()):
                model_ema.data = model_ema.data * 0.99 + model.data * (1. - 0.99)
            z1 = self.encoder_momentum(x1) # NxC
            z2 = self.encoder_momentum(x2) # NxC

            return p1, p2, z1.detach(), z2.detach()
        else:
            _ = self.encoder(x1)
            return self.embedding.squeeze()


class BYOLGSG(nn.Module):
    """
    Build a BYOLGSG model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(BYOLGSG, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.encoder_momentum = copy.deepcopy(self.encoder)
        for param in self.encoder_momentum.parameters():
            param.requires_grad = False

        # output embedding of avgpool
        self.encoder.avgpool.register_forward_hook(self._get_avg_output())
        self.embedding = None

    def _get_avg_output(self):
        def hook(model, input, output):
            self.embedding = output.detach()
        return hook

    def forward(self, x11, x12=None, x21=None, x22=None):
        """
        Input:
            x11: first views of first images
            x12: second views of first images
            x21: first views of second images
            x22: second views of second images
        Output:
            p11, p12, z11, z12, p21, p22, z21, z22: predictors and targets of the network
        """
        if self.training:
            for model_ema, model in zip(self.encoder_momentum.parameters(), self.encoder.parameters()):
                model_ema.data = model_ema.data * 0.99 + model.data * (1. - 0.99)

            # compute features for one view
            z11 = self.encoder(x11) # NxC
            z12 = self.encoder(x12) # NxC

            p11 = self.predictor(z11) # NxC
            p12 = self.predictor(z12) # NxC

            z11 = self.encoder_momentum(x11) # NxC
            z12 = self.encoder_momentum(x12) # NxC

            z21 = self.encoder(x21) # NxC
            z22 = self.encoder(x22) # NxC

            p21 = self.predictor(z21) # NxC
            p22 = self.predictor(z22) # NxC

            z21 = self.encoder_momentum(x21) # NxC
            z22 = self.encoder_momentum(x22) # NxC

            return p11, p12, z11.detach(), z12.detach(), p21, p22, z21.detach(), z22.detach()
        else:
            _ = self.encoder(x11)
            return self.embedding.squeeze()
