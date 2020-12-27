import torch
import torch.nn as nn


class ReliabilityDiagram(nn.Module):
    """
    Generate reliability diagram for binary classifier
    as described in [1] based on the code for multiple
    classes in [2]

    [1]: https://towardsdatascience.com/introduction-to-reliability-diagrams-for-probability-calibration-ed785b3f5d44
    [2]: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """
    def __init__(self, nb_bins=10):
        super(ReliabilityDiagram, self).__init__()
        self.nb_bins = nb_bins
        bin_boundaries = torch.linspace(0, 1, nb_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        confidences = torch.sigmoid(logits)
        positives_per_bin = []
        confidence_per_bin = []
        # expected calibration error
        ece = 0
        # maximum ece
        max_ece = 0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            # if there is any element in that range
            if in_bin.float().mean().item() > 0:
                # positives in bin
                positives_in_bin = labels[in_bin].float().mean().item()
                positives_per_bin.append(positives_in_bin)
                # confidence in bin
                confidences_in_bin = confidences[in_bin].float().mean().item()
                confidence_per_bin.append(confidences_in_bin)
                ece_i = abs(positives_in_bin - confidences_in_bin)
                if ece_i > max_ece:
                    max_ece = ece_i
                ece += ece_i

        return positives_per_bin, confidence_per_bin, ece, max_ece


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for binary classifier

    :param model: model to tune
    :param val_loader:
    """
    def __init__(self, model, val_loader):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.val_loader = val_loader
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, inp):
        logits = self.model(inp)
        temperature = self.temperature.expand(logits.size(0))
        return logits / temperature

    def temperature_scale(self):
        """
        Set temperature for model using validation set.

        Insipired from: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
        """
        nll = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        print("Temperature before scaling: %1.3f" % self.temperature)

        def eval():
            loss = 0
            for i in range(len(self.val_loader)):
                img, label = self.val_loader[i]
                loss = nll(self.forward(img), label)
                loss.backward()
            return loss

        optimizer.step(eval)
        print("Temperature after scaling: %1.3f" % self.temperature)

        return self







