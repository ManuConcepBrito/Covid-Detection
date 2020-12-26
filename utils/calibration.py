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
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            # if there is any element in that range
            if in_bin.float().mean().item() > 0:
                # positives in bin
                positives_per_bin.append(labels[in_bin].float().mean().item())
                # confidence in bin
                confidence_per_bin.append(confidences[in_bin].float().mean().item())

        return positives_per_bin, confidence_per_bin


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for binary classifier

    :param model: model to tune
    :param val_data: validation set to perform scaling
    :param val_labels: validation set labels
    """
    def __init__(self, model, val_data, val_labels):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.val_data = val_data
        self.val_labels = val_labels
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self):
        logits = self.model(self.val_data)
        temperature = self.temperature.expand(logits.size(0))
        return logits / temperature

    def temperature_scale(self):
        """
        Set temperature for model using validation set.

        Insipired from: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
        """
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iterm=50)

        def eval():
            loss = nll(self.forward(), self.val_labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        return self







