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
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.itme())
            # if there is any element in that range
            if in_bin.float().mean().item() > 0:
                # positives in bin
                positives_per_bin.append(labels[in_bin].float().mean().item())
                # confidence in bin
                confidence_per_bin.append(confidences[in_bin].float().mean().item())

        return positives_per_bin, confidence_per_bin

    def plot(self):
        pass
