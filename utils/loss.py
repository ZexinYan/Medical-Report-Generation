import torch
import numpy as np
from torch.nn.modules import loss


class WARPLoss(loss.Module):
    def __init__(self, num_labels=204):
        super(WARPLoss, self).__init__()
        self.rank_weights = [1.0 / 1]
        for i in range(1, num_labels):
            self.rank_weights.append(self.rank_weights[i - 1] + (1.0 / i + 1))

    def forward(self, input, target) -> object:
        """

        :rtype:
        :param input: Deep features tensor Variable of size batch x n_attrs.
        :param target: Ground truth tensor Variable of size batch x n_attrs.
        :return:
        """
        batch_size = target.size()[0]
        n_labels = target.size()[1]
        max_num_trials = n_labels - 1
        loss = 0.0

        for i in range(batch_size):

            for j in range(n_labels):
                if target[i, j] == 1:

                    neg_labels_idx = np.array([idx for idx, v in enumerate(target[i, :]) if v == 0])
                    neg_idx = np.random.choice(neg_labels_idx, replace=False)
                    sample_score_margin = 1 - input[i, j] + input[i, neg_idx]
                    num_trials = 0

                    while sample_score_margin < 0 and num_trials < max_num_trials:
                        neg_idx = np.random.choice(neg_labels_idx, replace=False)
                        num_trials += 1
                        sample_score_margin = 1 - input[i, j] + input[i, neg_idx]

                    r_j = np.floor(max_num_trials / num_trials)
                    weight = self.rank_weights[r_j]

                    for k in range(n_labels):
                        if target[i, k] == 0:
                            score_margin = 1 - input[i, j] + input[i, k]
                            loss += (weight * torch.clamp(score_margin, min=0.0))
        return loss


class MultiLabelSoftmaxRegressionLoss(loss.Module):
    def __init__(self):
        super(MultiLabelSoftmaxRegressionLoss, self).__init__()

    def forward(self, input, target) -> object:
        return -1 * torch.sum(input * target)


class LossFactory(object):
    def __init__(self, type, num_labels=156):
        self.type = type
        if type == 'BCE':
            # self.activation_func = torch.nn.Sigmoid()
            self.loss = torch.nn.BCELoss()
        elif type == 'CE':
            self.loss = torch.nn.CrossEntropyLoss()
        elif type == 'WARP':
            self.activation_func = torch.nn.Softmax()
            self.loss = WARPLoss(num_labels=num_labels)
        elif type == 'MSR':
            self.activation_func = torch.nn.LogSoftmax()
            self.loss = MultiLabelSoftmaxRegressionLoss()

    def compute_loss(self, output, target):
        # output = self.activation_func(output)
        # if self.type == 'NLL' or self.type == 'WARP' or self.type == 'MSR':
        #     target /= torch.sum(target, 1).view(-1, 1)
        return self.loss(output, target)
