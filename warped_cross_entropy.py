import torch
import torch.nn as nn
import torch.nn.functional as F


class WarpedCrossEntropy(nn.Module):
    def __init__(
        self,
        hidden_dimension: int = 3,
        number_of_classes: int = 8,
        initial_steps: int = 1000,
    ):
        super(WarpedCrossEntropy, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.number_of_classes = number_of_classes
        self.initial_steps = initial_steps
        
        vectors = self._compute_class_vectors()
        self.register_buffer("class_vectors", vectors)

    def _compute_class_vectors(self):
        points = torch.randn(
            self.number_of_classes, self.hidden_dimension, requires_grad=True
        )
        optimizer = torch.optim.Adam([points], lr=0.01)

        for _ in range(self.initial_steps):
            optimizer.zero_grad()
            normalized = points / points.norm(dim=1, keepdim=True)
            gram = torch.mm(normalized, normalized.t())
            loss = torch.sum(torch.exp(10 * gram))
            loss.backward()
            optimizer.step()

        return (points / points.norm(dim=1, keepdim=True)).detach() # normalize and detach

    def forward(self, inputs, targets):
        picked_class_vectors = self.class_vectors[targets]
        similarities = F.cosine_similarity(inputs, picked_class_vectors, dim=1)
        loss = -similarities.mean()
        return loss

    def predict(self, inputs):
        # Compute cosine similarity between each input and all class vectors
        similarities = F.cosine_similarity(
            inputs.unsqueeze(1), self.class_vectors.unsqueeze(0), dim=2
            # for class_vectors we apply the same class vector to each dimension of the inputs, so we unsqueeze on the last dimension
            # we go from [CLASSES, OUTPUT_DIMENSION] -> [1, CLASSES, OUTPUT_DIMENSION]
            # the inputs come [BATCH_SIZE, OUTPUT_DIMENSION] we want to apply to each vector the same operation with each of the class vectors
            # so we unsqueeze on the 1st index: [BATCH_SIZE, 1, OUTPUT_DIMENSION]
            # now output dimensions align, but we do at for each class and for each batch size
        )
        # Pick the class with the highest similarity for each input
        # Finally look at the dimension on (classes) and pick the best one
        return similarities.argmax(dim=1)


class CustomCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        label_smoothing=0,
    ):
        super().__init__(
            weight, size_average, ignore_index, reduce, reduction, label_smoothing
        )

    def predict(self, inputs):
        _, predicted = torch.max(inputs.data, 1)
        return predicted
