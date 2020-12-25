import torch


def binary_acc(model, data):
    """
    Calculate accuracy on binary classification model
    :param model: Model to evaluate
    :param data: Evaluation or Test data with
        [(img1, label1), (img2, label2)...]
    :return:
    """
    outputs = []
    accuracy = 0
    for i in range(len(data)):
        img, label = data[i]
        output = torch.sigmoid(model(img))
        outputs.append(output)
        accuracy += (label == torch.round(output)).item()
    accuracy /= len(data)
    return accuracy*100
