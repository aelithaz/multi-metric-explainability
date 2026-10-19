import torch


def sensitivity_score(model, image):
    model.eval()
    scores = []

    with torch.no_grad():
        original = model(image)
        predicted = original.argmax(dim=1).item()
        confidence = original.softmax(dim=1)[0, predicted].item()

        for row in range(0, 28, 4):
            for col in range(0, 28, 4):
                modified = image.clone()
                modified[:, :, row:row + 4, col:col + 4] = 0

                output = model(modified)
                new_confidence = output.softmax(dim=1)[0, predicted].item()
                scores.append(max(0, confidence - new_confidence))

    scores = torch.tensor(scores)
    if scores.max() > 0:
        scores = scores / scores.max()

    return scores


def score(model, image):
    """Return 49 region importance scores for one image shaped (1, 1, 28, 28)."""
    model.eval()
    with torch.no_grad():
        original = model(image).softmax(dim=1)
        predicted = original.argmax(dim=1).item()
        scores = []
        for row in range(7):
            for col in range(7):
                changed = image.clone()
                changed[:, :, row * 4:(row + 1) * 4, col * 4:(col + 1) * 4] = 0
                confidence = model(changed).softmax(dim=1)[0, predicted]
                scores.append(abs(original[0, predicted] - confidence).item())
    scores = torch.tensor(scores)
    return scores / scores.max().clamp(min=1e-8)
