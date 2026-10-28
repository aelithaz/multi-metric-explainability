
import torch


def sensitivity_score(model, image, method="mean", samples=10):
    model.eval()
    scores = []

    with torch.no_grad():
        original = model(image)
        predicted = original.argmax(dim=1).item()
        confidence = original.softmax(dim=1)[0, predicted].item()

        for row in range(0, 28, 4):
            for col in range(0, 28, 4):
                changes = []
                for _ in range(samples if method == "noise" else 1):
                    modified = image.clone()

                    if method == "black":
                        modified[:, :, row:row + 4, col:col + 4] = -0.1307 / 0.3081
                    elif method == "mean":
                        modified[:, :, row:row + 4, col:col + 4] = 0
                    elif method == "noise":
                        region = modified[:, :, row:row + 4, col:col + 4]
                        region += torch.randn_like(region) * 0.5
                        region.clamp_(-0.1307 / 0.3081, (1 - 0.1307) / 0.3081)
                    else:
                        raise ValueError("Unknown perturbation method")

                    output = model(modified)
                    new_confidence = output.softmax(dim=1)[0, predicted].item()
                    changes.append(max(0, confidence - new_confidence))

                scores.append(sum(changes) / len(changes))

    scores = torch.tensor(scores)
    if scores.max() > 0:
        scores = scores / scores.max()

    return scores
