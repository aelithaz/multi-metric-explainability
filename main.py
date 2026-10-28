from torchvision import datasets, transforms
from models.model import load_model
from metrics.sensitivity import sensitivity_score
import matplotlib.pyplot as plt


def main():
    model = load_model()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    image, label = dataset[0]
    image = image.unsqueeze(0)

    prediction = model(image).argmax(dim=1).item()
    scores = sensitivity_score(model, image)

    print("Actual:", label)
    print("Predicted:", prediction)
    print("Sensitivity scores:", scores)

    methods = ["black", "mean", "noise"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    axes[0].imshow(image.squeeze(), cmap="gray")
    axes[0].set_title("Original Image")

    for i, method in enumerate(methods):
        scores = sensitivity_score(model, image, method)
        axes[i + 1].imshow(scores.reshape(7, 7), cmap="hot", vmin=0, vmax=1)
        axes[i + 1].set_title(method.capitalize())

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()