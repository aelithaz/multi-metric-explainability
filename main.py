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

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(image.squeeze(), cmap="gray")
    axes[0].set_title("Original Image")

    axes[1].imshow(scores.reshape(7, 7), cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Sensitivity Heatmap")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()