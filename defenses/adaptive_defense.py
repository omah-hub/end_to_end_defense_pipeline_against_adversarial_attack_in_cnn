import torch
import torch.nn as nn


def adaptive_defense_training(model, dataloader, device, epochs=5):
    """
    Defense for adaptive attack using adversarial training.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):

        for images, labels in dataloader:

            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model