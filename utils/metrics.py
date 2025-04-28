import torch

def calculate_metrics(model, data_loader, device):
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for images, sensors, targets in data_loader:
            images, sensors, targets = images.to(device), sensors.to(device), targets.to(device)
            outputs = model(images, sensors)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    print(f"Validation/Test Loss: {total_loss/len(data_loader)}")
