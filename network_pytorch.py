import torch
import pickle

# Predict Helper Function 
def predict(model, input):
    model.eval()
    with torch.no_grad():
        output = model(input)

    return output

# Training Phase
def train(model, loss, optimizer, train_data_loader, val_data_loader, epochs=100, verbose=True, device="cpu"):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_error = 0

        for x, y in train_data_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x) # Fowards Pass

            error = loss(output, y) # Error calculation
            total_error += error.item()

            optimizer.zero_grad() # Clearing old gradients
            error.backward() # Backward Pass
            optimizer.step() # Updating Gradients

        avg_error = total_error / len(train_data_loader)

        # Evaluation Round
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_data_loader:
                x = x.to(device)
                y = y.to(device)

                output = model(x)
                preds = torch.argmax(output, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        accuracy = correct / total


        if verbose:
            print(f"{epoch + 1}/{epochs}, train error={avg_error}, val accuracy: {accuracy}")

# Save Model Function
def save(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved in 'models' folder.")

# Load Model Function
def load(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Model from {path} loaded succesfully.")

    return model