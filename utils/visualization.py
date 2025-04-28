import matplotlib.pyplot as plt

def plot_predictions(true_values, predicted_values):
    plt.figure(figsize=(10,6))
    plt.plot(true_values, label="True")
    plt.plot(predicted_values, label="Predicted")
    plt.legend()
    plt.title("Prediction vs Ground Truth")
    plt.show()
