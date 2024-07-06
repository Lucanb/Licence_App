import matplotlib.pyplot as plt

# Datele de acuratete pentru antrenare si testare
train_accuracies = [0.8500, 0.9043, 0.9218, 0.9296, 0.9320, 0.9334, 0.9360, 0.9372, 0.9383, 0.9388]
test_accuracies = [0.8968, 0.9135, 0.9257, 0.9305, 0.9337, 0.9361, 0.9240, 0.9331, 0.9369, 0.9394]

epochs = range(1, 11)  # Epoci de la 1 la 10

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o', color='blue')
plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o', color='red')

plt.title('Train vs. Test Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('overfitting_plot.png')
plt.show()
