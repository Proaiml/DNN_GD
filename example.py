"""
DNN_GD - Example Usage Script
Demonstrates training a Multi-Layer Deep Neural Network from scratch using
Gradient Descent and Backpropagation, using dnn.py created by İlhan Koçaslan.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import dnn

def main():
    print("=" * 70)
    print("       DNN + GD: Deep Neural Network Trained by Gradient Descent")
    print("                 Created from Scratch by İlhan Koçaslan")
    print("=" * 70)

    # 1. Load dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "samples.json")

    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
        x_list = data["inputs"]
        y_list = data["targets"]
    else:
        # Fallback default training patterns
        x_list = [[0, 0, 0, 0, 1], [1, 1, 1, 1, 0]]
        y_list = [[0, 1, 0], [1, 1, 1]]

    # 2. Hyperparameters
    input_node = len(x_list[0])        # 5 input features
    hidden_nodes = [6, 8, 4]           # Arbitrary depth multi-layer architecture
    output_node = len(y_list[0])       # 3 output targets
    learning_rate = 0.6                # Gradient descent step size
    epochs = 800                       # Training iterations

    print(f"\n[*] Network Architecture:")
    print(f"    - Input Features:   {input_node} neurons")
    print(f"    - Hidden Layers:    {hidden_nodes} ({len(hidden_nodes)} layers)")
    print(f"    - Output Targets:   {output_node} neurons")
    print(f"    - Total Layers:     {[input_node] + hidden_nodes + [output_node]}")
    print(f"    - Learning Rate:    {learning_rate}")
    print(f"    - Training Samples: {len(x_list)}")

    # 3. Instantiate model
    print("\n[+] Initializing neural network with Xavier normal weights...")
    model = dnn.neuralNetwork(
        inputnodes=input_node,
        hiddennodes=hidden_nodes,
        outputnodes=output_node,
        learningrate=learning_rate
    )

    # 4. Training loop with MSE loss tracking
    print(f"\n[+] Training for {epochs} epochs using Gradient Descent & Backpropagation...")
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train(inputs_list=x_list, targets_list=y_list)

        # Calculate Mean Squared Error
        outputs = model.allouts[f"o{len(model.alllayer)}"]  # Final layer output
        targets = np.array(y_list, ndmin=2).T
        mse = float(np.mean((targets - outputs) ** 2))
        loss_history.append(mse)

        if epoch % 100 == 0 or epoch == 1:
            progress = int((epoch / epochs) * 25)
            bar = "=" * progress + "-" * (25 - progress)
            print(f"    Epoch [{epoch:4d}/{epochs:4d}] [{bar}] - MSE Loss: {mse:.6f}")

    print("\n[OK] Training completed successfully!")
    print(f"[*] Initial MSE Loss: {loss_history[0]:.6f} -> Final MSE Loss: {loss_history[-1]:.6f}")

    # 5. Evaluate and display prediction results
    print("\n" + "=" * 70)
    print("                         TEST PREDICTIONS")
    print("=" * 70)
    print(f"{'Sample':<8} {'Input Pattern':<22} {'Expected Target':<18} {'Predicted Output':<20}")
    print("-" * 70)

    final_outputs = model.allouts[f"o{len(model.alllayer)}"].T
    for idx, (x_in, y_exp, y_pred) in enumerate(zip(x_list, y_list, final_outputs), 1):
        pred_rounded = [round(float(v), 3) for v in y_pred]
        print(f"#{idx:<7} {str(x_in):<22} {str(y_exp):<18} {str(pred_rounded):<20}")

    print("=" * 70)

    # 6. Plot loss convergence curve
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, color="#0284c7", linewidth=2, label="MSE Training Loss")
        plt.title("Deep Neural Network - Gradient Descent Loss Convergence", fontsize=12)
        plt.xlabel("Epochs", fontsize=10)
        plt.ylabel("Mean Squared Error (MSE)", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        print("\n[+] Displaying loss convergence plot...")
        plt.show()
    except Exception as e:
        print(f"Note: Could not open plot window: {e}")

if __name__ == "__main__":
    main()
