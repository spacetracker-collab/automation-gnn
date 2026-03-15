import torch
from model import AutomationGNN
from dataset import generate_automation_graph
import matplotlib.pyplot as plt

# Load graph
data = generate_automation_graph()

# Recreate model
model = AutomationGNN(
    input_dim=5,
    hidden_dim=32,
    output_dim=1
)

# Load trained weights
model.load_state_dict(torch.load("automation_gnn_model.pt"))

model.eval()

# Run prediction
with torch.no_grad():
    predictions = model(data.x, data.edge_index)

print(predictions)

values = predictions.squeeze().numpy()

plt.hist(values, bins=20)
plt.title("Predicted Agent Productivity Distribution")
plt.show()


