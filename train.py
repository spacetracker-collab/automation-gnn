import torch
import torch.nn.functional as F
from model import AutomationGNN
from dataset import generate_automation_graph

data = generate_automation_graph()

model = AutomationGNN(
    input_dim=5,
    hidden_dim=32,
    output_dim=1
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):

    model.train()
    optimizer.zero_grad()

    pred = model(data.x, data.edge_index).squeeze()

    loss = F.mse_loss(pred, data.y)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("Epoch:", epoch, "Loss:", loss.item())


torch.save(model.state_dict(), "automation_gnn_model.pt")
print("Model saved")
