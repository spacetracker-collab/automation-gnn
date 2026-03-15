import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx

def generate_automation_graph(num_nodes=50):

    G = nx.erdos_renyi_graph(num_nodes, 0.1)

    for node in G.nodes():

        agent_type = np.random.randint(0,5)

        feature = np.zeros(5)
        feature[agent_type] = 1

        G.nodes[node]['x'] = feature

        productivity = np.random.rand()
        G.nodes[node]['y'] = productivity

    data = from_networkx(G)

    data.x = data.x.float()
    data.y = data.y.float()

    return data
