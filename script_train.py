import torch
from torch.nn import Linear, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, GELU
from torch.nn.functional import relu
from torch_geometric.nn import BatchNorm, TAGConv
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import os
import json
import time
from config import GLAM_NODE_MODEL, GLAM_EDGE_MODEL, LOG_FILE, PARAMS,SAVE_FREQUENCY,PATH_GRAPHS_JSONS,PUBLAYNET_IMBALANCE, EDGE_IMBALANCE,EDGE_COEF
device = torch.device('cuda:0' if torch.cuda.device_count() != 0 else 'cpu')
# device = torch.device('cpu')
class NodeGLAM(torch.nn.Module):
    def __init__(self,  input_, h, output_):
        super(NodeGLAM, self).__init__()
        self.activation = GELU()
        self.batch_norm1 = BatchNorm(input_)
        self.linear1 = Linear(input_, h[0]) 
        self.tag1 = TAGConv(h[0], h[1])
        self.linear2 = Linear(h[1], h[2]) 
        self.tag2 = TAGConv(h[2], h[3])
        self.linear3 = Linear(h[3]+input_, h[4])
        self.linear4 =Linear(h[4], h[5])
        self.classifer = Linear(h[5], output_)

    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm1(x)
        h = self.linear1(x)
        h = self.activation(h)
        h = self.tag1(h, edge_index)
        h = self.activation(h)
        
        h = self.linear2(h)
        h = self.activation(h)
        h = self.tag2(h, edge_index)
        h = self.activation(h)
        a = torch.cat([x, h], dim=1)
        a = self.linear3(a)
        a = self.activation(a)
        a = self.linear4(a)

        cl = self.classifer(self.activation(a))
        # a = torch.softmax(a, dim=-1)
        return a, cl

class EdgeGLAM(torch.nn.Module):
    def __init__(self, input_, h, output_):
        super(EdgeGLAM, self).__init__()
        self.activation = GELU()
        self.batch_norm2 = BatchNorm(input_, output_)
        self.linear1 = Linear(input_, h[0]) 
        self.linear2 = Linear(h[0], output_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm2(x)
        h = self.linear1(x)
        h = self.activation(h)
        h = self.linear2(h)
        # h = torch.sigmoid(h)
        return torch.squeeze(h, 1)

class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
                    #BCELoss
        self.bce = BCEWithLogitsLoss(pos_weight=torch.tensor([EDGE_IMBALANCE]).to(device))
        self.ce = CrossEntropyLoss(weight=torch.tensor(PUBLAYNET_IMBALANCE).to(device))

    def forward(self, n_pred, n_true, e_pred, e_true):
        loss = self.ce(n_pred, n_true) + EDGE_COEF*self.bce(e_pred, e_true)
        return loss

class GLAMDataset(Dataset):
    def __init__(self, json_dir):
        self.json_dir = json_dir
        self.files = sorted(os.listdir(self.json_dir))
        self.count = len(self.files)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        path = os.path.join(self.json_dir, self.files[idx])
        with open(path, 'r') as f:
            data = json.load(f)
        return data

def delete_error_nodes(graph):
    error_nodes = [i for i, n in enumerate(graph["true_nodes"]) if n == -1]
    true_nodes = [i for i, n in enumerate(graph["true_nodes"]) if n != -1]
    for index in sorted(error_nodes, reverse=True):
        del graph["nodes_feature"][index]
        del graph["true_nodes"][index]

    error_edges = [i for i, e in enumerate(zip(graph["A"][0], graph["A"][1])) 
                   if e[0] in error_nodes or 
                      e[1] in error_nodes]
  
    for index in sorted(error_edges, reverse=True):
        del graph["A"][0][index]
        del graph["A"][1][index]
        del graph["edges_feature"][index]
        del graph["true_edges"][index]
    new = dict()
    for i, n in enumerate(true_nodes):
        new[n] = i
    
    for i in range(len(graph["A"][0])):
        graph["A"][0][i] = new[graph["A"][0][i]]
        graph["A"][1][i] = new[graph["A"][1][i]]
        

def get_tensor_from_graph(graph):
    def class_node(n):
        rez = [0, 0, 0, 0, 0]
        if n!= -1:
            rez[n] = 1
        return rez
    delete_error_nodes(graph)
    i = graph["A"]
    v_in = [1 for e in graph["edges_feature"]]
    y = graph["edges_feature"]

    v_true = graph["true_edges"]
    n_true = [class_node(n) for n in graph["true_nodes"]]
    x = graph["nodes_feature"]
    N = len(x)
    
    X = torch.tensor(data=x, dtype=torch.float32).to(device)
    Y = torch.tensor(data=y, dtype=torch.float32).to(device)
    sp_A = torch.sparse_coo_tensor(indices=i, values=v_in, size=(N, N), dtype=torch.float32).to(device)
    E_true = torch.tensor(data=v_true, dtype=torch.float32).to(device)
    N_true = torch.tensor(data=n_true, dtype=torch.float32).to(device)
    if X.shape[1] != PARAMS["node_featch"]:
        X = []
    if Y.shape[1] != PARAMS["edge_featch"]:
        X = []
    return X, Y, sp_A, E_true, N_true, i

def validation(models, dataset, criterion):
    my_loss_list = []
    for batch in dataset:
        my_loss_list_batch = []
        for j, graph in enumerate(batch):
            if not 'true_nodes' in graph.keys():
                continue
            X, Y, sp_A, E_true, N_true, i = get_tensor_from_graph(graph)
            if len(X) in (0, 1):                       
                continue
            Node_emb, Node_class = models[0](X, sp_A)
            Omega = torch.cat([Node_emb[i[0]],Node_emb[i[1]], X[i[0]], X[i[1]], Y],dim=1).to(device)
            E_pred = models[1](Omega)
            loss = criterion(Node_class, N_true, E_pred, E_true)
            my_loss_list_batch.append(loss.item())
        my_loss_list.append(np.mean(my_loss_list_batch))
        print(f"{(j+1)/len(dataset)*100:.2f} % loss = {my_loss_list[-1]:.5f} {' '*30}", end='\r')
    return np.mean(my_loss_list)


def split_index_train_val(dataset, val_split=0.2, shuffle=True, seed=1234,batch_size=64):
    N = len(dataset)
    count_batchs = int(N*(1-val_split))//batch_size
    train_size = count_batchs * batch_size 
    indexs = [i for i in range(N)]
    np.random.shuffle(indexs)
    train_indexs = indexs[:train_size]
    val_indexs = indexs[train_size:]
    batchs_train_indexs = [[train_indexs[k*batch_size+i] for i in range(batch_size)] for k in range(count_batchs)]
    return batchs_train_indexs, val_indexs    

def train_step(models, batch, optimizer, criterion):
    optimizer.zero_grad()
    my_loss_list = []
   
    for j, graph in enumerate(batch):
        if not 'true_nodes' in graph.keys():
                continue
        X, Y, sp_A, E_true, N_true, i = get_tensor_from_graph(graph)
        if len(X) in (0, 1):                       
            continue
        Node_emb, Node_class = models[0](X, sp_A)
        Omega = torch.cat([Node_emb[i[0]],Node_emb[i[1]], X[i[0]], X[i[1]], Y],dim=1).to(device)
        E_pred = models[1](Omega)
        loss = criterion(Node_class, N_true, E_pred, E_true)
        my_loss_list.append(loss.item())
        print(f"Batch loss={my_loss_list[-1]:.4f}" + " "*40, end="\r")
        loss.backward()
    optimizer.step()
    return np.mean(my_loss_list)

def train_model(params, models, dataset, save_frequency=5):  
    optimizer = torch.optim.Adam(
    list(models[0].parameters()) + list(models[1].parameters()),
    lr=params["learning_rate"],
    )
    criterion = CustomLoss()
    models[0].to(device)
    models[1].to(device)
    loss_list = []
    with open(LOG_FILE, 'a') as f:
        for key, val in params.items():
            f.write(f"{key}:\t{val}\n")
    
    train_dataset, val_dataset = split_index_train_val(dataset, val_split=0.1, batch_size=params["batch_size"])
    for k in range(params["epochs"]):
        my_loss_list = []
        if k == 0:
            start = time.time()
        for l, batch_indexs in enumerate(train_dataset):
            batch = [dataset[ind] for ind in batch_indexs]
            batch_loss = train_step(models, batch, optimizer, criterion)
            my_loss_list.append(batch_loss)
            print(f"Batch # {l+1} loss={my_loss_list[-1]:.4f}" + " "*40)
            if (k == 0 and l==0):
                print(f"Время обучения batch'а {time.time()-start:.2f} сек")
        train_val = np.mean(my_loss_list)
        loss_list.append(train_val)
        batchs = [[dataset[ind] for ind in val_dataset]]
        validation_val = validation(models, batchs, criterion)
        print("="*10, f"EPOCH #{k+1}","="*10, f"({train_val:.4f}/{validation_val:.4f})")
        if k == 0:
            print(f"Время обучения epoch {time.time()-start:.2f} сек")    
            
        with open(LOG_FILE, 'a') as f:
            f.write(f"EPOCH #{k}\t {train_val:.8f} (VAL: {validation_val:.8f})\n")  
        if (k+1) % save_frequency == 0:
            num = k//save_frequency
            torch.save(models[0].state_dict(), GLAM_NODE_MODEL+f"_tmp_{num}")
            torch.save(models[1].state_dict(), GLAM_EDGE_MODEL+f"_tmp_{num}")
    with open(LOG_FILE, 'a') as f:
        f.write(f"Время обучения: {time.time()-start:.2f} сек")
    torch.save(models[0].state_dict(), GLAM_NODE_MODEL)
    torch.save(models[1].state_dict(), GLAM_EDGE_MODEL)



if __name__ == "__main__":
    dataset = GLAMDataset(PATH_GRAPHS_JSONS)
     
    try:
        str_ = f"""DATASET INFO:
count row: {len(dataset)}
first: {dataset[0].keys()}
\t A:{np.shape(dataset[0]["A"])}
\t nodes_feature:{np.shape(dataset[0]["nodes_feature"])}
\t edges_feature:{np.shape(dataset[0]["edges_feature"])}
\t true_edges:{np.shape(dataset[0]["true_edges"])}
\t true_nodes:{np.shape(dataset[0]["true_nodes"])}
end:{dataset[-1].keys()}
\t A{np.shape(dataset[-1]["A"])}
\t nodes_feature:{np.shape(dataset[-1]["nodes_feature"])}
\t edges_feature:{np.shape(dataset[-1]["edges_feature"])}
\t true_edges:"{np.shape(dataset[-1]["true_edges"])}
\t true_nodes:{np.shape(dataset[-1]["true_nodes"])}

"""
        print(str_)
        with open(LOG_FILE, 'a') as f:    
            f.write(str_)
    except:
        print(dataset)

    COUNT_CLASS_NODE = 5
    node_glam = NodeGLAM(PARAMS["node_featch"], PARAMS["H1"], COUNT_CLASS_NODE)
    SIZE_VEC_FOR_EDGE = 2*PARAMS["node_featch"]+2*PARAMS["H1"][-1] + PARAMS["edge_featch"]
    edge_glam = EdgeGLAM(SIZE_VEC_FOR_EDGE, PARAMS["H2"], 1)
    train_model(PARAMS, [node_glam, edge_glam], dataset, save_frequency=SAVE_FREQUENCY)
