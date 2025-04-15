import torch
from torch.nn import Linear, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, GELU
from torch.nn.functional import relu
from torch_geometric.nn import BatchNorm, TAGConv
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import os
import json
import time
from config import GLAM_MODEL, LOG_FILE, PARAMS,SAVE_FREQUENCY,PATH_GRAPHS_JSONS
from config import TorchModel, CustomLoss
# device = torch.device('cuda:0' if torch.cuda.device_count() != 0 else 'cpu')
SKIP_INDEX = []
device = torch.device('cpu')


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

    def __str__(self):
        return f"""DATASET INFO:
count row: {len(self)}
first: {self[0].keys()}
\t A:{np.shape(self[0]["A"])}
\t nodes_feature:{np.shape(self[0]["nodes_feature"])}
\t edges_feature:{np.shape(self[0]["edges_feature"])}
\t true_edges:{np.shape(self[0]["true_edges"])}
\t true_nodes:{np.shape(self[0]["true_nodes"])}
end:{self[-1].keys()}
\t A{np.shape(self[-1]["A"])}
\t nodes_feature:{np.shape(self[-1]["nodes_feature"])}
\t edges_feature:{np.shape(self[-1]["edges_feature"])}
\t true_edges:"{np.shape(self[-1]["true_edges"])}
\t true_nodes:{np.shape(self[-1]["true_nodes"])}

"""

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
        if n != -1:
            rez[n] = 1
        return rez

    delete_error_nodes(graph)
    edge_index = torch.tensor(graph["A"], dtype=torch.long).to(device)
    edge_raw = graph["edges_feature"]
    v_in = [1 for _ in edge_raw]
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    v_true = graph["true_edges"]
    node_true = [class_node(n) for n in graph["true_nodes"]]

    nodes_feature = graph["nodes_feature"]
    N = len(nodes_feature)

    true_edge_cat = []
    for u, v, bin_label in zip(src_list, dst_list, v_true):
        if bin_label == 1 and node_true[u] == node_true[v]:
            true_edge_cat.append(node_true[u])
        else:
            true_edge_cat.append([0., 0., 0., 0., 0.])

    node_x = torch.tensor(data=nodes_feature, dtype=torch.float32).to(device)
    edge_raw_tensor = torch.tensor(data=edge_raw, dtype=torch.float32).to(device)
    sp_A = torch.sparse_coo_tensor(indices=edge_index, values=v_in, size=(N, N), dtype=torch.float32).to(device)
    true_edge_cat_tensor = torch.tensor(data=true_edge_cat, dtype=torch.float32).to(device)
    true_edge_bin_tensor = torch.tensor(data=v_true, dtype=torch.float32).to(device)

    # # Проверка размерностей: если фичей узлов или ребер не соответствует ожидаемым, вернуть None
    # if node_x.shape[1] != PARAMS["node_featch"]:
    #     return None
    # if edge_raw_tensor.shape[1] != PARAMS["edge_featch"]:
    #     return None

    return node_x, edge_raw_tensor, sp_A, true_edge_cat_tensor, true_edge_bin_tensor, edge_index


def step(model: torch.nn.Module, batch, optimizer, criterion, train=True):
    if train:
        optimizer.zero_grad()
    my_loss_list = []
    for j, graph in enumerate(batch):
        try:
            tensors = get_tensor_from_graph(graph)
            if tensors is None:
                continue
            node_x, edge_raw, sp_A, true_edge_cat, true_edge_bin, edge_index = tensors
            outputs = model(node_x, edge_raw, edge_index)
            edge_multi_class = outputs["edge_multi_class"]
            edge_bin_class = outputs["edge_bin_class"]
            loss = criterion(edge_multi_class, true_edge_cat, edge_bin_class, true_edge_bin)
            my_loss_list.append(loss.item())
            print(f"Batch loss={my_loss_list[-1]:.4f}" + " "*40, end="\r")
            if train:
                loss.backward()
        except Exception as e:
            print(e)
            if "edges_feature" in graph.keys():
                print(np.array(graph['edges_feature']).shape)
            if "nodes_feature" in graph.keys():
                print(np.array(graph['nodes_feature']).shape)


    if train:
        optimizer.step()
    return np.mean(my_loss_list)


def validation(model, dataset, criterion):
    my_loss_list = []
    for batch in dataset:
        rez = step(model, batch, optimizer=None, criterion=criterion, train=False)
        my_loss_list.append(rez)

    return np.mean(my_loss_list)

def split_index_train_val(dataset, val_split=0.2, shuffle=True, seed=1234,batch_size=64):
    N = len(dataset)
    count_batchs = int(N*(1-val_split))//batch_size
    train_size = count_batchs * batch_size
    indexs = [i for i in range(N)]
    if shuffle:
        np.random.shuffle(indexs)
    train_indexs = indexs[:train_size]
    val_indexs = indexs[train_size:]
    batchs_train_indexs = [[train_indexs[k*batch_size+i] for i in range(batch_size)] for k in range(count_batchs)]
    return batchs_train_indexs, val_indexs


def train_model(params, model, dataset, save_frequency=5, start_epoch=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    criterion = CustomLoss(params["loss_params"])

    model.to(device)
    criterion.to(device)

    loss_list = []

    train_dataset, val_dataset = split_index_train_val(dataset, val_split=0.1, batch_size=params["batch_size"])

    for k in range(start_epoch, params["epochs"]):
        batch_loss_list = []
        if k == 0:
            start = time.time()
        for l, batch_indices in enumerate(train_dataset):
            batch = [dataset[ind] for ind in batch_indices]
            batch_loss = step(model, batch, optimizer, criterion)
            batch_loss_list.append(batch_loss)
            print(f"Batch # {l + 1} loss={batch_loss_list[-1]:.4f}" + " " * 40)
            if (k == start_epoch and l == 0):
                print(f"Время обучения batch'а {time.time() - start:.2f} сек")

        train_loss = np.mean(batch_loss_list)
        loss_list.append(train_loss)

        val_batch = [[dataset[ind] for ind in val_dataset]]
        val_loss = validation(model, val_batch, criterion)
        print("=" * 10, f"EPOCH #{k + 1}", "=" * 10, f"({train_loss:.4f}/{val_loss:.4f})")
        if k == start_epoch:
            print(f"Время обучения epoch {time.time() - start:.2f} сек")

        log(f"EPOCH #{k}\t {train_loss:.8f} (VAL: {val_loss:.8f})\n")
        if (k + 1) % save_frequency == 0:
            num = k // save_frequency
            torch.save(model.state_dict(), f"{GLAM_MODEL}_tmp_{num}")
    log(f"Время обучения: {time.time()-start:.2f} сек")
    torch.save(model.state_dict(), GLAM_MODEL)

def load_checkpoint(model, path_model,restart_num=None):
    dir_model = os.path.dirname(path_model)
    name_model = os.path.basename(path_model)
    names = [n for n in os.listdir(dir_model) if name_model+'_tmp_' in n]
    if restart_num is None:
        list_num = [int(n.split("_tmp_")[-1]) for n in names]
        if len(list_num) == 0:
            return
        restart_num = max(list_num)

    checkpoint_path = os.path.join(dir_model, name_model+f"_tmp_{restart_num}")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print(checkpoint_path)
    return restart_num

def log(str_):
    with open(LOG_FILE, 'a') as f:
        f.write(str_)


if __name__ == "__main__":
    is_restart = False
    restart_num = None
    dataset = GLAMDataset(PATH_GRAPHS_JSONS)
    import datetime

    if is_restart:
        log("R E S T A R T ")
    log(datetime.datetime.now().__str__() + '\n')
    try:
        str_ = dataset.__str__()
        str_ += '\n'.join(f"{key}:\t{val}" for key, val in PARAMS.items())
        print(str_)
        if not is_restart:
            log(str_)
    except:
        pass
        # print(path)

    NUM_EDGE_CLASSES = 5
    model: torch.nn.Module = TorchModel(node_input_dim=PARAMS["node_featch"],
    node_hidden_dims=PARAMS["H1"],
    node_emb_dim=PARAMS["H1"][-1],
    edge_raw_dim=PARAMS["edge_featch"],
    edge_hidden_dims=PARAMS["H2"],
    edge_emb_dim=PARAMS["H2"][-1],
    cat_hidden_dims=[64, 32],  # пример для многоклассового классификатора
    num_edge_classes=NUM_EDGE_CLASSES,
    bin_hidden_dims=[64, 32])
    if is_restart:
        restart_num = load_checkpoint(model, GLAM_MODEL)

    start_epoch = 0 if restart_num is None else (restart_num + 1) * SAVE_FREQUENCY
    train_model(PARAMS, model, dataset, save_frequency=SAVE_FREQUENCY, start_epoch=start_epoch)