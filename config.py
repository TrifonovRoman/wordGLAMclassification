from example_exp import pager_models as exp_models
import os 

PATH_PUBLAYNET = os.environ["PATH_PUBLAYNET"]
START = 0
FINAL = 10000
PATH_WORDS_AND_STYLES_JSONS = os.environ["PATH_WORDS_AND_STYLES_JSONS"]
PATH_GRAPHS_JSONS = os.environ["PATH_GRAPH_JSONS"]

PATH_TEST_DATASET = PATH_PUBLAYNET
PATH_TEST_IMAGES = os.path.join(PATH_TEST_DATASET, "val")
PATH_TEST_JSON = os.path.join(PATH_TEST_DATASET, "val_compr.json")

GLAM_NODE_MODEL = "example_exp/glam_node_model"
GLAM_EDGE_MODEL = "example_exp/glam_edge_model"
LOG_FILE = "example_exp/log.txt"
PARAMS = {
        "node_featch": 37,
        "edge_featch": 2,
        "epochs": 400,
        "batch_size": 500,
        "learning_rate": 0.05,
        "H1": [256, 128, 64, 32, 16],
        "H2": [8]
    }
SAVE_FREQUENCY = 10
SEG_K = 0.5
PUBLAYNET_IMBALANCE = [1.378, 0.033, 2.327, 0.845, 0.417]
EDGE_IMBALANCE = 0.15
EDGE_COEF = 4 # 4

def get_preprocessing_models():
    return exp_models.img2words_and_styles, exp_models.words_and_styles2graph

def get_final_model():
    return exp_models.get_img2phis(conf={
            "path_node_gnn": GLAM_NODE_MODEL,
            "path_edge_linear": GLAM_EDGE_MODEL,
            "H1": PARAMS["H1"],
            "H2": PARAMS["H2"],
            "node_featch": PARAMS["node_featch"],
            "edge_featch": PARAMS["edge_featch"],
            "seg_k": SEG_K
                        
    })
