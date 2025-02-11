from example_exp import pager_models as exp_models
import os 

PATH_PUBLAYNET = ""
START = 0
FINAL = 100
PATH_WORDS_AND_STYLES_JSONS = ""
PATH_GRAPHS_JSONS = ""

PATH_TEST_DATASET = ""
PATH_TEST_IMAGES = os.path.join(PATH_TEST_DATASET, "images")
PATH_TEST_JSON = ""

GLAM_NODE_MODEL = "example_exp/glam_node_model"
GLAM_EDGE_MODEL = "example_exp/glam_edge_model"
LOG_FILE = "example_exp/log.txt"
PARAMS = {
        "node_featch": 37,
        "edge_featch": 0,
        "epochs": 100,
        "batch_size": 500,
        "learning_rate": 0.05,
        "H1": [64, 64, 64, 64, 64],
        "H2": [64]
    }
SAVE_FREQUENCY = 10
SEG_K = 0.5

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