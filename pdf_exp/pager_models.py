"""
обязательно нужно реализовать 
img2words_and_styles
words_and_styles2graph
get_img2phis
"""

from pager import PageModel, PageModelUnit
from pager.page_model.sub_models import ImageModel, PDFModel, WordsAndStylesModel, SpGraph4NModel, JsonWithFeatchs, BaseExtractor
from pager.page_model.sub_models.converters import Image2WordsAndStyles, PDF2WordsAndStyles
from pager.page_model.sub_models import PhisicalModel, WordsAndStylesToGLAMBlocks
from pager import PageModel, PageModelUnit, WordsAndStylesModel, SpGraph4NModel, WordsAndStylesToSpGraph4N, WordsAndStylesToSpDelaunayGraph
from pager.page_model.sub_models.dtype import ImageSegment, StyleWord
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv(override=True)

PATH_STYLE_MODEL = os.environ["PATH_STYLE_MODEL"]

WITH_TEXT = True
TYPE_GRAPH = "4N"
EXPERIMENT_PARAMS = {
    "node_featch": 35,
    "edge_featch": 2,
    "epochs": 20,
    "batch_size": 50,
    "learning_rate": 0.05,
    "H1": [256, 128, 64, 32, 16, 8],
    "H2": [8],
    "seg_k": 0.5
}

def featch_words_and_styles(img_name):
    img2words_and_styles.read_from_file(img_name)
    img2words_and_styles.extract()
    return [
        img2words_and_styles.page_units[-1].sub_model.to_dict(is_vec=True)['styles'],
        img2words_and_styles.page_units[-1].sub_model.to_dict(is_vec=True)['words']
    ]

def featch_A(styles,words):
    words_and_styles2graph.from_dict({
        "words": words,
        "styles": styles
    })
    words_and_styles2graph.extract()
    graph_json = words_and_styles2graph.to_dict()
    return [
        graph_json['A']
    ]

def nodes_feature(styles, words):
    if len(words) == 0:
        return [[]]
    fonts = dict()
    for st in styles:
        fonts[st['id']] = st['font2vec']
    style_vec = np.array([fonts[w['style_id']] for w in words])
    text_vec = ws2g_converter.word2vec([w['content'] for w in words])

    nodes_feature = np.concat([style_vec, text_vec], axis=1)
    return [nodes_feature.tolist()]

def edges_feature(A, words):
    edges_featch = []
    for i, j in zip(A[0], A[1]):
        w1 = ImageSegment(dict_2p= words[i]['segment'])
        w2 = ImageSegment(dict_2p= words[j]['segment'])
        edges_featch.append([w1.get_angle_center(w2), w1.get_min_dist(w2)])
    # print(edges_featch)
    return [edges_featch]
      

unit_image = PageModelUnit(id="image", 
                               sub_model=ImageModel(), 
                               converters={}, 
                               extractors=[])
    
conf_words_and_styles = {"path_model": PATH_STYLE_MODEL,"lang": "eng+rus", "psm": 4, "oem": 3,"onetone_delete": True, "k": 4 }
unit_words_and_styles = PageModelUnit(id="words_and_styles", 
                            sub_model=WordsAndStylesModel(), 
                            converters={"image": Image2WordsAndStyles(conf_words_and_styles)},
                            extractors=[])


unit_pdf = PageModelUnit(id="pdf", 
                               sub_model=PDFModel(), 
                               converters={}, 
                               extractors=[])

unit_words_and_styles_pdf = PageModelUnit(id="words_and_styles", 
                            sub_model=WordsAndStylesModel(), 
                            converters={"pdf": PDF2WordsAndStyles()},
                            extractors=[])

unit_words_and_styles_start = PageModelUnit(id="words_and_styles", 
                            sub_model=WordsAndStylesModel(), 
                            converters={}, 
                            extractors=[])
conf_graph = {"with_text": True} if WITH_TEXT else None
ws2g_converter=WordsAndStylesToSpDelaunayGraph(conf_graph) if TYPE_GRAPH == "Delaunay" else WordsAndStylesToSpGraph4N(conf_graph)
unit_graph = PageModelUnit(id="graph", 
                            sub_model=SpGraph4NModel(), 
                            extractors=[],  
                            converters={"words_and_styles":  ws2g_converter})
img2words_and_styles = PageModel(page_units=[
    unit_pdf, 
    unit_words_and_styles_pdf
]) # На самом деле pdf2words_and_styles

words_and_styles2graph = PageModel(page_units=[
    unit_words_and_styles_start,
    unit_graph
])

json_with_featchs = JsonWithFeatchs()

class JsonWithFeatchsExtractor(BaseExtractor):
    def extract(self, json_with_featchs: JsonWithFeatchs):
        json_with_featchs.add_featchs(lambda: featch_words_and_styles(json_with_featchs.name_file), names=['styles', 'words'], 
                            is_reupdate=False, rewrite=False)
        
        json_with_featchs.add_featchs(lambda: featch_A(json_with_featchs.json['styles'], json_with_featchs.json['words']), names=['A'], 
                            is_reupdate=False, rewrite=False)
        
        json_with_featchs.add_featchs(lambda: nodes_feature(json_with_featchs.json['styles'], json_with_featchs.json['words']), names=['nodes_feature'], 
                            is_reupdate=False, rewrite=False) 
        
        json_with_featchs.add_featchs(lambda: edges_feature(json_with_featchs.json['A'], json_with_featchs.json['words']), names=['edges_feature'], 
                            is_reupdate=False, rewrite=False) 

class JsonWithFeatchsWithRead(JsonWithFeatchs):
    def read_from_file(self, path_file):
        self.name_file = path_file
        return super().from_dict({})
    
class Json2Blocks(WordsAndStylesToGLAMBlocks):
    def convert(self, input_model: JsonWithFeatchs, output_model: PhisicalModel):
        graph = {
            "A": input_model.json["A"], 
            "nodes_feature": input_model.json["nodes_feature"], 
            "edges_feature": input_model.json["edges_feature"]
        } 
        words = [StyleWord(w) for w in input_model.json["words"]]
        self.set_output_block(output_model, words, graph)

def get_img2phis(conf):
    json_model = PageModelUnit(id="json_model", 
                        sub_model=JsonWithFeatchsWithRead(), 
                        extractors=[JsonWithFeatchsExtractor()],
                        converters={})

    unit_phis = PageModelUnit(id="phisical_model", 
                        sub_model=PhisicalModel(), 
                        extractors=[], 
                        converters={"json_model": Json2Blocks(conf=conf)})
    return PageModel(page_units=[
        json_model,
        unit_phis
    ])


