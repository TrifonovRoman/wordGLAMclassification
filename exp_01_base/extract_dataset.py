import os 
from pager.page_model.sub_models import JsonWithFeatchs
from pager.page_model.sub_models.dtype import ImageSegment
from .pager_models import featch_words_and_styles, featch_A, nodes_feature,edges_feature,nodes_feature_new_styles


def true_class_from_publaynet(blocks, words, A):
    # Вспомогательные функции
    def get_block_seg(json_block):
        seg = ImageSegment(dict_p_size=json_block)
        seg.add_info("label", json_block["label"])
        return seg
    def get_word_seg(json_word):
        return ImageSegment(dict_2p=json_word['segment'])
    
    def is_one_block(word1, word2, blocks):
        for block in blocks:
            if block.is_intersection(word1) and block.is_intersection(word2):
                return 1
        return 0
    
    def get_class_node(word, blocks):
        for block in blocks:
            if block.is_intersection(word):
                return block.get_info("label")
        return -1
    
    # Получение верных меток из датасета PubLayNet
    block_segs = [get_block_seg(bl) for bl in blocks]
    word_segs = [get_word_seg(w) for w in words]
    edges_ind = [is_one_block(word_segs[i],word_segs[j], block_segs) for i, j in zip(A[0], A[1])]
    nodes_ind = [get_class_node(w, block_segs) for w in word_segs]
    return [edges_ind, nodes_ind]

def extract(path_dataset, path_img_publaynet=None, path_pdf_publaynet=None):
    json_with_featchs = JsonWithFeatchs()
    files = os.listdir(path_dataset)
    N = len(files)
    for i, file in enumerate(files):
        print(file, end=' ')

        try:
            path_json = os.path.join(path_dataset, file)
            json_with_featchs.read_from_file(path_json)
            path_pdf = os.path.join(path_pdf_publaynet, json_with_featchs.json['file_name'][:-4]+'.pdf')

            json_with_featchs.add_featchs(lambda: featch_words_and_styles(path_pdf), names=['styles', 'words'],
                                is_reupdate=True, rewrite=True)

            json_with_featchs.add_featchs(lambda: featch_A(json_with_featchs.json['styles'], json_with_featchs.json['words']), names=['A'],
                                is_reupdate=True, rewrite=True)

            json_with_featchs.add_featchs(lambda: nodes_feature(json_with_featchs.json['styles'], json_with_featchs.json['words']), names=['nodes_feature'],
                                is_reupdate=True, rewrite=True) #???????????????

            json_with_featchs.add_featchs(lambda: nodes_feature_new_styles(json_with_featchs.json['styles'], json_with_featchs.json['words'], json_with_featchs.json['nodes_feature']), names=['nodes_feature'],
                                is_reupdate=False, rewrite=True)  #!!!!!!!!!!!!!!!
            json_with_featchs.add_featchs(lambda: edges_feature(json_with_featchs.json['A'], json_with_featchs.json['words']), names=['edges_feature'],
                                is_reupdate=True, rewrite=True)

            json_with_featchs.add_featchs(lambda: true_class_from_publaynet(json_with_featchs.json['blocks'], json_with_featchs.json['words'], json_with_featchs.json['A']), names=["true_edges", "true_nodes"],
                                is_reupdate=True, rewrite=True)
            
        except:
            print("error in ", file)
        
        print(f" ({(i+1)/N*100:.2f} %) "+20*" ", end='\n')  