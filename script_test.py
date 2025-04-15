from config import PATH_TEST_DATASET, PATH_TEST_IMAGES, PATH_TEST_JSON, PATH_TEST_PDF, get_final_model 
from pager.benchmark.seg_detection.seg_detection_word_IoU import  SegDetectionBenchmark


page = get_final_model()
benchmark = SegDetectionBenchmark(path_dataset=PATH_TEST_DATASET,
                    page_model=page,
                    path_images=PATH_TEST_IMAGES,
                    path_pdfs=PATH_TEST_PDF,
                    path_json=PATH_TEST_JSON,
                    # save_dir="/home/daniil/rez_seg",
                    only_seg=True,
                    count_image=10,
                    name="seg")
