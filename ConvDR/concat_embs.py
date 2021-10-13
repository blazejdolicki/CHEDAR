import os, pickle

SHARED_EMBS_PATH = "../../../../project/gpuuva006/CAST19_ANCE_embeddings"
OUTPUT_EMBS_PATH = "home/lcuur0054/CHEDAR/ConvDR/datasets/cast-shared/embeddings"

embs_list = []

for obj_id in range(4):
    # read embeddings of the same object
    for part_id in range(4):
        file_name = "passage_emb_p__data_obj_{}_job_4_{}.pb".format(obj_id, part_id)
        input_pickle_path = os.path.join(SHARED_EMBS_PATH, file_name)
        with open(input_pickle_path, 'rb') as handle:
            embs_array = pickle.load(handle)
        embs_list.append(embs_array)
    
    # concate them
    concat_embs = np.concatenate(data_list, axis=0)

    # write them to a single file, similarly to ConvDR repo
    out_file_name = "passage_emb_p__data_obj_{}.pb".format(obj_id)
    output_pickle_path = os.path.join(OUTPUT_EMBS_PATH, out_file_name)
    with open(output_pickle_path, 'wb') as handle:
        pickle.dump(data_array, handle, protocol=4)