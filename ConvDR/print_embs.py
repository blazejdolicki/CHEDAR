import os, pickle

file_name = "passage_emb_p__data_obj_0_job_4_0.pb"
print("filename", file_name)
path_to_shared_embs = "../../../../project/gpuuva006/CAST19_ANCE_embeddings"
pickle_path = os.path.join(path_to_shared_embs, file_name)

with open(pickle_path, 'rb') as handle:
  b = pickle.load(handle)

print("type", type(b))
print("shape", b.shape)
print("b", b)