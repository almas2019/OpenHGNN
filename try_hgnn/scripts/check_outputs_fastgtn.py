from dgl.data.utils import load_graphs
import dgl
glist_imdb = load_graphs("/home/almas/projects/def-gregorys/almas/OpenHGNN/openhgnn/dataset/imdb4GTN/graph.bin")
imdb=glist_imdb[0][0] 
imdb_hom= dgl.to_homogeneous(imdb)
glist_fast_gtn = load_graphs("/home/almas/projects/def-gregorys/almas/OpenHGNN/openhgnn/output/fastGTN/results_fastgtn.bin") 
glist_fast_gtn[0][0]