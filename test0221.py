import torch
from tqdm import tqdm

def avg_non_zero_elements(sparse_vec):
    non_zero_counts = []
    for vec in tqdm(sparse_vec, desc="Calculating non-zero counts"):
        non_zero_count = vec._indices().size(1)
        non_zero_counts.append(non_zero_count)
    
    avg_count = sum(non_zero_counts) // len(non_zero_counts) if non_zero_counts else 0
    return avg_count

if __name__ == "__main__":
    model_base_path = "/data1/zhh/baselines/mm/icrr/models/colqwen2-sparse-lambda0223"
    for dataset in ["VisRAG-Ret-Test-ArxivQA", "VisRAG-Ret-Test-ChartQA", "VisRAG-Ret-Test-InfoVQA", "VisRAG-Ret-Test-MP-DocVQA", "VisRAG-Ret-Test-PlotQA", "VisRAG-Ret-Test-SlideVQA"]:
        try:
            qs = torch.load(f"{model_base_path}/{dataset}_qs.pt", weights_only=True)
            ps = torch.load(f"{model_base_path}/{dataset}_ps.pt", weights_only=True)

            # print(len(qs), len(ps), "\n")
            # print("qs", avg_non_zero_elements(qs))
            # print("ps", avg_non_zero_elements(ps))
            print(f"\n\ndataset: {dataset}]\n len_qs: {len(qs)} len_ps: {len(ps)}\n mean_qs: {avg_non_zero_elements(qs)}, mean_ps: {avg_non_zero_elements(ps)}\n")
        except:
            print(f"{dataset} not found\n")
            continue
