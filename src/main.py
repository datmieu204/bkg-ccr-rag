import os
import torch
from tqdm import tqdm
from src.data_loader.dataset import BioMedDataset

def main():
    print("Connecting to Neo4j and preparing to download...")
    dataset = BioMedDataset(
        root=os.getcwd(),
        uri="bolt://localhost:25515",
        user="neo4j",
        password="datmieu2004cgx",
        split="all",
        neighbor_hops=2,
        max_texts=30,
        include_summary=True,
        include_relation_texts=True,
        use_cache=True,
        cache_dir="kaggle_dataset/cache"
    )
    
    print("Fetching Graph Metadata...")
    metadata = dataset.loader.fetch_hgt_metadata()
    torch.save(metadata, "kaggle_dataset/graph_metadata.pt")
    
    total = len(dataset)
    print(f"Found {total} graphs. Starting extraction...")
    
    for i in tqdm(range(total), desc="Exporting to .pt files"):
        _ = dataset[i]
        
    dataset.close()
    print("XUẤT DỮ LIỆU THÀNH CÔNG!")
    print("Hãy zip thư mục 'kaggle_dataset' lại và đưa lên Kaggle.")

if __name__ == "__main__":
    main()