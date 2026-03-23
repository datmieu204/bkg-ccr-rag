# ./kgs_builder/data_processing/data_loader.py

import os

def load_high(data_path: str) -> str:
    all_content = ""
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            all_content += line.strip() + "\n"
    return all_content