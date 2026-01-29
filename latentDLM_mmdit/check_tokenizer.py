# File: check_tokenizer.py
import os
from pathlib import Path

tokenizer_path = "/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/data/huggingface/tokenizers/bert-base-uncased"

print(f"Checking tokenizer directory: {tokenizer_path}")
print(f"Directory exists: {os.path.exists(tokenizer_path)}")

if os.path.exists(tokenizer_path):
    print("\nContents:")
    for item in os.listdir(tokenizer_path):
        full_path = os.path.join(tokenizer_path, item)
        if os.path.isdir(full_path):
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item} ({os.path.getsize(full_path)} bytes)")

# Check if it's actually a model directory instead
parent_dir = Path(tokenizer_path).parent
print(f"\nParent directory: {parent_dir}")
if os.path.exists(parent_dir):
    for item in os.listdir(parent_dir):
        print(f"  {item}")