#!/bin/bash
# download_openwebtext.sh
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/subsets

# Download all 21 files
for i in {00..20}; do
    printf -v num "%02d" $i
    echo "Downloading subset $num..."
    wget https://openwebtext2.s3.amazonaws.com/urlsf_subset${num}.tar
    
    # Optional: extract after download
    # tar -xvf urlsf_subset${num}.tar
done

echo "All files downloaded!"