#!/bin/bash
# ON-INSTANCE pipeline. Usage: run_conv.sh DEADLINE_EPOCH. Expects /workspace/q/.hf_token (0600). Token never echoed; deleted after download.
DL=$1; cd /workspace/q; mkdir -p out
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q "transformers==4.46.3" "peft==0.13.2" "bitsandbytes==0.44.1" "accelerate==1.1.1" hf_transfer huggingface_hub safetensors sentencepiece protobuf 2>&1 | grep -v WARNING | tail -2
HF_TOKEN="$(cat .hf_token)" python - <<'PY'
import time, json, os
from huggingface_hub import snapshot_download
t = time.time()
p = snapshot_download("meta-llama/Llama-3.1-8B-Instruct", local_dir="base", token=os.environ["HF_TOKEN"],
    allow_patterns=["*.json", "*.safetensors", "LICENSE", "USE_POLICY.md", "README.md"], ignore_patterns=["original/*"])
lic = open("base/LICENSE").read(); ok = "LLAMA 3.1 COMMUNITY LICENSE" in lic.upper()
json.dump({"repo": "meta-llama/Llama-3.1-8B-Instruct", "download_s": round(time.time() - t, 1), "license_llama31": ok,
           "use_policy_present": os.path.exists("base/USE_POLICY.md")}, open("out/base_download.json", "w"), indent=1)
assert ok
PY
rc=$?; shred -u .hf_token 2>/dev/null || rm -f .hf_token
[ $rc -eq 0 ] || { echo DOWNLOAD_OR_LICENSE_FAILED > out/STATUS; touch out/DONE; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > out/nvidia_smi.csv
python train_conv.py --deadline-epoch $DL --out out > out/train_stdout.log 2>&1
[ -f out/TRAIN_DONE ] && echo OK > out/STATUS || echo TRAIN_FAILED > out/STATUS
touch out/DONE
