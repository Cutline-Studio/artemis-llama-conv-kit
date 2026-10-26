#!/bin/bash
# BOX-SIDE launcher. Approved scope: ONE high-end GPU, HARD $4.00 total. Usage: launch.sh [--dry-run]
# Needs: $VAST_API_KEY in env; HF token (Meta Llama 3.1 license accepted) in $HF_TOKEN_FILE (default ./.hf_token, chmod 600).
set -u; K=/workspace/llama-conv-kit; V=/workspace/vast-venv/bin/vastai; CAP=4.00; MARGIN=0.35; DRY=${1:-}
TF=${HF_TOKEN_FILE:-$K/.hf_token}; cd $K
# ---- 1. preflight: gated access must work BEFORE renting (token never printed)
[ -s "$TF" ] || { echo "NO_HF_TOKEN: put a Hugging Face read token in $TF (chmod 600)"; exit 2; }
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(cat "$TF")" https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json)
[ "$code" = 200 ] || { echo "HF_ACCESS_DENIED http=$code: accept the Llama 3.1 license on huggingface.co/meta-llama/Llama-3.1-8B-Instruct"; exit 3; }
echo "HF gated access OK"
# ---- 2. pick offer (H100 > A100 80GB > A100 40GB > >=48GB), cost incl. ~17 GB download + 60 GB disk
pick=$(for q in "gpu_name in [H100_SXM,H100_PCIE,H100_NVL]" "gpu_name in [A100_SXM4,A100_PCIE] gpu_ram>=79000" "gpu_name in [A100_SXM4,A100_PCIE]" "gpu_ram>=47000"; do
  $V search offers "$q num_gpus=1 verified=true rentable=true direct_port_count>=1 disk_space>=60 inet_down>=500 reliability>0.98" -o dph_total --raw --api-key "$VAST_API_KEY" 2>/dev/null
done | python3 -c "
import sys, json
dec = json.JSONDecoder(); s = sys.stdin.read(); i = 0; offers = []
while i < len(s):
    s2 = s[i:].lstrip()
    if not s2: break
    o, n = dec.raw_decode(s2); offers.append(o); i = len(s) - len(s2) + n
CAP, M = $CAP, $MARGIN
for tier in offers:
    best = None
    for o in tier:
        dl = 17 * (o.get('inet_down_cost') or 0); st = 60 * (o.get('storage_cost') or 0) / 720 * 2
        life = (CAP - M - dl - st) / o['dph_total'] * 3600
        if life >= 2700 and (best is None or o['dph_total'] + dl / 1.5 < best[1]): best = (o, o['dph_total'] + dl / 1.5, dl, life)
    if best:
        o, _, dl, life = best; print(o['id'], o['gpu_name'], round(o['dph_total'], 4), round(dl, 3), int(min(life, 7200))); break
")
[ -n "$pick" ] || { echo "NO_OFFER fits \$$CAP with >=45 min"; exit 4; }
read OID GNAME DPH DLC LIFE <<< "$pick"; echo "offer $OID $GNAME \$$DPH/hr est_download \$$DLC lifetime ${LIFE}s (est total <= \$$(python3 -c "print(round($DPH*$LIFE/3600+$DLC,2))"))"
[ "$DRY" = "--dry-run" ] && exit 0
# ---- 3. rent
$V create instance $OID --image pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel --disk 60 --ssh --direct --label llama-conv-artemis --raw --api-key "$VAST_API_KEY" > create.json
IID=$(python3 -c "import json;print(json.load(open('create.json'))['new_contract'])") || exit 5
python3 - <<'PY'
import json; d = json.load(open("create.json")); d.pop("instance_api_key", None); json.dump(d, open("create.json", "w"))
PY
R=$(date +%s); echo $IID > iid.txt; echo $R > rent_epoch.txt; date > rent_start.txt; HARD=$((R + LIFE)); echo "IID=$IID HARD=$HARD"
# ---- 4. box-side guards first (killer at LIFE, harvester at LIFE-25min or on DONE)
bash guards.sh $IID $R $LIFE $K $K/out
# ---- 5. wait for ssh
for i in $(seq 1 60); do
  info=$($V show instance $IID --raw --api-key "$VAST_API_KEY" 2>/dev/null)
  st=$(echo "$info" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('actual_status'), d.get('public_ipaddr'), (d.get('ports') or {}).get('22/tcp',[{}])[0].get('HostPort'))" 2>/dev/null)
  set -- $st; [ "${1:-}" = running ] && [ "${3:-None}" != None ] && break; sleep 15
done
printf 'Host gpu\n  HostName %s\n  Port %s\n  User root\n  IdentityFile ~/.ssh/vastai_ed25519\n  IdentitiesOnly yes\n  StrictHostKeyChecking accept-new\n  UserKnownHostsFile %s/known_hosts\n  ServerAliveInterval 30\n' "$2" "$3" $K > ssh_cfg
until ssh -F ssh_cfg -o ConnectTimeout=15 gpu true 2>/dev/null; do sleep 10; done
# ---- 6. upload + on-instance self-destruct + start
sed "s/__HARD__/$HARD/" selfdestruct.tpl.sh > selfdestruct.sh
ssh -F ssh_cfg gpu 'mkdir -p /workspace/q/sft && pip install -q vastai 2>&1 | tail -1'
scp -F ssh_cfg -q train_conv.py run_conv.sh gpu:/workspace/q/ && scp -F ssh_cfg -q sft/conv_train.jsonl sft/conv_val.jsonl gpu:/workspace/q/sft/ && scp -F ssh_cfg -q selfdestruct.sh gpu:/workspace/
( umask 077; ssh -F ssh_cfg gpu 'umask 077; cat > /workspace/q/.hf_token' < "$TF" )
ssh -F ssh_cfg gpu 'chmod +x /workspace/selfdestruct.sh /workspace/q/run_conv.sh; cd /workspace && setsid nohup ./selfdestruct.sh >/dev/null 2>&1 </dev/null & cd /workspace/q && setsid nohup ./run_conv.sh '"$HARD"' > run.log 2>&1 </dev/null & sleep 2; pgrep -af "selfdestruct|run_conv"'
echo launched IID=$IID
