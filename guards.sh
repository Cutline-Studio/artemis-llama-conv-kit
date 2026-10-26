#!/bin/bash
# usage: guards3.sh IID RENT_EPOCH LIFETIME_S -> detached harvester (pull on out/DONE or at lifetime-25min; destroy; verify) + hard kill at lifetime
K=$4; DSTD=$5; IID=$1; R=$2; LIFE=$3; V=/workspace/vast-venv/bin/vastai
cat > $K/killer3.sh <<EOK
#!/bin/bash
END=$((R + LIFE))
while [ \$(date +%s) -lt \$END ]; do sleep 20; [ -f $K/STOP_KILLER ] && exit 0; done
echo "\$(date) hard-kill firing" >> $K/killer3.log
for i in 1 2 3 4 5; do $V destroy instance $IID -y --api-key "\$VAST_API_KEY" >> $K/killer3.log 2>&1 && break; sleep 15; done
$V show instance $IID --raw --api-key "\$VAST_API_KEY" >> $K/killer3.log 2>&1
EOK
cat > $K/harvester3.sh <<EOH
#!/bin/bash
DST=$DSTD; END=$((R + LIFE - 1500)); mkdir -p \$DST
log(){ echo "\$(date '+%F %T %Z') \$*" >> $K/harvester3.log; }
log start
while [ \$(date +%s) -lt \$END ]; do
  ssh -F $K/ssh_cfg -o ConnectTimeout=20 gpu 'test -f /workspace/q/out/DONE' 2>/dev/null && { log DONE-seen; break; }
  sleep 60
done
for i in 1 2 3; do
  timeout 1200 rsync -a -e "ssh -F $K/ssh_cfg" --exclude 'base/' gpu:/workspace/q/out/ \$DST/ >> $K/harvester3.log 2>&1; rc=\$?
  log "rsync rc=\$rc"; [ \$rc -eq 0 ] && break; sleep 10
done
for i in 1 2 3 4 5; do $V destroy instance $IID -y --api-key "\$VAST_API_KEY" >> $K/harvester3.log 2>&1 && { log destroyed; break; }; sleep 15; done
sleep 20; $V show instance $IID --raw --api-key "\$VAST_API_KEY" >> $K/harvester3.log 2>&1
log end
EOH
chmod +x $K/killer3.sh $K/harvester3.sh
cd $K && setsid nohup ./killer3.sh >/dev/null 2>&1 </dev/null &
cd $K && setsid nohup ./harvester3.sh >/dev/null 2>&1 </dev/null &
sleep 1; pgrep -af "killer3|harvester3"
