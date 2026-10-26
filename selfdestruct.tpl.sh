#!/bin/bash
# On-instance primary guard: destroy THIS instance with its scoped CONTAINER_API_KEY at HARD, or 20 min after out/DONE.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(CONTAINER_ID|CONTAINER_API_KEY)=' | sed 's/^/export /')"
HARD=__HARD__; DONEF=/workspace/q/out/DONE
while true; do now=$(date +%s); [ $now -ge $HARD ] && break
  if [ -f $DONEF ] && [ $(( now - $(stat -c %Y $DONEF) )) -ge 1200 ]; then break; fi; sleep 30; done
echo "$(date -u) self-destroy firing" >> /workspace/selfdestruct.log
for i in 1 2 3 4 5 6; do vastai destroy instance $CONTAINER_ID -y --api-key "$CONTAINER_API_KEY" >> /workspace/selfdestruct.log 2>&1 && break; sleep 20; done
sync; shutdown -h now 2>/dev/null; kill -9 1 2>/dev/null
