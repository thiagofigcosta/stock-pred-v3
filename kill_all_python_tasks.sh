#!/bin/bash

while true; do
    if [ -z "$(ps -aux | grep python | grep -v grep | grep -v kill_all_python_tasks)" ] ; then
      echo "No Python process found"
      exit
    fi
    sleep 3
    ps -aux | grep python | grep -v grep | grep -v kill_all_python_tasks | awk '{print $2}' | xargs -I {} kill -9 {}
done