#!/bin/bash

while true; do
    sleep 3
    ps -aux | grep python | grep -v grep | grep -v kill_all_python_tasks | awk '{print $2}' | xargs -I {} kill -9 {}
done