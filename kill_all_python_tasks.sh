#!/bin/bash

while true; do
    sleep 3
    ps -aux | grep python | grep -v grep | awk '{print $2}' | xargs -I {} kill -9 {}
done