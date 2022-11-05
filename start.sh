#!/bin/bash
source /root/anaconda3/bin/activate py36
nohup python /opt/csicc/sample02_visual.py >> /opt/csicc/data/log/output.log &