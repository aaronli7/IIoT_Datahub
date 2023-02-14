#!/bin/bash
###
 # @Author: Qi7
 # @Date: 2023-02-14 16:54:23
 # @LastEditors: aaronli-uga ql61608@uga.edu
 # @LastEditTime: 2023-02-14 17:03:42
 # @Description: capturing the packets using tshark. Usage: 
 #              ./packet_capture.sh <name-of-interface>
### 
if [ $# -eq 0]
    then
        echo "Please provide the <name-of-interface>"
        exit 1
fi

stdbuf -i0 -oL -e0 tshark -i $1 -q -T fields -e frame.time_relative -e _ws.col.Protocol -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport