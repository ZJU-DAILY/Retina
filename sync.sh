#!/bin/bash

function syncFolder() {
    if [ ! -d "$src_file_dir" ]; then
       echo "===>source dir [${src_file_dir}] does not exist, exiting."
       return
    fi

    echo "=========>begin sync folder, from $src_file_dir to $target_host_ip:$target_file_dir <========="

    rsync -avz --progress "$src_file_dir/" "$target_host_userName@$target_host_ip:$target_file_dir/"

    if [ "$?" -eq "0" ]; then
      echo "Folder sync successful!!!"
    else
      echo "Folder sync failed, please check the logs and retry manually."
    fi

    echo "======================================================="
}

src_file_dir=/data1/zhh/baselines/mm/icrr

target_host_userName=22351087

target_host_ip=10.82.8.200

target_file_dir=/share/home/22351087/mm/icrr

syncFolder
