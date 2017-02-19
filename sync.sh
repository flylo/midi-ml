#!/usr/bin/env bash

user_name="$( echo ${USER} | tr '[:upper:]' '[:lower:]')"
cluster_name="${user_name}-ml-cluster"
persistent_data_path="/home/jovyan/persistent_data"
local_data_path="`pwd`"
usage="Usage: sh sync.sh [get|push]"

auth_to_cluster() {
    gcloud container clusters get-credentials ${cluster_name}
    pod_name=`kubectl get pod -o=custom-columns=NAME:.metadata.name | grep jupyter-server-pod`
}

get_remote_files() {
    if [ $# -ne 1 ];
        then echo "Must specify a file in ${persistent_data_path}"
        exit 1
    fi
    file=$1
    auth_to_cluster
    kubectl cp ${pod_name}:${persistent_data_path}/${file} ${local_data_path}/${file}
}

copy_local_files() {
    if [ $# -ne 1 ];
        then echo "Must specify a file in ${local_data_path}"
        exit 1
    fi
    file=$1
    auth_to_cluster
    kubectl cp ${local_data_path}/${file} ${pod_name}:${persistent_data_path}/${file}
}

if [ $# -eq 0 ];
    then echo "${usage}"
    exit 1
fi

if [ $1 = "get" ]
then
    get_remote_files $2
elif [ $1 = "push" ]
then
    copy_local_files $2
else
    echo "${usage}"
fi
