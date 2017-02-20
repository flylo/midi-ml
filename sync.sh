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
    base_file=`basename ${file}`
    auth_to_cluster
    echoRun "kubectl cp ${pod_name}:${persistent_data_path}/${file} ${local_data_path}/${base_file}"
}

echoRun() {
    echo "${yellow}> $1${reset}"
    eval $1
}


copy_local_files() {
    if [ $# -ne 2 ];
        then echo "Must specify source file in ${local_data_path} and a target directory in ${persistent_data_path}"
        exit 1
    fi
    source_file=$1
    target_dir=$2
    base_file=`basename ${source_file}`
    auth_to_cluster
    echoRun "kubectl cp ${source_file} ${pod_name}:${persistent_data_path}/${target_dir}/"
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
    copy_local_files $2 $3
else
    echo "${usage}"
fi
