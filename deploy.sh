#!/usr/bin/env bash

image_version="0.0.1"
project_name=`python scripts/lookup_value_from_json google_service_key.json project_id`
repo_name="gcr.io/${project_name}/midi-ml"
user_name="$( echo $USER | tr '[:upper:]' '[:lower:]')"
cluster_name="${user_name}-ml-cluster"
zone="us-east1-b"
remote_image=`gcloud beta container images list-tags ${repo_name} | grep ${image_version} | tr -d '[:space:]'`


if [ $# -eq 0 ];
    then echo "Usage: sh deploy.sh [build|push|run|proxy]"
    exit 1
fi

if [ $1 = "build" ]
then
    echo "building image version ${image_version}"
    docker build  -t ${repo_name}:${image_version} .

elif [ $1 = "push" ]
then
    echo "pushing image version ${image_version}"
    docker build  -t ${repo_name}:${image_version} .
    gcloud docker -- push ${repo_name}:${image_version}

elif [ $1 = "run" ]
then
    if [ $# -le 2 ];
        then echo "Usage: sh deploy.sh run [local|remote] [docker_args]"
        exit 1
    else
        if [ $2 == "local" ]
        then
            if [ "`docker images | grep ${repo_name} | grep ${image_version}`" == '' ]
            then
                echo "Must build image before running locally with 'sh deploy_models.sh build'"
                exit 1
            fi
            echo "running image version ${image_version}" \
                 "locally with args '`echo "${@:3}"`'"
            docker run ${repo_name}:${image_version} run_pipeline "${@:3}"

#        PICK UP AFTER THIS
        elif [ $2 == "remote" ]
        then
            running_clusters=`gcloud container clusters list | awk '{ print $1 }' | grep ${cluster_name}`
            if [ running_clusters == '' ]
            then
                echo "creating cluster ${cluster_name} in project ${project_name}"
                gcloud container clusters create ${cluster_name} --machine-type=n1-highmem-32 --num-nodes=1
            fi
            gcloud container clusters get-credentials ${cluster_name}
            pod_id=`kubectl get pods --show-all | awk 'END{ print NR }'`
            pod_name="${cluster_name}-${pod_id}"
            kubectl run ${pod_name} --image=${repo_name}:${recent_image_version} \
                --restart Never -- run_pipeline "${@:3}"
            echo "running image version ${recent_image_version}" \
                "on pod ${pod_name}" \
                 "in cluster ${cluster_name}" \
                 "with args '`echo "${@:3}"`'"

        else
            echo "Usage: sh deploy_models.sh run [local|remote]"
            exit 1
        fi
    fi

elif [ $1 = "proxy" ]
then
    gcloud container clusters get-credentials ${cluster_name}
    echo "Dashboard running at http://localhost:8001/ui"
    kubectl proxy

else
    echo "Usage: sh deploy_models.sh [build|push|run|proxy]"
    exit 1
fi
