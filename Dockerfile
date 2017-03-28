# starting with the jupyter scipy image because it's got lots of python goodies already
FROM jupyter/scipy-notebook:2410ad57203a

USER root

# Never prompts the user for choices on installation/configuration of packages
ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux

RUN apt-get --allow-unauthenticated update \
  && apt-get install -yqq --no-install-recommends curl \
  && pip install google-api-python-client \
  && pip install googleapis-common-protos \
  && pip install google-cloud-storage \
  && pip install mido

ENV HOME /home/jovyan

# switch to jovyan user account to install Google Cloud SDK
USER jovyan
WORKDIR ${HOME}
ENV HOME ${HOME}
ENV CLOUDSDK_PYTHON_SITEPACKAGES 1
ENV CLOUDSDK_PYTHON /usr/bin/python2.7
ENV GOOGLE_APPLICATION_CREDENTIALS ${HOME}/google_service_key.json

# Install the Google Cloud SDK. This has to go down here since we're installing as the jovyan user
RUN curl https://dl.google.com/dl/cloudsdk/channels/rapid/google-cloud-sdk.zip > google-cloud-sdk.zip \
 && unzip google-cloud-sdk.zip && rm google-cloud-sdk.zip \
 && google-cloud-sdk/install.sh --usage-reporting=true --path-update=true --bash-completion=true \
 && google-cloud-sdk/bin/gcloud config set --installation component_manager/disable_update_check true \
 && sed -i -- 's/\"disable_updater\": false/\"disable_updater\": true/g' google-cloud-sdk/lib/googlecloudsdk/core/config.json \
 && mkdir .ssh

ADD google_service_key.json ${HOME}/google_service_key.json

ENV PATH google-cloud-sdk/bin:$PATH

USER root

ADD midi_ml ${HOME}/midi_ml
ADD test ${HOME}/test
ADD setup.py ${HOME}/setup.py
RUN mkdir ${HOME}/local_data_loc

RUN python3 setup.py install \
    && rm setup.py \
    && rm -r build

ENV PATH $PATH:${HOME}/google-cloud-sdk/bin

ADD scripts/run_pipeline /usr/local/bin/
ADD scripts/lookup_value_from_json /usr/local/bin

RUN python3 -m unittest discover -v \
    && rm -r test

RUN chmod +x /usr/local/bin/run_pipeline

# Cache data that we will re-use
ARG data_folder="data/"
ADD $data_folder ${HOME}/data_in
RUN mkdir ${HOME}/data_out
ENV DATA_IN_LOC ${HOME}/data_in/
ENV DATA_OUT_LOC ${HOME}/data_out/

CMD ["run_pipeline"]
