RUN pip3 install --upgrade pip
RUN pip3 install requests
RUN pip3 --default-timeout=1000 install azure-storage-blob==1.0.0
RUN pip3 --default-timeout=1000 install azure-storage-queue==1.0.0
RUN apt-get update 
RUN apt-get install -y libgstreamer1.0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools
RUN pip3 install opencv-contrib-python-headless

RUN chmod -R 777 /var/azureml-app/azureml-models
