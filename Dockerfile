FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
	&& apt-get -y install python3.8 \
	&& apt-get -y install python3-pip \
	&& apt-get -y install libgl1-mesa-glx \
	&& apt-get -y install libglib2.0-0

WORKDIR /app

RUN pip3 install ipython==7.30.1

RUN pip3 install matplotlib==3.5.1

RUN pip3 install numpy==1.21.4 \
	pandas==1.4.2 \
	regex==2021.11.10

RUN pip3 install pillow==8.4.0 \
	scikit-image==0.19.0 \
	segmentation-models-pytorch==0.2.1

RUN pip3 install torch==1.10.0 \
	torchvision==0.11.1 \
	tqdm==4.62.3

RUN pip3 install opencv-python==4.5.5.64 \
	pathlib==1.0.1 \
	tk==0.1.0 \
	seaborn==0.11.2 \
	sklearn==0.0 \
	gimpformats==2022.0.1

CMD ["/bin/bash"]