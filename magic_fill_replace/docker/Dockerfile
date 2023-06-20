ARG BASE_IMAGE

FROM $BASE_IMAGE

#Install any additional libraries
RUN echo "Adding conda package to Docker image"
RUN mkdir -p /home/condpackenv/
RUN mkdir -p /home/models/

# Copy conda env
COPY sam_env.tar.gz /home/condpackenv/sam_env.tar.gz
COPY stablediff_env.tar.gz /home/condpackenv/stablediff_env.tar.gz

COPY sam_vit_h_4b8939.pth.tar.gz /temp/
COPY stable_diff_inpaint.tar.gz /temp/
# Install tar
RUN apt-get update && apt-get install -y tar
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Untar the file
RUN tar -xzf /temp/sam_vit_h_4b8939.pth.tar.gz -C /home/models/
RUN tar -xzf /temp/stable_diff_inpaint.tar.gz -C /home/models/

RUN rm /temp/sam_vit_h_4b8939.pth.tar.gz
RUN rm /temp/stable_diff_inpaint.tar.gz