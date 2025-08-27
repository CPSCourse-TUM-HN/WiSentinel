FROM ubuntu:16.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && \
    apt-get install -y \
    bash make gcc g++ build-essential \
    libncurses5-dev zlib1g-dev \
    libtool autoconf automake cmake \
    gawk git bison flex wget curl patch \
    unzip bzip2 rsync file python python-minimal \
    time pkg-config libssl-dev \
    nano

# Add non-root user
RUN adduser --disabled-password --gecos "" builder
USER builder
WORKDIR /home/builder

# Clone CSI-patched OpenWRT
RUN git clone https://github.com/xieyaxiongfly/Atheros_CSI_tool_OpenWRT_src.git openwrt
WORKDIR /home/builder/openwrt

