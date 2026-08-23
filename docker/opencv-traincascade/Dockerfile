FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y -qq \
    build-essential cmake git \
    libjpeg-dev libpng-dev libtiff-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --branch 3.4 --depth 1 https://github.com/opencv/opencv.git

RUN mkdir -p /build/opencv/build
WORKDIR /build/opencv/build
RUN cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_LIST=core,imgproc,imgcodecs,highgui,objdetect,ml,calib3d,features2d,flann,photo,video,videoio \
    -DBUILD_opencv_apps=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=OFF \
    -DWITH_FFMPEG=OFF \
    -DWITH_GTK=OFF \
    -DWITH_QT=OFF \
    -DWITH_1394=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_OPENCL=OFF \
    ..

RUN make -j"$(nproc)" opencv_traincascade opencv_createsamples opencv_annotation

RUN mkdir -p /out && \
    cp bin/opencv_traincascade bin/opencv_createsamples bin/opencv_annotation /out/ && \
    cp -P lib/*.so* /out/ 2>/dev/null || true

WORKDIR /wally
