#!/bin/bash

docker run -it --rm \
    -p 1935:1935 \
    -p 8080:80 \
    --name local-rtmp \
        lfg/nginx-rtmp

