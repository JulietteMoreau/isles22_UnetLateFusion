#!/usr/bin/env bash

./build.sh

docker save unetlatefusion_cl | gzip -c > UNetLateFusion_CL.tar.gz
