#pragma once
#include <c10/cuda/CUDAStream.h>

#define MY_STREAM c10::cuda::getCurrentCUDAStream().stream()
