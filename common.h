// Copyright 2020 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Common file shared across C++ CPU code and GLSL GPU code.
#ifndef VK_MINI_PATH_TRACER_COMMON_H
#define VK_MINI_PATH_TRACER_COMMON_H

#ifdef __cplusplus
#include <cstdint>
using uint = uint32_t;
#endif  // #ifdef __cplusplus


#define FOV 0.20 //in radian

#define WORKGROUP_WIDTH 16
#define WORKGROUP_HEIGHT 8

#define BINDING_IMAGEDATA 0
#define BINDING_TLAS 1
#define BINDING_VERTICES 2
#define BINDING_INDICES 3
#define BINDING_PREVIOUS_IMAGE_DATA 4

#endif // #ifndef VK_MINI_PATH_TRACER_COMMON_H