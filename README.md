# Real_Time_Path_Tracing_With_SpatioTemporal_Filtering

## About Project
1 Introduction
 Path tracing is a physically-based rendering technique that simulates the way light interacts
 with surfaces to produce highly realistic images. By tracing the paths of many light rays as
 they bounce around a scene, path tracing accurately models global illumination, soft shadows,
 caustics, and other complex lighting effects. This method is widely used in offline rendering for
 film and visual effects due to its ability to generate photorealistic results.
 However, implementing path tracing in real-time rendering presents significant challenges.
 The primary difficulty arises from the immense computational power required to simulate the
 vast number of light interactions needed for high-quality images. Unlike traditional rasteri
zation or even ray tracing techniques, which approximate lighting for efficiency, path tracing
 requires a large number of samples per pixel to reduce noise and achieve convergence. Real-time
 applications, such as video games, have strict performance constraints, often targeting 30 to 60
 frames per second, making the brute-force approach of path tracing impractical.
 To overcome these limitations, advancements in hardware (such as dedicated ray-tracing
 cores in modern GPUs), denoising techniques, and clever sampling methods have been de
veloped. Despite these improvements, real-time path tracing remains an ongoing challenge,
 balancing visual fidelity and performance to make physically accurate rendering feasible for
 interactive applications. In this project, I focused on one of the denoising technique called
 adaptive spatiotemporal variance-guided filtering (A-SVGF) to implement a denoised real time
 path tracer.


## Configuration
The configuration is tested on Windows. It is not guaranteed to work on other OS.

You have to first download this project with command 
```
git clone https://github.com/OnurBasci/Real_Time_Path_Tracing_With_SpatioTemporal_Filtering.git
```
This project uses NVIDIA's nvpro_core library (see https://github.com/nvpro-samples/nvpro_core)
To install it go to the projects folder and install it with the command
```
cd .\Real_Time_Path_Tracing_With_SpatioTemporal_Filtering\
git clone https://github.com/nvpro-samples/nvpro_core.git
```
Finaly you can generate the project with cmake.



