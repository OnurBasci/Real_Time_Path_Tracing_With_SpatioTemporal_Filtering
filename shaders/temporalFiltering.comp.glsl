#version 460
#extension GL_GOOGLE_include_directive : require
#include "../common.h"

struct VisibilityData {
    vec3 v1;
    vec3 v2;
    vec3 v3;
};

struct PushConstants
{
    uint sample_batch;
    uint frameNumber;

    vec3 cameraPos;
    vec3 lightPos;
    vec3 lightPosPrev;

    vec3 currentCameraColor;
    vec3 previousCameraColor;

    int waveletIteration;
    int maxWaveletIteration;
};

layout(local_size_x = WORKGROUP_WIDTH, local_size_y = WORKGROUP_HEIGHT, local_size_z = 1) in;

layout(binding = 0, set = 0, rgba32f) uniform image2D storageImage; //output image
layout(binding = 1, set = 0, rgba32f) uniform image2D colorImage; //output image
layout(binding = 2, set = 0, rgba32f) uniform image2D depthImage; //output image
layout(binding = 3, set = 0) buffer visibilityLUT
{
    VisibilityData visibilitylut[];
};
layout(binding = 4, set = 0, r16f) uniform image2D visibilityBuffer;
layout(binding = 5, set = 0, rgba32f) uniform image2D previousFrameImage;
layout(binding = 6, set = 0) buffer visibilityLUTPrev
{
    VisibilityData visibilitylutPrev[];
};
layout(binding = 7, set = 0, rgba32f) uniform image2D worldPosImage;
layout(binding = 8) uniform UniformBufferObject{
    mat4 model;
    mat4 view;
    mat4 proj;

    mat4 modelPrev;
    mat4 viewPrev;
    mat4 projPrev;
} ubo;

layout(binding = 9, set = 0, rgba32f) uniform image2D temporalGradient;

layout(push_constant) uniform PushConsts
{
  PushConstants pushConstants;
};

// Compute normal-based weight
float computeNormalWeight(vec3 normalP, vec3 normalQ, float s_n) {
    return pow(max(0.0, dot(normalP, normalQ)), s_n);
}

// Compute depth-based weight
float computeDepthWeight(float depthP, float depthQ, float s_z) {
    float depthDifference = length(depthP - depthQ);
    return exp(-depthDifference / (s_z));
}

// Compute luminance-based weight
float computeLuminanceWeight(vec3 lumP, vec3 lumQ, float s_l) {
    return exp(-length(lumP - lumQ) / s_l);
}

float weightFunction(vec3 np, vec3 nq, float dp, float dq, vec3 lp, vec3 lq, float s_n, float s_z, float s_l) {
    return computeNormalWeight(np, nq, s_n) * computeDepthWeight(dp, dq, s_z) * computeLuminanceWeight(lp, lq, s_l);
}

vec3 getNormalFromTriangleIndex(float primitiveID) {

    //set the normal to an arbitrary vector if ID = 0 meaning that it is the background
    if (primitiveID == 0) return vec3(0.0, 0.0, 1.0);
    VisibilityData triangleVertices = visibilitylut[int(primitiveID)];

    vec3 v1 = triangleVertices.v1;
    vec3 v2 = triangleVertices.v2;
    vec3 v3 = triangleVertices.v3;

    return normalize(cross(v2 - v1, v3 - v1));
}

const float gaussianKernel2D[5][5] = float[5][5](
    float[5](1, 4.0, 7.0, 4.0, 1.0),
    float[5](4.0, 16.0, 26.0, 16.0, 4.0),
    float[5](7.0, 26.0, 41.0, 26.0, 7.0),
    float[5](4.0, 16.0, 26.0, 16.0, 4.0),
    float[5](1.0, 4.0 , 7.0 ,  4.0, 1.0)
    );

const float boxKernel2D[7][7] = float[7][7](
    float[7](1,1,1,1,1,1,1),    
    float[7](1,1,1,1,1,1,1),    
    float[7](1,1,1,1,1,1,1),  
    float[7](1,1,1,1,1,1,1),    
    float[7](1,1,1,1,1,1,1),
    float[7](1,1,1,1,1,1,1),
    float[7](1,1,1,1,1,1,1)
    );

const float boxKernel[3][3] = float[3][3](
    float[3](1, 1, 1),
    float[3](1, 1, 1),
    float[3](1, 1, 1)
    );

//we need 2 functions for wavelet transform since we use ping pong buffers
vec3 waveletTransformOddIteration(ivec2 pixel, uint waveletIteration, float s_n, float s_z, float s_l) {
    uint k = waveletIteration;

    //get normal, depth and color
    vec3 cp = imageLoad(colorImage, pixel).rgb;
    float dp = imageLoad(depthImage, pixel).r;

    float primitiveID = imageLoad(visibilityBuffer, pixel).r;

    vec3 np = getNormalFromTriangleIndex(primitiveID);

    //wavelet transform
    vec3 numerator = vec3(0.0, 0.0, 0.0);
    vec3 denominator = vec3(0.0, 0.0, 0.0);
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            //clamp neighbor pixel
            ivec2 pixelq = ivec2(pixel.x + i * k, pixel.y + j * k);
            pixelq = clamp(pixelq, ivec2(0), imageSize(colorImage) - 1);

            vec3 cq = imageLoad(colorImage, pixelq).rgb;

            float dq = imageLoad(depthImage, pixelq).r;
            float primitiveIDq = imageLoad(visibilityBuffer, pixelq).r;
            vec3 nq = getNormalFromTriangleIndex(primitiveIDq);

            float w = weightFunction(np, nq, dp, dq, cp, cq, s_n, s_z, s_l);
            float h = boxKernel[i + 1][j + 1] * (1 / 9.0);
            numerator += h * w * cq;
            denominator += h * w;
        }
    }
    vec3 filteredColor = numerator / denominator;

    imageStore(storageImage, pixel, vec4(filteredColor, 0.0));

    return filteredColor;
}

float getAreaOfTriangle(vec3 v0, vec3 v1, vec3 v2) {
    vec3 dir1 = v1 - v0;
    vec3 dir2 = v2 - v0;

    return length(cross(dir1, dir2)) * 0.5;
}

vec3 getBarycentricCoordinates(vec3 p, vec3 v0, vec3 v1, vec3 v2) {
    //this function computes barycentric coordinates from a point on the triangle surface and trinangle vertices
    // Areas
    float areaTotal = getAreaOfTriangle(v0, v1, v2);
    float area1 = getAreaOfTriangle(p, v1, v2);
    float area2 = getAreaOfTriangle(v0, p, v2);
    float area3 = getAreaOfTriangle(v0, v1, p);

    // Barycentric coordinates
    vec3 barCoord = vec3(area1 / areaTotal, area2 / areaTotal, area3 / areaTotal);

    return barCoord;
}

vec2 worldToPixel(vec3 worldPos, mat4 viewMatrix, mat4 projMatrix, ivec2 resolution) {
    // Transform world position to clip space
    vec4 clipPos = projMatrix * viewMatrix * vec4(worldPos, 1.0);

    // Convert to Normalized Device Coordinates (NDC)
    vec3 ndcPos = clipPos.xyz / clipPos.w;

    // Convert to screen space coordinates
    vec2 screenPos = (ndcPos.xy * 0.5 + 0.5) * resolution;

    return screenPos;
}

void main() {

    const ivec2 resolution = imageSize(storageImage);
    const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

    // If the pixel is outside of the image, don't do anything:
    if ((pixel.x >= resolution.x) || (pixel.y >= resolution.y))
    {
        return;
    }

    //apply filtering
    float sigma_n = 128.0;
    float sigma_z = 1.0;
    float sigma_l = 4.0;

    //ping pong algorithm for wavelet transform
    int maxIt = pushConstants.maxWaveletIteration;
    int k = pushConstants.waveletIteration;
    vec3 filteredColor = waveletTransformOddIteration(pixel, k, sigma_n, sigma_z, sigma_l);

    //back propagation for exponential moving average
    vec3 worldPos = imageLoad(worldPosImage, pixel).rgb;
    float primitiveID = imageLoad(visibilityBuffer, pixel).r;
    //If background set the previous pixel to current
    ivec2 previousPixelPos;
    if (primitiveID < 1) {
        previousPixelPos = pixel;
    }
    else {
        VisibilityData currentTriangleVertices = visibilitylut[int(primitiveID)];

        VisibilityData previousTriangleVertices = visibilitylutPrev[int(primitiveID)];

        vec3 v1 = previousTriangleVertices.v1;
        vec3 v2 = previousTriangleVertices.v2;
        vec3 v3 = previousTriangleVertices.v3;

        vec3 barCoord = getBarycentricCoordinates(worldPos, v1, v2, v3);

        vec3 v1p = previousTriangleVertices.v1;
        vec3 v2p = previousTriangleVertices.v2;
        vec3 v3p = previousTriangleVertices.v3;

        //we deduce previous frame world position from the barycentering coordinates
        vec3 worldPosPrevious = barCoord.x * v1p + barCoord.y * v2p + barCoord.z * v3p;

        previousPixelPos = ivec2(worldToPixel(worldPosPrevious, ubo.viewPrev, ubo.projPrev, resolution));
    }

    //alpha blending between path traced image and filtered image if it is the last iteration
    if (k == maxIt) {
        float alpha = 0.3; //1 means current 0 means previous 
        vec3 cp = imageLoad(colorImage, pixel).rgb;
        
        //adaptive alpha selection, a value between constant alpha and 1
        //float tempGrad = imageLoad(temporalGradient, pixel).r;
        //alpha = (1 - tempGrad) * alpha + tempGrad;

        vec3 alphaBlendColor;
        if (pushConstants.frameNumber > 0) {
            //Read the previous frame image
            const vec3 preiousFrameColor = imageLoad(previousFrameImage, previousPixelPos).rgb;
            alphaBlendColor = preiousFrameColor * (1.0 - alpha) + filteredColor * alpha;
        }
        else {
            //set the output color to the filtered
            alphaBlendColor = filteredColor;
        }


        // Set the blended color to the path traced image (to be shown in the swap chain)
        imageStore(colorImage, pixel, vec4(alphaBlendColor, 0.0));
    }
}