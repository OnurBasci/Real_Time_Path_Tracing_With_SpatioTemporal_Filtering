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

layout(push_constant) uniform PushConsts
{
  PushConstants pushConstants;
};

layout(binding = 0, set = 0, r16f) uniform image2D visibilityBuffer;

layout(binding = 1, set = 0) buffer visibilityLUT
{
    VisibilityData visibilitylut[];
};

layout(binding = 2, set = 0) buffer visibilityLUTPrev
{
    VisibilityData visibilitylutPrev[];
};

layout(binding = 3, set = 0, rgba32f) uniform image2D storageImage;

layout(binding = 4, set = 0, rgba32f) uniform image2D worldPosImage;

float getAreaOfTriangle(vec3 v0, vec3 v1, vec3 v2) {
    vec3 dir1 = v1 - v0;
    vec3 dir2 = v2 - v0;

    return length(cross(dir1, dir2)) *0.5;
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

vec3 phongShading(vec3 p, vec3 n, vec3 camPos, vec3 lightPos, vec3 lightColor) {
    
    //vec3 objectColor = vec3(192.0/255.0, 23.0/255.0, 23.0/255.0);
    vec3 objectColor = vec3(0.7, 0.7, 0.7);

    vec3 lightDir = normalize(lightPos - p);

    //ambient color
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    //diffuse part
    float diff;

    diff = max(dot(n, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    //specular part
    float specularStrength = 0.5;
    vec3 viewDir = normalize(camPos - p);
    vec3 reflectDir = reflect(-lightDir, n);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 128);
    vec3 specular = specularStrength * spec * lightColor;

    //attenuation
    float distance = length(lightPos - p);
    float attenuation = 1.0;

    vec3 result = (ambient + diffuse + specular) * attenuation * objectColor;
    return result;
}


void main() {
    
    const vec3 cameraOrigin = pushConstants.cameraPos;

    const vec3 lightPos = pushConstants.lightPos;
    const vec3 lightPosPrev = pushConstants.lightPosPrev;
    const vec3 lightColor = pushConstants.currentCameraColor;
    const vec3 previousLightColor = pushConstants.previousCameraColor;

    // The resolution of the image:
    const ivec2 resolution = imageSize(storageImage);

    const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

    //delete the pixel value from the previous frame
    imageStore(storageImage, pixel, vec4(0.0,0.0,0.0, 0.0));

    // If the pixel is outside of the image, don't do anything:
    if ((pixel.x >= resolution.x) || (pixel.y >= resolution.y))
    {
        return;
    }

    //get triangle id
    float primitiveID = imageLoad(visibilityBuffer, pixel).r;

    //return if in the corresponding pixel there is no triangle
    if (primitiveID == 0) return;

    //get current frame data
    vec3 worldPos = imageLoad(worldPosImage, pixel).rgb;

    VisibilityData triangleVertices = visibilitylut[int(primitiveID)];

    vec3 v1 = triangleVertices.v1;
    vec3 v2 = triangleVertices.v2;
    vec3 v3 = triangleVertices.v3;

    vec3 normal = normalize(cross(v2 - v1, v3 - v1));
    vec3 barCoord = getBarycentricCoordinates(worldPos, v1, v2, v3);
    
    //get previous frame data
    VisibilityData previousTriangleVertices = visibilitylutPrev[int(primitiveID)];

    vec3 v1p = previousTriangleVertices.v1;
    vec3 v2p = previousTriangleVertices.v2;
    vec3 v3p = previousTriangleVertices.v3;

    //we deduce previous frame world position from the barycentering coordinates
    vec3 worldPosPrevious = barCoord.x * v1p + barCoord.y * v2p + barCoord.z * v3p;

    vec3 normalPrev = normalize(cross(v2p - v1p, v3p - v1p));

    //light calculation of the firt frame
    vec3 currentframeColor = phongShading(worldPos, normal, cameraOrigin, lightPos, lightColor);

    //lightning calculation of the previous frame
    vec3 previousFrameColor = phongShading(worldPosPrevious, normal, cameraOrigin, lightPosPrev, previousLightColor);

    vec3 temporalGradient = (currentframeColor - previousFrameColor);

    //compute the ralative change
    float delta = max(length(currentframeColor), length(previousFrameColor));
    float lamda = min(1, length(temporalGradient) / delta);

    //write the temporal gradient
    imageStore(storageImage, pixel, vec4(vec3(lamda), 0.0));

}