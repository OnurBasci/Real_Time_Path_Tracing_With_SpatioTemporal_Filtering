// Copyright 2020 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_query : require
#extension GL_GOOGLE_include_directive : require
#include "../common.h"

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

//we define a simple sphere light source
struct sphereLight {
    vec3 center;
    float radius;
    vec3 color;
};

struct VisibilityData {
    vec3 v1;
    vec3 v2;
    vec3 v3;
};

layout(local_size_x = WORKGROUP_WIDTH, local_size_y = WORKGROUP_HEIGHT, local_size_z = 1) in;

// Binding BINDING_IMAGEDATA in set 0 is a storage image with four 32-bit floating-point channels,
// defined using a uniform image2D variable.
layout(binding = BINDING_IMAGEDATA, set = 0, rgba32f) uniform image2D storageImage;
layout(binding = BINDING_TLAS, set = 0) uniform accelerationStructureEXT tlas;
// The scalar layout qualifier here means to align types according to the alignment
// of their scalar components, instead of e.g. padding them to std140 rules.
layout(binding = BINDING_VERTICES, set = 0, scalar) buffer Vertices
{
    vec3 vertices[];
};
layout(binding = BINDING_INDICES, set = 0, scalar) buffer Indices
{
    uint indices[];
};
layout(binding = BINDING_PREVIOUS_IMAGE_DATA, set = 0, rgba32f) uniform image2D previousFrameImage;
layout(binding = 5, set = 0, rgba32f) uniform image2D temporalGradient;

//these are used for SVGF
layout(binding = 6, set = 0, rgba32f) uniform image2D depthBuffer;

layout(binding = 7, set = 0) buffer visibilityLUT
{
    VisibilityData visibilitylut[];
};

layout(push_constant) uniform PushConsts
{
  PushConstants pushConstants;
};

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float stepAndOutputRNGFloat(inout uint rngState)
{
    // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
    rngState = rngState * 747796405 + 1;
    uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
    word = (word >> 22) ^ word;
    return float(word) / 4294967295.0f;
}

const float k_pi = 3.14159265;

// Uses the Box-Muller transform to return a normally distributed (centered
// at 0, standard deviation 1) 2D point.
vec2 randomGaussian(inout uint rngState)
{
    // Almost uniform in (0, 1] - make sure the value is never 0:
    const float u1 = max(1e-38, stepAndOutputRNGFloat(rngState));
    const float u2 = stepAndOutputRNGFloat(rngState);  // In [0, 1]
    const float r = sqrt(-2.0 * log(u1));
    const float theta = 2 * k_pi * u2;  // Random in [0, 2pi]
    return r * vec2(cos(theta), sin(theta));
}

// Returns the color of the sky in a given direction (in linear color space)
vec3 skyColor(vec3 direction)
{
    //return vec3(0.0);
    // +y in world space is up, so:
    if (direction.y > 0.0f)
    {
        return mix(vec3(1.0f), vec3(0.25f, 0.5f, 1.0f), direction.y);
    }
    else
    {
        return vec3(0.03f);
    }
}

struct HitInfo
{
    vec3 color;
    vec3 worldPosition;
    vec3 worldNormal;
};

HitInfo getObjectHitInfo(rayQueryEXT rayQuery)
{
    HitInfo result;
    // Get the ID of the triangle
    const int primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);

    // Get the indices of the vertices of the triangle
    const uint i0 = indices[3 * primitiveID + 0];
    const uint i1 = indices[3 * primitiveID + 1];
    const uint i2 = indices[3 * primitiveID + 2];

    // Get the vertices of the triangle
    const vec3 v0 = vertices[i0];
    const vec3 v1 = vertices[i1];
    const vec3 v2 = vertices[i2];

    // Get the barycentric coordinates of the intersection
    vec3 barycentrics = vec3(0.0, rayQueryGetIntersectionBarycentricsEXT(rayQuery, true));
    barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

    // Compute the coordinates of the intersection
    const vec3 objectPos = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
    // For the main tutorial, object space is the same as world space:
    result.worldPosition = objectPos;

    // Compute the normal of the triangle in object space, using the right-hand rule:
    //    v2      .
    //    |\      .
    //    | \     .
    //    |/ \    .
    //    /   \   .
    //   /|    \  .
    //  L v0---v1 .
    // n
    const vec3 objectNormal = normalize(cross(v1 - v0, v2 - v0));
    // For the main tutorial, object space is the same as world space:
    result.worldNormal = objectNormal;

    //set some color
    if (dot(objectNormal, vec3(1.0, 0.0, 0.0)) > 0.99) {
        result.color = vec3(1.0, 0.0, 0.0);
    }
    else if (dot(objectNormal, vec3(-1.0, 0.0, 0.0)) > 0.99) {
        result.color = vec3(0.0, 1.0, 0.0);
    }
    else {
        result.color = vec3(0.7f);
    }

    return result;
}

bool checkRayLightIntersection(vec3 rayOrigin, vec3 rayDir, sphereLight light, out float t) {
    //returns true if the ray intersect with the light source
    vec3 oc = rayOrigin - light.center; // Vector from ray origin to sphere center
    float a = dot(rayDir, rayDir); // Always 1 if rayDir is normalized
    float b = 2.0 * dot(oc, rayDir);
    float c = dot(oc, oc) - (light.radius * light.radius);

    float discriminant = (b * b) - (4.0 * a * c);

    if (discriminant < 0.0) {
        return false; // No intersection
    }

    // Compute the nearest intersection t (smallest positive solution)
    float sqrtD = sqrt(discriminant);
    float t1 = (-b - sqrtD) / (2.0 * a);
    float t2 = (-b + sqrtD) / (2.0 * a);

    // Ensure we return the closest positive intersection
    if (t1 > 0.0) {
        t = t1;
    }
    else if (t2 > 0.0) {
        t = t2;
    }
    else {
        return false; // Both intersections are behind the ray
    }

    return true;
}

vec3 computePathTracedColorFromPixel(vec3 rayOrigin, vec3 rayDirection, sphereLight light, float alpha, uint rngState) {
    vec3 accumulatedRayColor = vec3(1.0);  // The amount of light that made it to the end of the current ray.

    // Limit the kernel to trace at most 32 segments.
    for (int tracedSegments = 0; tracedSegments < 32; tracedSegments++)
    {
        // Trace the ray and see if and where it intersects the scene!
        // First, initialize a ray query object:
        rayQueryEXT rayQuery;
        rayQueryInitializeEXT(rayQuery,              // Ray query
            tlas,                  // Top-level acceleration structure
            gl_RayFlagsOpaqueEXT,  // Ray flags, here saying "treat all geometry as opaque"
            0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
            rayOrigin,             // Ray origin
            0.0,                   // Minimum t-value
            rayDirection,          // Ray direction
            10000.0);              // Maximum t-value

        // Start traversal, and loop over all ray-scene intersections. When this finishes,
        // rayQuery stores a "committed" intersection, the closest intersection (if any).
        while (rayQueryProceedEXT(rayQuery))
        {
        }

        //if intersected with the light source on the first ray cast return lightcolor
        float t;
        if (checkRayLightIntersection(rayOrigin, rayDirection, light, t)) {
            //set the alpha to 0 to always get the current light color for the light pixel
            if (tracedSegments == 0) {
                accumulatedRayColor *= light.color / 5.0; //decrease for eye safety :)
                alpha = 0.0;
                break;
            }
            accumulatedRayColor *= light.color; //decrease for eye safety :)
            break;
        }

        // Get the type of committed (true) intersection - nothing, a triangle, a generated object
        if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
        {
            // Ray hit a triangle
            HitInfo hitInfo = getObjectHitInfo(rayQuery);

            // Apply color absorption
            accumulatedRayColor *= hitInfo.color;

            // Flip the normal so it points against the ray direction:
            hitInfo.worldNormal = faceforward(hitInfo.worldNormal, rayDirection, hitInfo.worldNormal);

            // Start a new ray at the hit position, but offset it slightly along the normal:
            rayOrigin = hitInfo.worldPosition + 0.0001 * hitInfo.worldNormal;

            // For a random diffuse bounce direction, we follow the approach of
            // Ray Tracing in One Weekend, and generate a random point on a sphere
            // of radius 1 centered at the normal. This uses the random_unit_vector
            // function from chapter 8.5:
            const float theta = 2.0 * k_pi * stepAndOutputRNGFloat(rngState);  // Random in [0, 2pi]
            const float u = 2.0 * stepAndOutputRNGFloat(rngState) - 1.0;   // Random in [-1, 1]
            const float r = sqrt(1.0 - u * u);
            rayDirection = hitInfo.worldNormal + vec3(r * cos(theta), r * sin(theta), u);
            // Then normalize the ray direction:
            rayDirection = normalize(rayDirection);
        }
        else
        {
            // Ray hit the sky
            accumulatedRayColor *= skyColor(rayDirection);
            break;
        }
    }
    return accumulatedRayColor;
}

void main()
{
    //float alpha = float(pushConstants.frameNumber) / float(pushConstants.frameNumber + 1);
    float alpha = 0.0; // 0 means current frame 1 means previous

    sphereLight light;
    light.center = pushConstants.lightPos;
    light.radius = 0.20;
    light.color = pushConstants.currentCameraColor * 30; //to hdr
    //light.color = vec3(1.0,1.0,1.0) * 30; //to hdr
    const vec3 cameraOrigin = pushConstants.cameraPos.rgb;

    const ivec2 resolution = imageSize(storageImage);
    const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

    // If the pixel is outside of the image, don't do anything:
    if ((pixel.x >= resolution.x) || (pixel.y >= resolution.y))
    {
        return;
    }

    alpha = clamp(alpha, 0, 1);

    // State of the random number generator with an initial seed.
    uint rngState = uint(pixel.x * 3266489917U + pixel.y * 668265263U) ^ uint(pushConstants.frameNumber * 374761393U) ^ uint(pushConstants.sample_batch * 2654435761U);

    // Define the field of view by the vertical slope of the topmost rays:
    const float fovVerticalSlope = tan(FOV);

    // The sum of the colors of all of the samples.
    vec3 summedPixelColor = vec3(0.0);

    //samples per pixel
    const int NUM_SAMPLES = 1;
    for (int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++)
    {
        vec3 rayOrigin = cameraOrigin;
        // Compute the direction of the ray for this pixel. To do this, we first
        // transform the screen coordinates to look like this, where a is the
        // aspect ratio (width/height) of the screen:

        const vec2 randomPixelCenter = vec2(pixel) + vec2(0.5) + 0.375 * randomGaussian(rngState);
        const vec2 screenUV = vec2((2.0 * randomPixelCenter.x - resolution.x) / resolution.y,    //
            -(2.0 * randomPixelCenter.y - resolution.y) / resolution.y);  // Flip the y axis

        // Create a ray direction:
        vec3 rayDirection = vec3(fovVerticalSlope * screenUV.x, fovVerticalSlope * screenUV.y, -1.0);
        rayDirection = normalize(rayDirection);

        //PATH TRACING PER PIXEL
        vec3 rayColor = computePathTracedColorFromPixel(rayOrigin, rayDirection, light, alpha, rngState);

        summedPixelColor += rayColor;
    }

    vec3 averagePixelColor = summedPixelColor / float(NUM_SAMPLES);

    //if the temporal gradient is high then alpha is high

    //alpha blending with the previous frame
    /*if (pushConstants.frameNumber > 0) {
        //Read the previous frame image
        const vec3 preiousFrameColor = imageLoad(previousFrameImage, pixel).rgb;
        averagePixelColor = averagePixelColor * (1.0 - alpha) + preiousFrameColor * alpha;
    }*/
    
    vec3 outputColor = averagePixelColor;
    //outputColor = vec3(alpha, alpha, alpha);

    // Set the color of the pixel `pixel` in the storage image to `averagePixelColor`:
    imageStore(storageImage, pixel, vec4(outputColor, 0.0));
}



/*if (pushConstants.sample_batch != 0)
{
    // Read the storage image:
    const vec3 previousAverageColor = imageLoad(storageImage, pixel).rgb;

    // Compute the new average:
    averagePixelColor =
        (pushConstants.sample_batch * previousAverageColor + averagePixelColor) / (pushConstants.sample_batch + 1);
}*/