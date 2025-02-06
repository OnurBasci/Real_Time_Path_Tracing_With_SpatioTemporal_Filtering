#version 450

layout(binding = 0) uniform UniformBufferObject{
    mat4 model;
    mat4 view;
    mat4 proj;

    mat4 modelPrev;
    mat4 viewPrev;
    mat4 projPrev;
} ubo;

layout(location = 0) in vec3 inPosition;  // Input vertex position
layout(location = 0) out vec3 worldPos;  // Output color to fragment shader

// Simple random function based on input value
float random(float x) {
    return fract(sin(x) * 43758.5453);
}

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

    worldPos = (ubo.model * vec4(inPosition, 1.0)).rgb;
}