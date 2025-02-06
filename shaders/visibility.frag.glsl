#version 450

layout(location = 0) in vec3 worldPos;   // Input world position from vertex shader
layout(location = 0) out float outColor;  // Output color to the framebuffer
layout(location = 1) out vec3 worldPosition;

float random(float x) {
    return fract(sin(x) * 43758.5453);
}

void main() {
    // Generate random colors based on triangleId
    float r = random(float(gl_PrimitiveID));  // Random red component
    float g = random(float(gl_PrimitiveID + 1));  // Random green component
    float b = random(float(gl_PrimitiveID + 2));  // Random blue component

    // Create a color vector
    vec3 randomColor = vec3(r, g, b); //for color based representation

    worldPosition = worldPos;

    // set the final output as the index of the triangle
    outColor = gl_PrimitiveID+1;
    //outColor = vec4(gl_PrimitiveID/31.0, gl_PrimitiveID/31.0, gl_PrimitiveID/31.0, 1.0);  // Set random color with full opacity
}