#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) in vec3 worldpos[3];   // Input world position from vertex shader
layout(location = 0) out vec3 worldPos;

struct VisibilityData {
    vec3 v1;
    vec3 v2;
    vec3 v3;
};

layout(binding = 1, set = 0) buffer visibilityLUT
{
    VisibilityData visibilitydata[];
};

void main()
{
    int primitiveID = gl_PrimitiveIDIn;
    vec3 v1 = worldpos[0];
    //vec3 v1 = gl_in[0].gl_Position.xyz;
    vec3 v2 = worldpos[1];
    //vec3 v2 = gl_in[1].gl_Position.xyz;
    vec3 v3 = worldpos[2];
    //vec3 v3 = gl_in[2].gl_Position.xyz;

    gl_PrimitiveID = gl_PrimitiveIDIn;

    //the lut starts from 1 to avoid geting incorrect vertices values for the background
    visibilitydata[primitiveID+1].v1 = v1;
    visibilitydata[primitiveID+1].v2 = v2;
    visibilitydata[primitiveID+1].v3 = v3;

    worldPos = worldpos[0];
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    worldPos = worldpos[1];
    gl_Position = gl_in[1].gl_Position;
    EmitVertex();
    worldPos = worldpos[2];
    gl_Position = gl_in[2].gl_Position;
    EmitVertex();
    EndPrimitive();
}