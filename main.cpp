// Copyright 2020-2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <nvh/fileoperations.hpp>  // For nvh::loadFile
#include <nvvk/swapchain_vk.hpp>
#include"context.hpp"
#include <nvvk/descriptorsets_vk.hpp>  // For nvvk::DescriptorSetContainer
#include <nvvk/error_vk.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvk/renderpasses_vk.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <nvvk/raytraceKHR_vk.hpp>        // For nvvk::RaytracingBuilderKHR
#include <nvvk/resourceallocator_vk.hpp>  // For NVVK memory allocators
#include <nvvk/shaders_vk.hpp>            // For nvvk::createShaderModule

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <math.h>

#include <unordered_map>

#include <iostream>
#include "common.h"
#include <windows.h>

struct PushConstants
{
    uint sample_batch;
    uint frameNumber;

    alignas(16) glm::vec3 cameraPos;
    alignas(16) glm::vec3 lightPos;
    alignas(16) glm::vec3 lightPosPrev;

    alignas(16) glm::vec3 currentCameraColor;
    alignas(16) glm::vec3 previousCameraColor;

    int waveletIteration;
    int maxWaveletIteration;
};

PushConstants  pushConstants;
const uint32_t render_width = 1000;
const uint32_t render_height = 800;

const int maxWaveletIteration = 9; //must be an odd number

struct Vertex {
    alignas(16) glm::vec3 pos;

    bool operator==(const Vertex& other) const {
        return pos == other.pos;
    }
};

glm::vec3 cameraOrigin(-0.001, 1.0, 6.0);
glm::vec4 cameraColorCurrent(1.0, 0.0, 0.0, 1.0);
glm::vec4 cameraColorPrevious(0.0, 1.0, 0.0, 1.0);
const float speed = 0.1;

glm::vec3 lightPos(1, 1.0, -0.4);
glm::vec3 lightPosPrev;
glm::vec3 lightColor(0.5, 0.5, 0.5);

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return (hash<glm::vec3>()(vertex.pos));
        }
    };
}

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;

    alignas(16) glm::mat4 modelPrev;
    alignas(16) glm::mat4 viewPrev;
    alignas(16) glm::mat4 projPrev;
};

VkCommandBuffer AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool)
{
    VkCommandBufferAllocateInfo cmdAllocInfo{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                             .commandPool = cmdPool,
                                             .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                             .commandBufferCount = 1 };
    VkCommandBuffer             cmdBuffer;
    NVVK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuffer));
    VkCommandBufferBeginInfo beginInfo{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    NVVK_CHECK(vkBeginCommandBuffer(cmdBuffer, &beginInfo));
    return cmdBuffer;
}

void EndSubmitWaitAndFreeCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool cmdPool, VkCommandBuffer& cmdBuffer)
{
    NVVK_CHECK(vkEndCommandBuffer(cmdBuffer));
    VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmdBuffer };
    NVVK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    NVVK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}

VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer)
{
    VkBufferDeviceAddressInfo addressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buffer };
    return vkGetBufferDeviceAddress(device, &addressInfo);
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

nvvk::Image createImage(VkDevice device, nvvk::ResourceAllocatorDedicated& allocator, uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples,
    VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = numSamples;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    return allocator.createImage(imageInfo);
}

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
}

class PathTracingApplication {
public:

    PathTracingApplication(const char** argv) {
        this->argv = argv;
    }
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
    }

private:
    GLFWwindow* window;
    VkSurfaceKHR surface;

    nvvk::Context context;
    nvvk::ResourceAllocatorDedicated allocator;
    nvvk::DebugUtil debugUtil;
    std::vector<std::string> searchPaths;
    nvvk::SwapChain swapChain;
    //2 images to swap for denoising
    nvvk::Image image; //image to render the scene
    nvvk::Image previousImage;
    VkImageView imageView;
    VkImageView previousImageView;
    std::vector<tinyobj::real_t> objVertices;
    std::vector<uint32_t> objIndices;
    VkCommandPool cmdPool;
    nvvk::Buffer vertexBuffer, indexBuffer;
    nvvk::Buffer uniformBuffer;
    void* uniformBuffersMapped;
    nvvk::RaytracingBuilderKHR raytracingBuilder;
    nvvk::DescriptorSetContainer descriptorSetContainer;
    VkShaderModule rayTraceModule;
    VkPipeline pathTracePipeline;
    //following attributes are for generating visibility buffer
    nvvk::Image visibilityBuffer;
    nvvk::Image previousVisibilityBuffer;
    nvvk::Buffer visibilityLUT; //visibilityLUT[primitiveId] = v1,v2,v3
    nvvk::Buffer visibilityLUTprevious;
    nvvk::Image depthImage;
    nvvk::Image positionBuffer; //g buffer containing world position
    VkImageView visibilityBufferView;
    VkImageView previousVisibilityBufferView;
    VkImageView depthImageView;
    VkImageView positionBufferView;
    VkFramebuffer visibilityFramebuffer;
    nvvk::DescriptorSetContainer visibilityDescriptorSetContainer;
    VkRenderPass renderPass;
    VkPipeline visibilityPipeline;
    VkPipelineLayout visibilityPipelineLayout;
    //following attributes are to generate the temporal gradient for each pixel
    nvvk::DescriptorSetContainer temporalGradientDescriptorContainer;
    VkShaderModule temporalGradientModule;
    nvvk::Image temporalGradientBuffer;
    VkImageView temporalGradientBufferView;
    VkPipeline temporalGradientPipeline;
    //followng attributes are for temporal filtering
    nvvk::Image filteredImageBuffer;
    VkImageView filteredImageBufferView;
    VkPipeline temporalFilteringPipepline;
    VkShaderModule temporalFilteringModule;
    nvvk::DescriptorSetContainer temporalFilteringDescriptorContainer;

    UniformBufferObject ubo{};

    bool cameraMoved = false;
    float time;

    //TO CHANGE
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    VkBuffer vertexBuffer2;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer2;
    VkDeviceMemory indexBufferMemory;

    const char** argv;

    int frameCount = 0;

    VkFence inFlightFence;

    bool frameBufferResized = false;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(render_width, render_height, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<PathTracingApplication*>(glfwGetWindowUserPointer(window));
        app->frameBufferResized = true;
    }

    void initVulkan() {
        initializeDevice();
        createSurface();
        createSwapChain();
        createSyncObjects();
        loadMesh(argv);
        createBuffers();
        createCommandPool();
        uploadBuffers();
        buildAccelerationStructure();
        createAndBindDescriptorSet();
        createComputePipeline();
        //these are for the visibility buffer generation
        createVertexBuffer();
        createIndexBuffer();
        initializeSceneConstants();
        createRenderPass();
        createFramebuffer();
        createGraphicsPipelines();

    }
    void mainLoop() {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawScene();
        }
        freeRessources();
    }

    void initializeDevice() {
        // Create the Vulkan context, consisting of an instance, device, physical device, and queues.
        nvvk::ContextCreateInfo deviceInfo;  // One can modify this to load different extensions or pick the Vulkan core version
        deviceInfo.apiMajor = 1;             // Specify the version of Vulkan we'll use
        deviceInfo.apiMinor = 2;
        //add instance extensions required for glfw
        deviceInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME);
        deviceInfo.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
        // Required by KHR_acceleration_structure; allows work to be offloaded onto background threads and parallelized
        deviceInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        deviceInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME); //for windows surface
        VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
        deviceInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &asFeatures);
        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
        deviceInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayQueryFeatures);

        context.init(deviceInfo);  // Initialize the context

        // Create the allocator
        debugUtil = nvvk::DebugUtil(context);
        allocator.init(context, context.m_physicalDevice);
    }

    void createSurface() {

        if (glfwCreateWindowSurface(context.m_instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }

        context.setGCTQueueWithPresent(surface);
    }

    void createSwapChain() {
        swapChain.init(context.m_device, context.m_physicalDevice, context.m_queueGCT, context.m_queueGCT.familyIndex,
            surface, VK_FORMAT_R32G32B32A32_SFLOAT);
    }

    void createSyncObjects() {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        if (vkCreateFence(context, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create fence");
        }
    }

    void createBuffers() {
        //RESSOURCES FOR PATH TRACING
        //create the image on the gpu
        image = createImage(context, allocator, render_width, render_height, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        imageView = createImageView(context, image.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        debugUtil.setObjectName(image.image, "image");

        //create previous Image
        previousImage = createImage(context, allocator, render_width, render_height, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        previousImageView = createImageView(context, previousImage.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        debugUtil.setObjectName(previousImage.image, "previous image");

        //RESOURCES FOR VISIBILITY BUFFER
        //create visibility buffer image
        visibilityBuffer = createImage(context, allocator, render_width, render_height, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        visibilityBufferView = createImageView(context, visibilityBuffer.image, VK_FORMAT_R16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        debugUtil.setObjectName(visibilityBuffer.image, "visibility buffer");

        previousVisibilityBuffer = createImage(context, allocator, render_width, render_height, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        previousVisibilityBufferView = createImageView(context, previousVisibilityBuffer.image, VK_FORMAT_R16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        debugUtil.setObjectName(previousVisibilityBuffer.image, "previous visibility");

        //create depth resources
        depthImage = createImage(context, allocator, render_width, render_height, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_D32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        depthImageView = createImageView(context, depthImage.image, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
        debugUtil.setObjectName(depthImage.image, "depth");

        //create g buffer for world pos
        positionBuffer = createImage(context, allocator, render_width, render_height, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        positionBufferView = createImageView(context, positionBuffer.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        debugUtil.setObjectName(positionBuffer.image, "position buffer");

        //create visibility lookup table
        VkDeviceSize       bufferSizeBytes = objIndices.size() * 9 * sizeof(float); //For each triangle we have 3 corresponding vertices
        VkBufferCreateInfo bufferCreateInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                            .size = bufferSizeBytes,
                                            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT };
        visibilityLUT = allocator.createBuffer(bufferCreateInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        visibilityLUTprevious = allocator.createBuffer(bufferCreateInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        debugUtil.setObjectName(visibilityLUT.buffer, "visibility lut");

        //RESSOURCES FOR TEMPORAL GRADIENT
        temporalGradientBuffer = createImage(context, allocator, render_width, render_height, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT);
        temporalGradientBufferView = createImageView(context, temporalGradientBuffer.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        debugUtil.setObjectName(temporalGradientBuffer.image, "temporal gradient buffer");

        //RESSOURCES FOR TEMPORAL FILTERING
        filteredImageBuffer = createImage(context, allocator, render_width, render_height, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT);
        filteredImageBufferView = createImageView(context, filteredImageBuffer.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        debugUtil.setObjectName(filteredImageBuffer.image, "temporal filtered image");
    }

    void loadMesh(const char** argv) {
        // Load the mesh of the first shape from an OBJ file
        const std::string        exePath(argv[0], std::string(argv[0]).find_last_of("/\\") + 1);
        searchPaths = { "C:/Users/onurb/Visual_Studio_Projects/IG3DA_Project/dependencies/vk_mini_path_tracer/_edit"};

        std::cout << searchPaths[0] << std::endl;
        //std::cout << searchPaths[1] << std::endl;
        tinyobj::ObjReader       reader;  // Used to read an OBJ file
        reader.ParseFromFile(nvh::findFile("C:/Users/onurb/Visual_Studio_Projects/IG3DA_Project/dependencies/vk_mini_path_tracer/scenes/CornellBox-Original-Merged.obj", searchPaths));
        assert(reader.Valid());  // Make sure tinyobj was able to parse this file
        objVertices = reader.GetAttrib().GetVertices();
        const std::vector<tinyobj::shape_t>& objShapes = reader.GetShapes();  // All shapes in the file
        assert(objShapes.size() == 1);                                          // Check that this file has only one shape
        const tinyobj::shape_t& objShape = objShapes[0];                        // Get the first shape
        // Get the indices of the vertices of the first mesh of `objShape` in `attrib.vertices`:
        objIndices.reserve(objShape.mesh.indices.size());
        for (const tinyobj::index_t& index : objShape.mesh.indices)
        {
            objIndices.push_back(index.vertex_index);
        }


        //TO CHANGE
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        const char* path = "C:\\Users\\onurb\\Visual_Studio_Projects\\IG3DA_Project\\dependencies\\vk_mini_path_tracer\\scenes\\CornellBox-Original-Merged.obj";
        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path)) {
            throw std::runtime_error(warn + err);
        }

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                if (uniqueVertices.count(vertex) == 0) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);
            }
        }
    }

    void createCommandPool() {
        // Create the command pool
        VkCommandPoolCreateInfo cmdPoolInfo{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,  //
                                            .queueFamilyIndex = context.m_queueGCT };
        NVVK_CHECK(vkCreateCommandPool(context, &cmdPoolInfo, nullptr, &cmdPool));
    }

    void uploadBuffers() {
        // Start a command buffer for uploading the buffers
        VkCommandBuffer uploadCmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);
        // We get these buffers' device addresses, and use them as storage buffers and build inputs.
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vertexBuffer = allocator.createBuffer(uploadCmdBuffer, objVertices, usage);
        indexBuffer = allocator.createBuffer(uploadCmdBuffer, objIndices, usage);

        //upload uniform buffer
        ubo.model = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(cameraOrigin, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj = glm::perspective((float)FOV * 2, render_width / (float)render_height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        //initialize previous matrices as the current for the first frame
        ubo.modelPrev = ubo.model;
        ubo.viewPrev = ubo.view;
        ubo.projPrev = ubo.proj;

        std::vector<UniformBufferObject> ubos;
        ubos.push_back(ubo);

        const VkBufferUsageFlags usageubo = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        const VkMemoryPropertyFlags propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        uniformBuffer = allocator.createBuffer(uploadCmdBuffer, ubos, usageubo, propertyFlags);

        // Also, let's transition the layout of `image` to `VK_IMAGE_LAYOUT_GENERAL`,
        // and the layout of `imageLinear` to `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL`.
        // Although we use `imageLinear` later, we're transferring its layout as
        // early as possible. For more complex applications, tracking images and
        // operations using a graph is a good way to handle these types of images
        // automatically. However, for this tutorial, we'll show how to write
        // image transitions by hand.

        // To do this, we combine both transitions in a single pipeline barrier.
        // This pipeline barrier will say "Make it so that all writes to memory by
        const VkAccessFlags srcAccesses = 0;  // (since image and imageLinear aren't initially accessible)
        // finish and can be read correctly by
        const VkAccessFlags dstImageAccesses = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;  // for image
        // "

        // Here's how to do that:
        const VkPipelineStageFlags srcStages = nvvk::makeAccessMaskPipelineStageFlags(srcAccesses);
        const VkPipelineStageFlags dstStages = nvvk::makeAccessMaskPipelineStageFlags(dstImageAccesses);
        VkImageMemoryBarrier imageBarriers[5];
        // Image memory barrier for images from UNDEFINED to GENERAL layout:
        imageBarriers[0] = nvvk::makeImageMemoryBarrier(image.image,                    // The VkImage
            srcAccesses, dstImageAccesses,  // Source and destination access masks
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,  // Source and destination layouts
            VK_IMAGE_ASPECT_COLOR_BIT);  // Aspects of an image (color, depth, etc.)
        imageBarriers[1] = nvvk::makeImageMemoryBarrier(previousImage.image, srcAccesses, dstImageAccesses, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
        imageBarriers[2] = nvvk::makeImageMemoryBarrier(previousVisibilityBuffer.image, srcAccesses, dstImageAccesses, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
        imageBarriers[3] = nvvk::makeImageMemoryBarrier(temporalGradientBuffer.image, srcAccesses, dstImageAccesses, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
        imageBarriers[4] = nvvk::makeImageMemoryBarrier(filteredImageBuffer.image, srcAccesses, dstImageAccesses, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
        // Include the two image barriers in the pipeline barrier:
        vkCmdPipelineBarrier(uploadCmdBuffer,       // The command buffer
            srcStages, dstStages,  // Src and dst pipeline stages
            0,                     // Flags for memory dependencies
            0, nullptr,            // Global memory barrier objects
            0, nullptr,            // Buffer memory barrier objects
            5, imageBarriers);     // Image barrier objects

        EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, uploadCmdBuffer);
        allocator.finalizeAndReleaseStaging();

    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(context.m_physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(context, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(context, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(context, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(context, buffer, bufferMemory, 0);
    }

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = cmdPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(context, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(context.m_queueGCT, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(context.m_queueGCT);

        vkFreeCommandBuffers(context, cmdPool, 1, &commandBuffer);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(context, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)bufferSize);
        vkUnmapMemory(context, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer2, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer2, bufferSize);

        vkDestroyBuffer(context, stagingBuffer, nullptr);
        vkFreeMemory(context, stagingBufferMemory, nullptr);
    }

    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(context, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(context, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer2, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer2, bufferSize);

        vkDestroyBuffer(context, stagingBuffer, nullptr);
        vkFreeMemory(context, stagingBufferMemory, nullptr);
    }

    void initializeSceneConstants() {
        pushConstants.currentCameraColor = lightColor;
        lightPos = lightPos;
        pushConstants.lightPos = lightPos;
        pushConstants.lightPosPrev = lightPosPrev;
    }

    void imageLayoutTranstion(VkCommandBuffer cmdBuffer, VkImage image, VkAccessFlags srcAcs, VkAccessFlags dstAcs, VkImageLayout srcLayout, VkImageLayout dstLayout, VkImageAspectFlagBits aspectBit = VK_IMAGE_ASPECT_COLOR_BIT) {
        //transition layout for the image from VK_IMAGE_LAYOUT_UNDEFINED to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        const VkAccessFlags        srcAccesses = srcAcs;
        const VkAccessFlags        dstAccesses = dstAcs;
        const VkPipelineStageFlags srcStages = nvvk::makeAccessMaskPipelineStageFlags(srcAccesses);
        const VkPipelineStageFlags dstStages = nvvk::makeAccessMaskPipelineStageFlags(dstAccesses);
        const VkImageMemoryBarrier barrier =
            nvvk::makeImageMemoryBarrier(image,               // The VkImage
                srcAccesses, dstAccesses,  // Src and dst access masks
                srcLayout, dstLayout,  // Src and dst layouts
                aspectBit);
        vkCmdPipelineBarrier(cmdBuffer,             // Command buffer
            srcStages, dstStages,  // Src and dst pipeline stages
            0,                     // Dependency flags
            0, nullptr,            // Global memory barriers
            0, nullptr,            // Buffer memory barriers
            1, &barrier);          // Image memory barriers
    }

    void buildAccelerationStructure() {
        // Describe the bottom-level acceleration structure (BLAS)
        std::vector<nvvk::RaytracingBuilderKHR::BlasInput> blases;
        {
            nvvk::RaytracingBuilderKHR::BlasInput blas;
            // Get the device addresses of the vertex and index buffers
            VkDeviceAddress vertexBufferAddress = GetBufferDeviceAddress(context, vertexBuffer.buffer);
            VkDeviceAddress indexBufferAddress = GetBufferDeviceAddress(context, indexBuffer.buffer);
            // Specify where the builder can find the vertices and indices for triangles, and their formats:
            VkAccelerationStructureGeometryTrianglesDataKHR triangles{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                .vertexData = {.deviceAddress = vertexBufferAddress},
                .vertexStride = 3 * sizeof(float),
                .maxVertex = static_cast<uint32_t>(objVertices.size() / 3 - 1),
                .indexType = VK_INDEX_TYPE_UINT32,
                .indexData = {.deviceAddress = indexBufferAddress},
                .transformData = {.deviceAddress = 0}  // No transform
            };
            // Create a VkAccelerationStructureGeometryKHR object that says it handles opaque triangles and points to the above:
            VkAccelerationStructureGeometryKHR geometry{ .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                                                        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                                                        .geometry = {.triangles = triangles},
                                                        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR };
            blas.asGeometry.push_back(geometry);
            // Create offset info that allows us to say how many triangles and vertices to read
            VkAccelerationStructureBuildRangeInfoKHR offsetInfo{
                .primitiveCount = static_cast<uint32_t>(objIndices.size() / 3),  // Number of triangles
                .primitiveOffset = 0,                                             // Offset added when looking up triangles
                .firstVertex = 0,  // Offset added when looking up vertices in the vertex buffer
                .transformOffset = 0   // Offset added when looking up transformation matrices, if we used them
            };
            blas.asBuildOffsetInfo.push_back(offsetInfo);
            blases.push_back(blas);
        }
        // Create the BLAS
        raytracingBuilder.setup(context, &allocator, context.m_queueGCT);
        raytracingBuilder.buildBlas(blases, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
            | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);

        // Create an instance pointing to this BLAS, and build it into a TLAS:
        std::vector<VkAccelerationStructureInstanceKHR> instances;
        {
            VkAccelerationStructureInstanceKHR instance{};
            instance.accelerationStructureReference = raytracingBuilder.getBlasDeviceAddress(0);  // The address of the BLAS in `blases` that this instance points to
            // Set the instance transform to the identity matrix:
            instance.transform.matrix[0][0] = instance.transform.matrix[1][1] = instance.transform.matrix[2][2] = 1.0f;
            instance.instanceCustomIndex = 0;  // 24 bits accessible to ray shaders via rayQueryGetIntersectionInstanceCustomIndexEXT
            // Used for a shader offset index, accessible via rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT
            instance.instanceShaderBindingTableRecordOffset = 0;
            instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;  // How to trace this instance
            instance.mask = 0xFF;
            instances.push_back(instance);
        }
        raytracingBuilder.buildTlas(instances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
    }

    void createAndBindDescriptorSet() {
        //1) Descriptor set of the path tracer compute pipeline
        descriptorSetContainer.init(context);
        descriptorSetContainer.addBinding(BINDING_IMAGEDATA, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        descriptorSetContainer.addBinding(BINDING_TLAS, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        descriptorSetContainer.addBinding(BINDING_VERTICES, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        descriptorSetContainer.addBinding(BINDING_INDICES, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        descriptorSetContainer.addBinding(BINDING_PREVIOUS_IMAGE_DATA, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        descriptorSetContainer.addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT); // temporal gradient
        descriptorSetContainer.addBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT); //depth buffer
        descriptorSetContainer.addBinding(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT); //visibility lut table
        // Create a layout from the list of bindings
        descriptorSetContainer.initLayout();
        // Create a descriptor pool from the list of bindings with space for 1 set, and allocate that set
        descriptorSetContainer.initPool(1);
        // Create a push constant range describing the amount of data for the push constants.
        static_assert(sizeof(PushConstants) % 4 == 0, "Push constant size must be a multiple of 4 per the Vulkan spec!");
        VkPushConstantRange pushConstantRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,  //
                                              .offset = 0,                            //
                                              .size = sizeof(PushConstants) };
        // Create a pipeline layout from the descriptor set layout and push constant range:
        descriptorSetContainer.initPipeLayout(1,                    // Number of push constant ranges
            &pushConstantRange);  // Pointer to push constant ranges

        // Write values into the descriptor set.
        std::array<VkWriteDescriptorSet, 8> writeDescriptorSets;
        // Color image
        VkDescriptorImageInfo descriptorImageInfo{ .imageView = imageView,  // How the image should be accessed
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
        writeDescriptorSets[0] = descriptorSetContainer.makeWrite(0 /*set index*/, BINDING_IMAGEDATA /*binding*/, &descriptorImageInfo);
        // Top-level acceleration structure (TLAS)
        VkAccelerationStructureKHR tlasCopy = raytracingBuilder.getAccelerationStructure();  // So that we can take its address
        VkWriteDescriptorSetAccelerationStructureKHR descriptorAS{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                                                                  .accelerationStructureCount = 1,
                                                                  .pAccelerationStructures = &tlasCopy };
        writeDescriptorSets[1] = descriptorSetContainer.makeWrite(0, BINDING_TLAS, &descriptorAS);
        // Vertex buffer
        VkDescriptorBufferInfo vertexDescriptorBufferInfo{ .buffer = vertexBuffer.buffer, .range = VK_WHOLE_SIZE };
        writeDescriptorSets[2] = descriptorSetContainer.makeWrite(0, BINDING_VERTICES, &vertexDescriptorBufferInfo);
        // Index buffer
        VkDescriptorBufferInfo indexDescriptorBufferInfo{ .buffer = indexBuffer.buffer, .range = VK_WHOLE_SIZE };
        writeDescriptorSets[3] = descriptorSetContainer.makeWrite(0, BINDING_INDICES, &indexDescriptorBufferInfo);
        //previous frame image
        VkDescriptorImageInfo descriptorImageInfo2{ .imageView = previousImageView,  // How the image should be accessed
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
        writeDescriptorSets[4] = descriptorSetContainer.makeWrite(0 /*set index*/, BINDING_PREVIOUS_IMAGE_DATA /*binding*/, &descriptorImageInfo2);
        //temporal gradient texture
        VkDescriptorImageInfo descriptorImageInfo3{ .imageView = temporalGradientBufferView,  // How the image should be accessed
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
        writeDescriptorSets[5] = descriptorSetContainer.makeWrite(0, 5, &descriptorImageInfo3);
        //depth buffer
        VkDescriptorImageInfo descriptorImageInfo4{ .imageView = depthImageView,  // How the image should be accessed
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
        writeDescriptorSets[6] = descriptorSetContainer.makeWrite(0, 6, &descriptorImageInfo4);
        //visibility lut
        VkDescriptorBufferInfo visibilityLUTDescriptorBufferInfo{ .buffer = visibilityLUT.buffer, .range = VK_WHOLE_SIZE };
        writeDescriptorSets[7] = descriptorSetContainer.makeWrite(0, 7, &visibilityLUTDescriptorBufferInfo);

        vkUpdateDescriptorSets(context,                                            // The context
            static_cast<uint32_t>(writeDescriptorSets.size()),  // Number of VkWriteDescriptorSet objects
            writeDescriptorSets.data(),                         // Pointer to VkWriteDescriptorSet objects
            0, nullptr);  // An array of VkCopyDescriptorSet objects (unused)

        //2) Descriptor set of the visibility pipeline
        visibilityDescriptorSetContainer.init(context);
        visibilityDescriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT);
        visibilityDescriptorSetContainer.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT);
        visibilityDescriptorSetContainer.initLayout();
        visibilityDescriptorSetContainer.initPool(1);
        visibilityDescriptorSetContainer.initPipeLayout();

        std::array<VkWriteDescriptorSet, 2> writeDescriptorSetsVisibility;
        // Uniform buffer for visibility frame
        VkDescriptorBufferInfo uboDescriptorBufferInfo{ .buffer = uniformBuffer.buffer, .range = VK_WHOLE_SIZE };
        writeDescriptorSetsVisibility[0] = visibilityDescriptorSetContainer.makeWrite(0 /*set index*/, 0 /*binding*/, &uboDescriptorBufferInfo);
        uniformBuffersMapped = allocator.map(uniformBuffer);
        //visibility lut
        writeDescriptorSetsVisibility[1] = visibilityDescriptorSetContainer.makeWrite(0, 1, &visibilityLUTDescriptorBufferInfo);

        vkUpdateDescriptorSets(context, static_cast<uint32_t>(writeDescriptorSetsVisibility.size()), writeDescriptorSetsVisibility.data(), 0, nullptr);

        //3) Descriptor set of the temporal gradient pipeline
        //we need 1 visibility buffer of the current frame and 2 visibilityLUT from frames i and frames i-1, 1 storage image to store temporal gradient, 1 position buffer
        temporalGradientDescriptorContainer.init(context);
        temporalGradientDescriptorContainer.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        temporalGradientDescriptorContainer.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        temporalGradientDescriptorContainer.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        temporalGradientDescriptorContainer.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        temporalGradientDescriptorContainer.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);

        temporalGradientDescriptorContainer.initLayout();
        temporalGradientDescriptorContainer.initPool(1);
        temporalGradientDescriptorContainer.initPipeLayout(1, &pushConstantRange);

        std::array<VkWriteDescriptorSet, 5> writeDescriptorSetsTemporalGradient;
        VkDescriptorImageInfo temporalGradientDescriptorImageInfo1{ .imageView = visibilityBufferView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
        writeDescriptorSetsTemporalGradient[0] = temporalGradientDescriptorContainer.makeWrite(0, 0, &temporalGradientDescriptorImageInfo1);
        writeDescriptorSetsTemporalGradient[1] = temporalGradientDescriptorContainer.makeWrite(0, 1, &visibilityLUTDescriptorBufferInfo);
        VkDescriptorBufferInfo visibilityLUTPreviousDescriptorBufferInfo{ .buffer = visibilityLUTprevious.buffer, .range = VK_WHOLE_SIZE };
        writeDescriptorSetsTemporalGradient[2] = temporalGradientDescriptorContainer.makeWrite(0, 2, &visibilityLUTPreviousDescriptorBufferInfo);
        VkDescriptorImageInfo temporalGradientDescriptorImageInfo2{ .imageView = temporalGradientBufferView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
        writeDescriptorSetsTemporalGradient[3] = temporalGradientDescriptorContainer.makeWrite(0, 3, &temporalGradientDescriptorImageInfo2);
        VkDescriptorImageInfo temporalGradientDescriptorImageInfo3{ .imageView = positionBufferView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
        writeDescriptorSetsTemporalGradient[4] = temporalGradientDescriptorContainer.makeWrite(0, 4, &temporalGradientDescriptorImageInfo3);

        vkUpdateDescriptorSets(context, static_cast<uint32_t>(writeDescriptorSetsTemporalGradient.size()), writeDescriptorSetsTemporalGradient.data(), 0, nullptr);

        //4) Temporal filtering descriptor
        //we need to pass a storage image and depth, visibilityLUT and color for the filtering 
        temporalFilteringDescriptorContainer.init(context);
        temporalFilteringDescriptorContainer.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT); //storage image
        temporalFilteringDescriptorContainer.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT); //path traced image
        temporalFilteringDescriptorContainer.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT); //depth image
        temporalFilteringDescriptorContainer.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT); //visibilityLUT
        temporalFilteringDescriptorContainer.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT); //visibilityBuffer
        temporalFilteringDescriptorContainer.addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT); //visibilityBuffer
        temporalFilteringDescriptorContainer.addBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT); //previousVisibilityLUT
        temporalFilteringDescriptorContainer.addBinding(7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT); //worldPosition
        temporalFilteringDescriptorContainer.addBinding(8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT); //ubo
        temporalFilteringDescriptorContainer.addBinding(9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT); //worldPosition

        temporalFilteringDescriptorContainer.initLayout();
        temporalFilteringDescriptorContainer.initPool(1);
        temporalFilteringDescriptorContainer.initPipeLayout(1, &pushConstantRange);

        std::array<VkWriteDescriptorSet, 10> writeDescriptorSetsTemporalFiltering;
        // Color image
        VkDescriptorImageInfo temporalFilteringDescriptorInfo1{ .imageView = filteredImageBufferView,
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
        writeDescriptorSetsTemporalFiltering[0] = temporalFilteringDescriptorContainer.makeWrite(0, 0, &temporalFilteringDescriptorInfo1);
        VkDescriptorImageInfo temporalFilteringDescriptorInfo2{ .imageView = imageView,
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
        writeDescriptorSetsTemporalFiltering[1] = temporalFilteringDescriptorContainer.makeWrite(0, 1, &temporalFilteringDescriptorInfo2);
        VkDescriptorImageInfo temporalFilteringDescriptorInfo3{ .imageView = depthImageView,
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
        writeDescriptorSetsTemporalFiltering[2] = temporalFilteringDescriptorContainer.makeWrite(0, 2, &temporalFilteringDescriptorInfo3);
        //visibility lut
        VkDescriptorBufferInfo temporalFilteringDescriptorInfo4{ .buffer = visibilityLUT.buffer, .range = VK_WHOLE_SIZE };
        writeDescriptorSetsTemporalFiltering[3] = temporalFilteringDescriptorContainer.makeWrite(0, 3, &temporalFilteringDescriptorInfo4);
        //visibility buffer
        VkDescriptorImageInfo temporalFilteringDescriptorInfo5{ .imageView = visibilityBufferView,
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
        writeDescriptorSetsTemporalFiltering[4] = temporalFilteringDescriptorContainer.makeWrite(0, 4, &temporalFilteringDescriptorInfo5);
        //previous frame image
        VkDescriptorImageInfo temporalFilteringDescriptorInfo6{ .imageView = previousImageView,
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
        writeDescriptorSetsTemporalFiltering[5] = temporalFilteringDescriptorContainer.makeWrite(0, 5, &temporalFilteringDescriptorInfo6);
        //previous visibility lut
        VkDescriptorBufferInfo temporalFilteringDescriptorInfo7{ .buffer = visibilityLUTprevious.buffer, .range = VK_WHOLE_SIZE };
        writeDescriptorSetsTemporalFiltering[6] = temporalFilteringDescriptorContainer.makeWrite(0, 6, &temporalFilteringDescriptorInfo7);
        //world postition
        VkDescriptorImageInfo temporalFilteringDescriptorInfo8{ .imageView = positionBufferView,
                                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
        writeDescriptorSetsTemporalFiltering[7] = temporalFilteringDescriptorContainer.makeWrite(0, 7, &temporalFilteringDescriptorInfo8);
        //add ubo
        VkDescriptorBufferInfo temporalFilteringDescriptorInfo9{ .buffer = uniformBuffer.buffer, .range = VK_WHOLE_SIZE };
        writeDescriptorSetsTemporalFiltering[8] = temporalFilteringDescriptorContainer.makeWrite(0, 8, &temporalFilteringDescriptorInfo9);
        //add temporal gradient
        VkDescriptorImageInfo temporalFilteringDescriptorInfo10{ .imageView = temporalGradientBufferView,
                                                   .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
        writeDescriptorSetsTemporalFiltering[9] = temporalFilteringDescriptorContainer.makeWrite(0, 9, &temporalFilteringDescriptorInfo10);

        vkUpdateDescriptorSets(context, static_cast<uint32_t>(writeDescriptorSetsTemporalFiltering.size()), writeDescriptorSetsTemporalFiltering.data(), 0, nullptr);

    }

    void createComputePipeline() {
        //this function creates compute pipelines for the path tracer and the temporal gradient computation        
        // Shader loading and pipeline creation
        rayTraceModule = nvvk::createShaderModule(context, nvh::loadFile("shaders/raytrace.comp.glsl.spv", true, searchPaths));

        // Describes the entrypoint and the stage to use for this shader module in the pipeline
        VkPipelineShaderStageCreateInfo shaderStageCreateInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                              .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                              .module = rayTraceModule,
                                                              .pName = "main" };

        // Create the compute pipeline
        VkComputePipelineCreateInfo pipelineCreateInfo{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                                       .stage = shaderStageCreateInfo,
                                                       .layout = descriptorSetContainer.getPipeLayout() };
        // Don't modify flags, basePipelineHandle, or basePipelineIndex
        NVVK_CHECK(vkCreateComputePipelines(context,                 // Device
            VK_NULL_HANDLE,          // Pipeline cache (uses default)
            1, &pipelineCreateInfo,  // Compute pipeline create info
            nullptr,                 // Allocator (uses default)
            &pathTracePipeline));      // Output

        //temporal gradient pipeline generation
        temporalGradientModule = nvvk::createShaderModule(context, nvh::loadFile("shaders/temporalGradient.comp.glsl.spv", true, searchPaths));
        shaderStageCreateInfo.module = temporalGradientModule;


        pipelineCreateInfo.stage = shaderStageCreateInfo;
        pipelineCreateInfo.layout = temporalGradientDescriptorContainer.getPipeLayout();

        NVVK_CHECK(vkCreateComputePipelines(context,                 // Device
            VK_NULL_HANDLE,          // Pipeline cache (uses default)
            1, &pipelineCreateInfo,  // Compute pipeline create info
            nullptr,                 // Allocator (uses default)
            &temporalGradientPipeline));      // Output

        //temporal filtering pipeline
        temporalFilteringModule = nvvk::createShaderModule(context, nvh::loadFile("shaders/temporalFiltering.comp.glsl.spv", true, searchPaths));
        shaderStageCreateInfo.module = temporalFilteringModule;

        pipelineCreateInfo.stage = shaderStageCreateInfo;
        pipelineCreateInfo.layout = temporalFilteringDescriptorContainer.getPipeLayout();

        NVVK_CHECK(vkCreateComputePipelines(context,                 // Device
            VK_NULL_HANDLE,          // Pipeline cache (uses default)
            1, &pipelineCreateInfo,  // Compute pipeline create info
            nullptr,                 // Allocator (uses default)
            &temporalFilteringPipepline));      // Output
    }

    void createFramebuffer() {
        //create color 
        std::array<VkImageView, 3> attachments = {
            visibilityBufferView,
            positionBufferView,
            depthImageView
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = render_width;
        framebufferInfo.height = render_height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(context, &framebufferInfo, nullptr, &visibilityFramebuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }

    void createRenderPass() {

        VkAttachmentDescription colorAttachments[2] = {};
        //color attachment for visibility
        colorAttachments[0].format = VK_FORMAT_R16_SFLOAT;
        colorAttachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ///attachment for position buffer
        colorAttachments[1] = colorAttachments[0];
        colorAttachments[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = VK_FORMAT_D32_SFLOAT;
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRefs[2] = {
            {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {1,  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
        };

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 2;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 2;
        subpass.pColorAttachments = colorAttachmentRefs;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 3> attachments = { colorAttachments[0], colorAttachments[1], depthAttachment };
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(context, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createGraphicsPipelines() {
        //this function creates the graphics pipeline for the visibility buffer generation and temporal gradient pipeline
        //visibilityPipeline generation
        nvvk::GraphicsPipelineState visibilityPipelineState;
        visibilityPipelineState.addBindingDescription({ 0, sizeof(Vertex) });
        visibilityPipelineState.addAttributeDescriptions({ {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, pos))} });
        visibilityPipelineState.depthStencilState.depthTestEnable = true;
        visibilityPipelineState.rasterizationState.cullMode = VK_CULL_MODE_NONE;  // back face culling is disabled since in path tracer we can still see planes from behind

        //add blend attachment required for multiple output of the fragment shader
        VkPipelineColorBlendAttachmentState colorBlendAttachment = {
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
        };

        visibilityPipelineState.addBlendAttachmentState(colorBlendAttachment);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &visibilityDescriptorSetContainer.getLayout();

        if (vkCreatePipelineLayout(context, &pipelineLayoutInfo, nullptr, &visibilityPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        nvvk::GraphicsPipelineGenerator visibilityPipelineGenerator(context, visibilityPipelineLayout, renderPass, visibilityPipelineState);
        VkShaderModule vertexShaderModule = nvvk::createShaderModule(context, nvh::loadFile("shaders/visibility.vert.glsl.spv", true, searchPaths));
        VkShaderModule geometryShaderModule = nvvk::createShaderModule(context, nvh::loadFile("shaders/visibility.geom.glsl.spv", true, searchPaths));
        VkShaderModule fragmentShaderModule = nvvk::createShaderModule(context, nvh::loadFile("shaders/visibility.frag.glsl.spv", true, searchPaths));
        visibilityPipelineGenerator.addShader(vertexShaderModule, VK_SHADER_STAGE_VERTEX_BIT);
        visibilityPipelineGenerator.addShader(geometryShaderModule, VK_SHADER_STAGE_GEOMETRY_BIT);
        visibilityPipelineGenerator.addShader(fragmentShaderModule, VK_SHADER_STAGE_FRAGMENT_BIT);



        visibilityPipeline = visibilityPipelineGenerator.createPipeline();

        vkDestroyShaderModule(context, vertexShaderModule, nullptr);
        vkDestroyShaderModule(context, geometryShaderModule, nullptr);
        vkDestroyShaderModule(context, fragmentShaderModule, nullptr);

    }

    void drawScene() {
        //wait for the old frame to render
        vkWaitForFences(context, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(context, 1, &inFlightFence);

        //image transition to shader readable if it is not the firs frame
        if (frameCount > 0) {
            VkCommandBuffer cmdTransition = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);
            imageLayoutTranstion(cmdTransition, image.image, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
            imageLayoutTranstion(cmdTransition, previousImage.image, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
            //imageLayoutTranstion(cmdTransition, visibilityBuffer.image, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
            imageLayoutTranstion(cmdTransition, previousVisibilityBuffer.image, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
            EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdTransition);
        }

        updateScene();
        drawVisbilityBuffer(); //calculate the visibility buffer
        computeTemporalGradient();  //calculate temporal gradient
        drawSceneToImage(); //path trace new scene
        applyTemporalFiltering(); //temporal filtering
        copyImageToSwapChainsCurrentImage();
        frameCount++;
        std::cout << "scene drew and renderes" << std::endl;
    }

    void updateScene() {
        //this function updates the scene by user input

        //Camera movment
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cameraOrigin.z += speed;
            cameraMoved = true;
        }
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            cameraOrigin.z -= speed;
            cameraMoved = true;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            cameraOrigin.x -= speed;
            cameraMoved = true;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            cameraOrigin.x += speed;
            cameraMoved = true;
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            cameraOrigin.y += speed;
            cameraMoved = true;
        }
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            cameraOrigin.y -= speed;
            cameraMoved = true;
        }

        //Light movment
        if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
            lightPos.z -= speed;
        }
        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
            lightPos.z += speed;
        }
        if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) {
            lightPos.x += speed;
            if (lightPos.x > 2) {
                lightPos.x = -20;
            }
        }
        if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) {
            lightPos.x -= speed;
            if (lightPos.x < -20) {
                lightPos.x = 2;
            }
        }
        if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) {
            lightPos.y += speed;
        }
        if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS) {
            lightPos.y -= speed;
        }

        time = glfwGetTime();
        pushConstants.frameNumber = frameCount;

        pushConstants.previousCameraColor = pushConstants.currentCameraColor;
        //pushConstants.currentCameraColor = glm::vec3((sin(time* 5) + 1) * 0.5, (cos(time*5) + 1) * 0.5, 0.0);
        pushConstants.currentCameraColor = lightColor;

        pushConstants.lightPosPrev = pushConstants.lightPos;
        pushConstants.lightPos = lightPos;

        updateUBO();
        if (cameraMoved || frameCount == 0) {
            pushConstants.cameraPos = glm::vec4(cameraOrigin.x, cameraOrigin.y, cameraOrigin.z, 0.0);
            cameraMoved = false;
        }
    }

    void drawVisbilityBuffer() {
        //Draw the visibility buffer
        VkCommandBuffer cmdVisibilityBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);

        recordVisibilityCommandBuffer(cmdVisibilityBuffer);

        imageLayoutTranstion(cmdVisibilityBuffer, visibilityBuffer.image, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

        imageLayoutTranstion(cmdVisibilityBuffer, positionBuffer.image, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        imageLayoutTranstion(cmdVisibilityBuffer, depthImage.image, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_DEPTH_BIT);

        EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdVisibilityBuffer);
    }

    void computeTemporalGradient() {
        //this function computes the temporal gradient for each pixel by writing it to a texture
        VkCommandBuffer cmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);

        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporalGradientPipeline);
        VkDescriptorSet descriptorSet = temporalGradientDescriptorContainer.getSet(0);
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporalGradientDescriptorContainer.getPipeLayout(), 0, 1,
            &descriptorSet, 0, nullptr);

        vkCmdPushConstants(cmdBuffer, temporalGradientDescriptorContainer.getPipeLayout(),  // Pipeline layout
            VK_SHADER_STAGE_COMPUTE_BIT,             // Stage flags
            0,                                       // Offset
            sizeof(PushConstants),                   // Size in bytes
            &pushConstants);

        vkCmdDispatch(cmdBuffer, (render_width + WORKGROUP_WIDTH - 1) / WORKGROUP_WIDTH,
            (render_height + WORKGROUP_HEIGHT - 1) / WORKGROUP_HEIGHT, 1);

        EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdBuffer);
    }

    void drawSceneToImage() {
        const uint32_t NUM_SAMPLE_BATCHES = 1;
        for (uint32_t sampleBatch = 0; sampleBatch < NUM_SAMPLE_BATCHES; sampleBatch++)
        {
            // Create and start recording a command buffer
            VkCommandBuffer cmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);

            // Bind the compute shader pipeline
            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pathTracePipeline);
            // Bind the descriptor set
            VkDescriptorSet descriptorSet = descriptorSetContainer.getSet(0);
            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, descriptorSetContainer.getPipeLayout(), 0, 1,
                &descriptorSet, 0, nullptr);

            // Push push constants:
            pushConstants.sample_batch = sampleBatch;

            vkCmdPushConstants(cmdBuffer,                               // Command buffer
                descriptorSetContainer.getPipeLayout(),  // Pipeline layout
                VK_SHADER_STAGE_COMPUTE_BIT,             // Stage flags
                0,                                       // Offset
                sizeof(PushConstants),                   // Size in bytes
                &pushConstants);                         // Data

            // Run the compute shader with enough workgroups to cover the entire buffer:
            vkCmdDispatch(cmdBuffer, (render_width + WORKGROUP_WIDTH - 1) / WORKGROUP_WIDTH,
                (render_height + WORKGROUP_HEIGHT - 1) / WORKGROUP_HEIGHT, 1);

            // End and submit the command buffer, then wait for it to finish:
            EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdBuffer);
        }
    }

    void applyTemporalFiltering() {
        //this function dispatchs the command buffer that denoises the path traced image and applies adaptive temporal filtering
        //To do this we apply wavelet transform with multiple iteration
        pushConstants.maxWaveletIteration = maxWaveletIteration;
        for (int k = 1; k <= maxWaveletIteration; k++) {
            pushConstants.waveletIteration = k;

            //update descriptor depending on the iteration
            std::array<VkWriteDescriptorSet, 2> writeDescriptorSetsTemporalFiltering;
            if (k % 2 == 0) {
                VkDescriptorImageInfo temporalFilteringDescriptorInfo1{ .imageView = imageView,
                                                          .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
                writeDescriptorSetsTemporalFiltering[0] = temporalFilteringDescriptorContainer.makeWrite(0, 0, &temporalFilteringDescriptorInfo1);
                VkDescriptorImageInfo temporalFilteringDescriptorInfo2{ .imageView = filteredImageBufferView,
                                                          .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
                writeDescriptorSetsTemporalFiltering[1] = temporalFilteringDescriptorContainer.makeWrite(0, 1, &temporalFilteringDescriptorInfo2);
                vkUpdateDescriptorSets(context, static_cast<uint32_t>(writeDescriptorSetsTemporalFiltering.size()), writeDescriptorSetsTemporalFiltering.data(), 0, nullptr);
            }
            else {
                VkDescriptorImageInfo temporalFilteringDescriptorInfo1{ .imageView = filteredImageBufferView,
                                                          .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
                writeDescriptorSetsTemporalFiltering[0] = temporalFilteringDescriptorContainer.makeWrite(0, 0, &temporalFilteringDescriptorInfo1);
                VkDescriptorImageInfo temporalFilteringDescriptorInfo2{ .imageView = imageView,
                                                          .imageLayout = VK_IMAGE_LAYOUT_GENERAL };  // The image's layout
                writeDescriptorSetsTemporalFiltering[1] = temporalFilteringDescriptorContainer.makeWrite(0, 1, &temporalFilteringDescriptorInfo2);
                vkUpdateDescriptorSets(context, static_cast<uint32_t>(writeDescriptorSetsTemporalFiltering.size()), writeDescriptorSetsTemporalFiltering.data(), 0, nullptr);
            }

            VkCommandBuffer cmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);

            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporalFilteringPipepline);
            VkDescriptorSet descriptorSet = temporalFilteringDescriptorContainer.getSet(0);
            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporalFilteringDescriptorContainer.getPipeLayout(), 0, 1,
                &descriptorSet, 0, nullptr);

            vkCmdPushConstants(cmdBuffer, temporalFilteringDescriptorContainer.getPipeLayout(),  // Pipeline layout
                VK_SHADER_STAGE_COMPUTE_BIT,             // Stage flags
                0,                                       // Offset
                sizeof(PushConstants),                   // Size in bytes
                &pushConstants);

            vkCmdDispatch(cmdBuffer, (render_width + WORKGROUP_WIDTH - 1) / WORKGROUP_WIDTH,
                (render_height + WORKGROUP_HEIGHT - 1) / WORKGROUP_HEIGHT, 1);

            //image layout transition for the images before copying
            if (k == maxWaveletIteration) {
                imageLayoutTranstion(cmdBuffer, image.image, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            }

            EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdBuffer);
        }
    }

    void copyImageToSwapChainsCurrentImage() {
        bool recreated;
        if (!swapChain.acquireAutoResize(render_width, render_height, &recreated, nullptr)) {
            throw std::runtime_error("failed to acquire image");
        }

        //get the current frame frow the swap chain
        VkImage swapImage = swapChain.getActiveImage();
        VkImageView swapImageView = swapChain.getActiveImageView();

        //Image Layout modication necessary for image copying to previous frames
        VkCommandBuffer cmdTransition = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);
        //transition layout for the image from VK_IMAGE_LAYOUT_UNDEFINED to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        imageLayoutTranstion(cmdTransition, swapImage, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        //layout transition to transfer optimal for previous frame
        imageLayoutTranstion(cmdTransition, previousImage.image, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        //layout transition for the visibility buffer from color attachmente to transfer src optimal
        imageLayoutTranstion(cmdTransition, visibilityBuffer.image, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        //layout transtion for the previous visibility buffer
        imageLayoutTranstion(cmdTransition, previousVisibilityBuffer.image, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdTransition);

        //start recording to copy the image
        VkCommandBuffer cmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);

        //copy the rendered image to swapchain active image
        VkImageBlit blitRegion{
            .srcSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,    // Aspect: color channels
                .mipLevel = 0,                              // Mipmap level
                .baseArrayLayer = 0,                        // First layer
                .layerCount = 1                             // One layer
            },
            .srcOffsets = {
                {0, 0, 0},                                  // Top-left corner (0, 0, 0) in the source image
                {render_width, render_height, 1}             // Bottom-right corner in the source image (full extent)
            },
            .dstSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,    // Aspect: color channels
                .mipLevel = 0,                              // Mipmap level
                .baseArrayLayer = 0,                        // First layer
                .layerCount = 1                             // One layer
            },
            .dstOffsets = {
                {0, 0, 0},                                  // Top-left corner in the destination image
                {render_width, render_height, 1}             // Bottom-right corner in the destination image
            }
        };

        vkCmdBlitImage(cmdBuffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_LINEAR);

        //Copy path traced image to the previous frame
        vkCmdBlitImage(cmdBuffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, previousImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_LINEAR);

        //copy visibility buffer to the previous frame
        vkCmdBlitImage(cmdBuffer, visibilityBuffer.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, previousVisibilityBuffer.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_LINEAR);

        //copy the visibilityLUT to the previous visibility LUT
        VkDeviceSize bufferSize = objIndices.size() * 9 * sizeof(float);
        VkBufferCopy copyRegion{ .srcOffset = 0, .dstOffset = 0, .size = bufferSize };
        vkCmdCopyBuffer(cmdBuffer, visibilityLUT.buffer, visibilityLUTprevious.buffer, 1, &copyRegion);

        NVVK_CHECK(vkEndCommandBuffer(cmdBuffer));

        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;

        //we need to ensure to wait for the swapchain image to have been read already before present
        VkSemaphore swapchainReadSemaphore = swapChain.getActiveReadSemaphore();
        VkPipelineStageFlags swapChainReadFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &swapchainReadSemaphore;
        submitInfo.pWaitDstStageMask = &swapChainReadFlags;

        //once this submit completed it means we have written the swapchain image
        VkSemaphore swapChainWrittenSemaphore = swapChain.getActiveWrittenSemaphore();
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &swapChainWrittenSemaphore;

        //submit it
        if (vkQueueSubmit(context.m_queueGCT, 1, &submitInfo, inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw comand buffer");
        }
        //wait until the image is written
        vkQueueWaitIdle(context.m_queueGCT);

        //image transition layout from VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        cmdTransition = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);
        imageLayoutTranstion(cmdTransition, swapImage, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdTransition);

        //present the frame
        swapChain.present(context.m_queueGCT);
    }

    void recordVisibilityCommandBuffer(VkCommandBuffer commandBuffer) {

        VkExtent2D extent{ render_width, render_height };

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = visibilityFramebuffer;
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = extent;

        std::array<VkClearValue, 3> clearValues{};
        clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearValues[1].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearValues[2].depthStencil = { 1.0f, 0 };

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, visibilityPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)extent.width;
        viewport.height = (float)extent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = extent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkBuffer vertexBuffers[] = { vertexBuffer2 };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

        vkCmdBindIndexBuffer(commandBuffer, indexBuffer2, 0, VK_INDEX_TYPE_UINT32);

        //uint32_t vertex_size = static_cast<uint32_t>(vertices.size());
        //uint32_t indices_size = static_cast<uint32_t>(indices.size());

        VkDescriptorSet descriptorSet = visibilityDescriptorSetContainer.getSet();
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, visibilityDescriptorSetContainer.getPipeLayout(), 0, 1, &descriptorSet, 0, nullptr);

        //vkCmdDraw(commandBuffer, vertices.size(), 1, 0, 0);
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);
    }

    void updateUBO() {
        //copy the old matrices to previous
        ubo.modelPrev = ubo.model;
        ubo.viewPrev = ubo.view;
        ubo.projPrev = ubo.proj;

        ubo.model = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(cameraOrigin, glm::vec3(cameraOrigin.x, cameraOrigin.y, cameraOrigin.z - 6), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj = glm::perspective((float)FOV * 2, render_width / (float)render_height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped, &ubo, sizeof(ubo));
    }

    void freeRessources() {
        vkDestroyPipeline(context, pathTracePipeline, nullptr);
        vkDestroyPipeline(context, temporalGradientPipeline, nullptr);
        vkDestroyPipeline(context, visibilityPipeline, nullptr);
        vkDestroyPipeline(context, temporalFilteringPipepline, nullptr);
        vkDestroyFramebuffer(context, visibilityFramebuffer, nullptr);
        vkDestroyPipelineLayout(context, visibilityPipelineLayout, nullptr);
        vkDestroyShaderModule(context, rayTraceModule, nullptr);
        vkDestroyShaderModule(context, temporalGradientModule, nullptr);
        vkDestroyShaderModule(context, temporalFilteringModule, nullptr);
        vkDestroyRenderPass(context, renderPass, nullptr);
        descriptorSetContainer.deinit();
        temporalGradientDescriptorContainer.deinit();
        visibilityDescriptorSetContainer.deinit();
        temporalFilteringDescriptorContainer.deinit();

        vkDestroyBuffer(context, indexBuffer2, nullptr);
        vkFreeMemory(context, indexBufferMemory, nullptr);

        vkDestroyBuffer(context, vertexBuffer2, nullptr);
        vkFreeMemory(context, vertexBufferMemory, nullptr);

        raytracingBuilder.destroy();
        allocator.destroy(vertexBuffer);
        allocator.destroy(uniformBuffer);
        allocator.destroy(indexBuffer);
        allocator.destroy(visibilityLUT);
        allocator.destroy(visibilityLUTprevious);
        allocator.destroy(temporalGradientBuffer);
        vkDestroyCommandPool(context, cmdPool, nullptr);
        vkDestroyImageView(context, visibilityBufferView, nullptr);
        vkDestroyImageView(context, previousVisibilityBufferView, nullptr);
        vkDestroyImageView(context, depthImageView, nullptr);
        vkDestroyImageView(context, positionBufferView, nullptr);
        vkDestroyImageView(context, imageView, nullptr);
        vkDestroyImageView(context, previousImageView, nullptr);
        vkDestroyImageView(context, temporalGradientBufferView, nullptr);
        vkDestroyImageView(context, filteredImageBufferView, nullptr);
        allocator.destroy(image);
        allocator.destroy(previousImage);
        allocator.destroy(visibilityBuffer);
        allocator.destroy(previousVisibilityBuffer);
        allocator.destroy(depthImage);
        allocator.destroy(positionBuffer);
        allocator.destroy(filteredImageBuffer);
        allocator.deinit();
        swapChain.deinit();
        vkDestroyFence(context, inFlightFence, nullptr);
        vkDestroySurfaceKHR(context.m_instance, surface, nullptr);
        context.deinit();  // Don't forget to clean up at the end of the program!
        glfwDestroyWindow(window);
    }
};


int main(int argc, const char** argv)
{
    PathTracingApplication app(argv);

    app.run();
}
