cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Set nvpro_core base directory
set(BASE_DIRECTORY ${CMAKE_SOURCE_DIR})

if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

#set(SUPPORTS_AFTERMATH ON)

#####################################################################################
# Add sample packages
#
_add_package_VulkanSDK()
_add_nvpro_core_lib()

# Include nvpro_core
#add_subdirectory("C:/Users/onurb/Visual_Studio_Projects/IG3DA_Project/Project/deneme_2/nvpro_core")

# Collect C++ source files
file(GLOB CPP_SOURCES "*.cpp")

# Collect shader files
file(GLOB SHADER_SOURCES "${CMAKE_SOURCE_DIR}/shaders/*.glsl" "${CMAKE_SOURCE_DIR}/shaders/*.spv")

# Mark shader files as header-only so they aren’t compiled.
set_source_files_properties(${SHADER_SOURCES} PROPERTIES HEADER_FILE_ONLY TRUE)

# Combine the lists
set(PROJECT_SOURCES ${CPP_SOURCES} ${SHADER_SOURCES})

add_executable(MyProject ${PROJECT_SOURCES})

# Pass the project directory as a macro to your code.
target_compile_definitions(MyProject PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Link against nvpro_core
target_link_libraries(MyProject PRIVATE nvpro_core)
