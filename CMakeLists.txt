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

# Add your own source files
#add_executable(MyProject main.cpp)

file(GLOB PROJECT_SOURCES "*.cpp")

add_executable(MyProject ${PROJECT_SOURCES})

# Link against nvpro_core
target_link_libraries(MyProject PRIVATE nvpro_core)
