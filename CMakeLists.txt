# This file should be used as a template for creating new projects with Python
# wrapping using the CMake tools

# ##############################################################################
# * 1. To create your own project, replace "phomo" with the actual name
# of your project
cmake_minimum_required(VERSION 3.9)
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD) # Don't complain about boost
endif()
project(phomo CXX C)
set(CXX_STANDARD 11)

# ##############################################################################
# * 2. Find GTSAM components so we have access to the GTSAM Cython install path
find_package(GTSAM REQUIRED) # Uses installed package
find_package(GTSAMCMakeTools)
include(GtsamBuildTypes)
include(GtsamTesting)

# Note: Since Jan-2019, GTSAMConfig.cmake defines exported CMake targets that
# automatically do include the include_directories() without the need to call
# include_directories(), just target_link_libraries(NAME gtsam)
# include_directories(${GTSAM_INCLUDE_DIR})

# ############## Decide on BOOST ######################################
# Enable or disable serialization with GTSAM_ENABLE_BOOST_SERIALIZATION
option(GTSAM_ENABLE_BOOST_SERIALIZATION "Enable Boost serialization" ON)

if(GTSAM_ENABLE_BOOST_SERIALIZATION)
  add_definitions(-DGTSAM_ENABLE_BOOST_SERIALIZATION)
endif()

option(GTSAM_USE_BOOST_FEATURES "Enable Features that use Boost" ON)

if(GTSAM_USE_BOOST_FEATURES)
  add_definitions(-DGTSAM_USE_BOOST_FEATURES)
endif()

if(GTSAM_ENABLE_BOOST_SERIALIZATION OR GTSAM_USE_BOOST_FEATURES)
  include(cmake/HandleBoost.cmake)
endif()

# ##############################################################################
# Handle other libraries.
include(cmake/HandleEigen.cmake)            # Eigen3

# ##############################################################################
# * 3. Add the local source directory for CMake Ensure that local folder is
# searched before library folders
include_directories(BEFORE "${PROJECT_SOURCE_DIR}")

# ##############################################################################
# * 4. Build static library from common sources. This is a common cmake step
# where we load all the source files and link against the GTSAM library.
add_library(
  ${PROJECT_NAME} 
  SHARED 
  cpp/SunSensorFactor.h 
  cpp/Unit3Point3SmoothnessFactor.h 
  cpp/McEwenFactor2.h 
  cpp/LunarLambert_AffinePoly3Factor2.h
  cpp/LunarLambert_AffinePoly4Factor2.h)
# add_library(${PROJECT_NAME} SHARED cpp/*)
target_link_libraries(${PROJECT_NAME} gtsam)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# ##############################################################################
# * 5. Install library
install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

# ##############################################################################
# # Build tests (CMake tracks the dependecy to link with GTSAM through our
# project's static library) gtsamAddTestsGlob("${PROJECT_NAME}"
# "tests/test*.cpp" "" "${PROJECT_NAME}")
# gtsamAddTestsGlob("${PROJECT_NAME}" "tests/test*.cpp" "" "${PROJECT_NAME}")
add_subdirectory(tests)

# ##############################################################################
# # Build scripts (CMake tracks the dependecy to link with GTSAM through our
# project's static library) gtsamAddExamplesGlob("*.cpp" "" "${PROJECT_NAME}")

# ##############################################################################
# Upto this point, we've only focused on building the C++ library. For details
# on how to build the python wrapper, please see the CMakeLists.txt file in the
# `python` directory.
add_subdirectory(python)

# ##############################################################################
# Add "make uninstall" target.
if(NOT TARGET uninstall)
  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake
  )
endif()
