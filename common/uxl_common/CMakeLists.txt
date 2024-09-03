cmake_minimum_required(VERSION 3.17)
project(uxl_common)

## Note: Requires this:
##
## export DPCPP_HOME=~/sycl_workspace
## export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
## export CPATH=$DPCPP_HOME/llvm/build/include:$CPATH
## export CPATH=$DPCPP_HOME/llvm/build/include/sycl:$CPATH
## export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/install/lib:$LD_LIBRARY_PATH
## >> This is the one that makes autoware work finding sycl library:
## export LIBRARY_PATH=$DPCPP_HOME/llvm/build/install/lib:$LIBRARY_PATH

find_package(ament_cmake REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/uxl_common.cpp
)

target_link_libraries(${PROJECT_NAME}
  libsycl.so ## Make sure this is set: export LIBRARY_PATH=$DPCPP_HOME/llvm/build/install/lib:$LIBRARY_PATH
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

set_target_properties(${PROJECT_NAME}
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_compile_options(${PROJECT_NAME} PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wno-deprecated-declarations -Wno-error=comment -Wno-error=deprecated-declarations -Wno-error=unused-function -Wno-unused-function -Wno-deprecated -Wno-deprecated-declarations -Wno-comment
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

ament_export_include_directories("include/${PROJECT_NAME}")
ament_export_targets(export_${PROJECT_NAME})

ament_export_dependencies(
)

ament_package()
