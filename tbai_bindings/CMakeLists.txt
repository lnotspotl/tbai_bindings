cmake_minimum_required(VERSION 3.0.2)
project(tbai_bindings)

set(LIBTORCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/libtorch)
set(LIBTORCH_INCLUDE_DIR ${LIBTORCH_DIR}/include)
set(LIBTORCH_LIB_DIR ${LIBTORCH_DIR}/lib)
set(LIBTORCH_CMAKE_DIR ${LIBTORCH_DIR}/share/cmake/Torch)

find_package(catkin REQUIRED COMPONENTS
    ocs2_oc
    ocs2_mpc
    ocs2_robotic_tools
    ocs2_centroidal_model
    ocs2_legged_robot
    ocs2_legged_robot_ros
    pybind11_catkin
    ocs2_ros_interfaces
    ocs2_msgs
    message_generation
    gridmap_interface
    perceptive_mpc
    ocs2_anymal_commands
)
set(CMAKE_CXX_STANDARD 17)

execute_process(
  COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pinocchio REQUIRED)
find_package (Python3 3.8.18 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

set(Torch_DIR ${LIBTORCH_CMAKE_DIR})
find_package(Torch)
if (NOT Torch_FOUND)
    message(FATAL_ERROR "PyTorch Not Found! Move your libtorch to the ${LIBTORCH_DIR} folder.")
endif()
set(OTHER_LIBRARIES ${LIBTORCH_LIB_DIR}/libtorch_python.so)

add_message_files(
  FILES
    bindings_visualize.msg
)

generate_messages(
  DEPENDENCIES
    ocs2_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES  ${PROJECT_NAME} 
  CATKIN_DEPENDS ocs2_oc ocs2_mpc ocs2_robotic_tools ocs2_centroidal_model ocs2_legged_robot ocs2_legged_robot_ros  ocs2_msgs message_runtime
  DEPENDS 
)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${CMAKE_SOURCE_DIR}/../dependencies/thread_pool/include
 ${Python3_INCLUDE_DIRS}
 ${TORCH_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
  ${pinocchio_INCLUDE_DIRS}
)


add_library(${PROJECT_NAME}
  src/Utils.cpp
  src/TbaiIsaacGymInterface.cpp
)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${TORCH_LIBRARIES} ${Python3_LIBRARIES} ${pinocchio_LIBRARIES} ${OTHER_LIBRARIES} -lffi)
pybind11_add_module(tbai_ocs2_interface SHARED
  src/TbaiIsaacGymInterfaceBindings.cpp
)    
target_link_libraries(tbai_ocs2_interface PRIVATE
    ${TORCH_LIBRARIES}
    ${Python_LIBRARIES}
    ${PROJECT_NAME}
    ${OTHER_LIBRARIES}
    -lffi
)

add_executable(visualizer src/visualizer.cpp)
target_link_libraries(visualizer ${PROJECT_NAME})

# Install the python module
install(TARGETS tbai_ocs2_interface
  ARCHIVE DESTINATION ${CMAKE_SOURCE_DIR}/out
  LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/out
  RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/out
)

## Testing
catkin_add_gtest(test_utils
  test/testUtils.cpp
)
target_link_libraries(test_utils
  ${PROJECT_NAME}
)