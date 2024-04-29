cmake_minimum_required(VERSION 3.0.2)
project(tbai_bindings)

set(Torch_DIR /home/kuba/Downloads/libtorch/share/cmake/Torch)

find_package(catkin REQUIRED COMPONENTS
    ocs2_oc
    ocs2_mpc
    ocs2_robotic_tools
    ocs2_centroidal_model
    ocs2_legged_robot
    ocs2_legged_robot_ros
    pybind11_catkin
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

find_package(Torch REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES  ${PROJECT_NAME} 
  CATKIN_DEPENDS ocs2_oc ocs2_mpc ocs2_robotic_tools ocs2_centroidal_model ocs2_legged_robot ocs2_legged_robot_ros
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

# Apparently, this is every important!!!
set(OTHER_LIBRARIES /home/kuba/Downloads/libtorch/lib/libtorch_python.so)

add_library(${PROJECT_NAME}
  src/Utils.cpp
  src/TbaiIsaacGymInterface.cpp
)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${TORCH_LIBRARIES} ${Python3_LIBRARIES} ${pinocchio_LIBRARIES} ${OTHER_LIBRARIES})
pybind11_add_module(ig_interface SHARED
  src/TbaiIsaacGymInterfaceBindings.cpp
)    
target_link_libraries(ig_interface PRIVATE
    ${TORCH_LIBRARIES}
    ${Python_LIBRARIES}
    ${PROJECT_NAME}
    ${OTHER_LIBRARIES}
)

# Install the python module
install(TARGETS ig_interface
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