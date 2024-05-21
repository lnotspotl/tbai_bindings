cmake_minimum_required(VERSION 3.0.2)
project(perceptive_mpc)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
    ocs2_quadruped_interface
    ocs2_anymal_mpc
    ocs2_switched_model_interface
    tf
    ocs2_anymal_commands
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES perceptive_mpc
  CATKIN_DEPENDS ocs2_switched_model_interface
  DEPENDS system_lib
)

include_directories(
include
${catkin_INCLUDE_DIRS}
)


add_library(${PROJECT_NAME}
  src/perceptive_mpc.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES} -lffi
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   FILES_MATCHING PATTERN "*.hpp"
   PATTERN ".svn" EXCLUDE
 )