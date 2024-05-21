cmake_minimum_required(VERSION 3.0.2)
project(gridmap_interface)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
    grid_map_core
    grid_map_ros
    grid_map_msgs
    convex_plane_decomposition
    convex_plane_decomposition_ros
    visualization_msgs  
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES gridmap_interface
  CATKIN_DEPENDS grid_map_core grid_map_ros grid_map_msgs 
  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/GridmapInterface.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES} -lffi
)

add_executable(main src/main.cpp)
target_link_libraries(main ${PROJECT_NAME})

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