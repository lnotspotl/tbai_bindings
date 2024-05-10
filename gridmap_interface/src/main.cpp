#include <Eigen/Dense>
#include <gridmap_interface/GridmapInterface.hpp>
#include <ros/ros.h>

using tbai::bindings::GridmapInterface;

int main(int argc, char *argv[]) {
    ROS_INFO_STREAM("Launching main.cpp");
    ros::init(argc, argv, "main");

    GridmapInterface gridmapInterface(true);

    // We can now start playing with the grid map.
    auto &map = gridmapInterface.getMap();

    // Set map geometry
    double length_x = 15.0; // meters
    double length_y = 15.0; // meters
    double resolution = 0.1; // meters
    grid_map::Length length(length_x, length_y);
    grid_map::Position position(0.0, 0.0);
    map.setGeometry(length, resolution, position);
    map.setFrameId("map");

    // Generate flat terrrain map
    const size_t N_x = map.getLength().x() / map.getResolution();
    const size_t N_y = map.getLength().y() / map.getResolution();

    // Create a random map
    Eigen::MatrixXf data = Eigen::MatrixXf::Random(N_x, N_y);
    for(int i = 0; i < N_x; i++) {
        for(int j = 0; j < N_y; j++) {
            constexpr float PI = 3.14159265359;
            float magnitude = static_cast<float>(i)/static_cast<float>(N_x);
            float phase = 4 * PI * (static_cast<float>(j)/static_cast<float>(N_x));
            float value = magnitude * std::sin(phase);
            data(i, j) = value;
        }
    }

    map.add("elevation", data);

    // Visualize the map
    while(ros::ok()) {
        auto t1 = std::chrono::high_resolution_clock::now();
        gridmapInterface.computeSegmentedPlanes();
        auto &planarTerrain = gridmapInterface.getPlanarTerrain();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e3;
        ROS_INFO_STREAM("Plane segmentation took " << duration << " ms.");

        gridmapInterface.visualizePlanarTerrain(planarTerrain);
        ros::Duration(0.3).sleep();
    }

    return 0;
}