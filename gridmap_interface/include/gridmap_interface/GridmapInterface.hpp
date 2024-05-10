#pragma once

#include <memory>
#include <string>

#include <convex_plane_decomposition/PlanarRegion.h>
#include <convex_plane_decomposition/PlaneDecompositionPipeline.h>
#include <convex_plane_decomposition_ros/RosVisualizations.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <ros/ros.h>

namespace tbai {
namespace bindings {

using convex_plane_decomposition::PlanarTerrain;
using convex_plane_decomposition::PlaneDecompositionPipeline;

class GridmapInterface {
   public:
    GridmapInterface(bool rviz_visualize);

    void visualizeMap(const grid_map::GridMap &map) { gridmapPublisher_.publish(toMessage(map)); }

    void computeSegmentedPlanes();

    void updateFromFlattened(Eigen::VectorXf &heights, float length_x, float length_y,
                             float resolution, float x, float y);

    PlanarTerrain &getPlanarTerrain() { return planeDecompositionPipeline_->getPlanarTerrain(); }

    void visualizePlanarTerrain(PlanarTerrain &terrain);

    grid_map::GridMap &getMap() { return map_; }
    const grid_map::GridMap &getMap() const { return map_; }

   private:
    void setupPlaneDecompositionPipeline();
    std::unique_ptr<PlaneDecompositionPipeline> planeDecompositionPipeline_;
    grid_map::GridMap map_;

    bool rviz_visualize_;
    ros::Publisher gridmapPublisher_;
    ros::Publisher insetPublisher_;
    ros::Publisher boundaryPublisher_;
    grid_map_msgs::GridMap toMessage(const grid_map::GridMap &gridMap);
};

}  // namespace bindings
}  // namespace tbai