#include "gridmap_interface/GridmapInterface.hpp"

#include <convex_plane_decomposition_ros/ParameterLoading.h>
#include <visualization_msgs/MarkerArray.h>

namespace tbai {
namespace bindings {

GridmapInterface::GridmapInterface(bool rviz_visualize) : rviz_visualize_(rviz_visualize) {
    setupPlaneDecompositionPipeline();

    if (rviz_visualize_) {
        ros::NodeHandle nodeHandle;
        gridmapPublisher_ = nodeHandle.advertise<grid_map_msgs::GridMap>("map", 1);
        boundaryPublisher_ = nodeHandle.advertise<visualization_msgs::MarkerArray>("boundaries", 1);
        insetPublisher_ = nodeHandle.advertise<visualization_msgs::MarkerArray>("insets", 1);
    }
}

void GridmapInterface::setupPlaneDecompositionPipeline() {
    using namespace convex_plane_decomposition;
    PlaneDecompositionPipeline::Config config;

    // These values are simply taken from the config file
    // https://github.com/leggedrobotics/elevation_mapping_cupy/blob/0126df19f0d3a57f940eb08e4481d9feed52ef51/plane_segmentation/convex_plane_decomposition_ros/config/parameters.yaml
    config.preprocessingParameters.resolution = 0.02;
    config.preprocessingParameters.kernelSize = 3;
    config.preprocessingParameters.numberOfRepeats = 1;

    config.contourExtractionParameters.marginSize = 1;

    config.ransacPlaneExtractorParameters.probability = 0.001;
    config.ransacPlaneExtractorParameters.min_points = 4;
    config.ransacPlaneExtractorParameters.epsilon = 0.025;
    config.ransacPlaneExtractorParameters.cluster_epsilon = 0.041;
    config.ransacPlaneExtractorParameters.cluster_epsilon = 25;

    config.slidingWindowPlaneExtractorParameters.kernel_size = 3;
    config.slidingWindowPlaneExtractorParameters.planarity_opening_filter = 0;
    config.slidingWindowPlaneExtractorParameters.plane_inclination_threshold = std::cos(30.0 * M_PI / 180.0);
    config.slidingWindowPlaneExtractorParameters.local_plane_inclination_threshold = std::cos(35.0 * M_PI / 180.0);
    config.slidingWindowPlaneExtractorParameters.plane_patch_error_threshold = 0.02;
    config.slidingWindowPlaneExtractorParameters.min_number_points_per_label = 4;
    config.slidingWindowPlaneExtractorParameters.connectivity = 4;
    config.slidingWindowPlaneExtractorParameters.include_ransac_refinement = true;
    config.slidingWindowPlaneExtractorParameters.global_plane_fit_distance_error_threshold = 0.025;
    config.slidingWindowPlaneExtractorParameters.global_plane_fit_angle_error_threshold_degrees = 25.0;

    config.postprocessingParameters.extracted_planes_height_offset = 0.0;
    config.postprocessingParameters.nonplanar_height_offset = 0.02;
    config.postprocessingParameters.nonplanar_horizontal_offset = 1;
    config.postprocessingParameters.smoothing_dilation_size = 0.2;
    config.postprocessingParameters.smoothing_box_kernel_size = 0.1;
    config.postprocessingParameters.smoothing_gauss_kernel_size = 0.05;

    planeDecompositionPipeline_ = std::make_unique<PlaneDecompositionPipeline>(config);
}

void GridmapInterface::computeSegmentedPlanes() {
    const double submapLength = map_.getLength().x();
    const double submapWidth = map_.getLength().y();
    const std::string elevationLayer = "elevation";

    // Extract submap
    bool success;
    const grid_map::Position submapPosition = [&]() {
        // The map center might be between cells. Taking the submap there can result in changing submap dimensions.
        // project map center to an index and index to center s.t. we get the location of a cell.
        grid_map::Index centerIndex;
        grid_map::Position centerPosition;
        map_.getIndex(map_.getPosition(), centerIndex);
        map_.getPosition(centerIndex, centerPosition);
        return centerPosition;
    }();
    grid_map::GridMap elevationMap = map_.getSubmap(submapPosition, Eigen::Array2d(submapLength, submapWidth), success);

    // Run pipeline.
    planeDecompositionPipeline_->update(std::move(elevationMap), elevationLayer);
}

void GridmapInterface::updateFromFlattened(Eigen::VectorXf &heights, float length_x, float length_y,
                                           float resolution, float x, float y) {
    const size_t Nx = length_x / resolution;
    const size_t Ny = length_y / resolution;
    Eigen::MatrixXf mapMatrix = Eigen::Map<Eigen::MatrixXf>(heights.data(), Nx, Ny);
    auto &map = getMap();
    grid_map::Length length(length_x, length_y);
    grid_map::Position position(x, y);
    map.setGeometry(length, resolution, position);
    map.setFrameId("odom");
    map.add("elevation", mapMatrix);
}

void GridmapInterface::visualizePlanarTerrain(PlanarTerrain &terrain) {
    if (!rviz_visualize_) {
        ROS_ERROR_STREAM("rviz_visualize_ is false. Cannot visualize planar terrain.");
        return;
    }

    using namespace convex_plane_decomposition;
    const std::string elevationLayer = "elevation";

    const grid_map::Matrix elevationRaw = map_.get(elevationLayer);
    terrain.gridMap.add("elevation_raw", elevationRaw);

    // Add segmentation
    terrain.gridMap.add("segmentation");
    planeDecompositionPipeline_->getSegmentation(terrain.gridMap.get("segmentation"));

    visualizeMap(terrain.gridMap);

    const double lineWidth = 0.005;  // [m] RViz marker size
    boundaryPublisher_.publish(convertBoundariesToRosMarkers(terrain.planarRegions, terrain.gridMap.getFrameId(),
                                                             terrain.gridMap.getTimestamp(), lineWidth));
    insetPublisher_.publish(convertInsetsToRosMarkers(terrain.planarRegions, terrain.gridMap.getFrameId(),
                                                      terrain.gridMap.getTimestamp(), lineWidth));
}

grid_map_msgs::GridMap GridmapInterface::toMessage(const grid_map::GridMap &gridMap) {
    grid_map_msgs::GridMap message;
    grid_map::GridMapRosConverter::toMessage(gridMap, message);
    return message;
}

}  // namespace bindings
}  // namespace tbai