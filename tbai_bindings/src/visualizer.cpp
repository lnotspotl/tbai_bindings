#include <iostream>
#include <memory>
#include <string>

#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_legged_robot/LeggedRobotInterface.h>
#include <ocs2_legged_robot_ros/visualization/LeggedRobotVisualizer.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_ros_interfaces/common/RosMsgConversions.h>
#include <ros/ros.h>
#include <tbai_bindings/bindings_visualize.h>

#include <tbai_bindings/Macros.hpp>

#define protected public
#define private public

#include <ocs2_ros_interfaces/mrt/MRT_ROS_Interface.h>

using namespace ocs2;
using namespace ocs2::legged_robot;

class Visualizer {
   public:
    Visualizer() {
        ros::NodeHandle nh;
        sub_ = nh.subscribe("visualize", 1, &Visualizer::visualize, this);

        std::string taskFile;
        std::string urdfFile;
        std::string referenceFile;

        TBAI_BINDINGS_STD_THROW_IF(!nh.getParam("task_file", taskFile), "Failed to get task_file parameter");
        TBAI_BINDINGS_STD_THROW_IF(!nh.getParam("urdf_file", urdfFile), "Failed to get urdf_file parameter");
        TBAI_BINDINGS_STD_THROW_IF(!nh.getParam("reference_file", referenceFile), "Failed to get reference_file parameter");

        TBAI_BINDINGS_PRINT("Task file: " + taskFile);
        TBAI_BINDINGS_PRINT("URDF file: " + urdfFile);
        TBAI_BINDINGS_PRINT("Reference file: " + referenceFile);

        TBAI_BINDINGS_PRINT("Creating interface...");
        interface_ = std::make_unique<LeggedRobotInterface>(taskFile, urdfFile, referenceFile);

        TBAI_BINDINGS_PRINT("Creating pinocchio mapping...");
        pinocchioMapping_ = std::make_unique<CentroidalModelPinocchioMapping>(interface_->getCentroidalModelInfo());
        
        TBAI_BINDINGS_PRINT("Creating end effector kinematics...");
        endEffectorKinematics_ = std::make_unique<PinocchioEndEffectorKinematics>(
            interface_->getPinocchioInterface(), *pinocchioMapping_, interface_->modelSettings().contactNames3DoF);

        TBAI_BINDINGS_PRINT("Creating visualizer...");
        leggedRobotVisualizer_ = std::make_unique<LeggedRobotVisualizer>(
            interface_->getPinocchioInterface(), interface_->getCentroidalModelInfo(), *endEffectorKinematics_, nh);

        TBAI_BINDINGS_PRINT("Visualizer ready.");
    }

    void visualize(const tbai_bindings::bindings_visualize::ConstPtr &msg) {
        auto observation = ocs2::ros_msg_conversions::readObservationMsg(msg->observation);
        auto targetTrajectories = ocs2::ros_msg_conversions::readTargetTrajectoriesMsg(msg->target_trajectories);

        CommandData commandData;
        PrimalSolution primalSolution;
        PerformanceIndex performanceIndex;
        MRT_ROS_Interface::readPolicyMsg(msg->flattened_controller, commandData, primalSolution, performanceIndex);

        std::cout << observation << std::endl;
        leggedRobotVisualizer_->update(observation, primalSolution, commandData);
    }

   private:
    ros::Subscriber sub_;
    std::unique_ptr<LeggedRobotInterface> interface_;
    std::unique_ptr<CentroidalModelPinocchioMapping> pinocchioMapping_;
    std::unique_ptr<PinocchioEndEffectorKinematics> endEffectorKinematics_;
    std::unique_ptr<LeggedRobotVisualizer> leggedRobotVisualizer_;
};

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "visualizer");

    Visualizer visualizer;

    ros::spin();
}