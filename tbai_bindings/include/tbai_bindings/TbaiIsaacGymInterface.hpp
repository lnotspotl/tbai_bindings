#pragma once

// clang-format off
#include <pinocchio/fwd.hpp>
// clang-format on

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "tbai_bindings/Utils.hpp"
#include <BS_thread_pool.hpp>
#include <Eigen/Dense>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/PinocchioCentroidalDynamics.h>
#include <ocs2_centroidal_model/CentroidalModelRbdConversions.h>
#include <ocs2_core/misc/LinearInterpolation.h>
#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_legged_robot/LeggedRobotInterface.h>
#include <ocs2_legged_robot/gait/ModeSequenceTemplate.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_sqp/SqpSolver.h>
#include <tbai_bindings/Types.hpp>
#include <torch/extension.h>
#include <torch/torch.h>
#include <gridmap_interface/GridmapInterface.hpp>
#include <perceptive_mpc/perceptive_mpc.hpp>

#include <tbai_bindings/Rotations.hpp>

#include <ros/ros.h>
#include <tbai_bindings/bindings_visualize.h>

namespace tbai {
namespace bindings {

using switched_model::BaseReferenceCommand;
using switched_model::BaseReferenceHorizon;
using switched_model::BaseReferenceState;

using namespace ocs2;
using namespace ocs2::legged_robot;
using ocs2::CentroidalModelPinocchioMapping;
using ocs2::PinocchioCentroidalDynamics;

class TbaiIsaacGymInterface {
   public:
    TbaiIsaacGymInterface(const std::string &taskFile, const std::string &urdfFile, const std::string &referenceFile,
                          const std::string &gaitFile, const std::string &gaitName, int numEnvs, int numThreads,
                          torch::Device device = torch::kCPU, bool visualize = false);

    /** Public interface **/
    void resetAllSolvers(scalar_t time);
    void resetSolvers(scalar_t time, const torch::Tensor &envIds);
    void updateCurrentStatesPerceptive(const torch::Tensor &newStates, const torch::Tensor &envIds);
    void updateOptimizedStates(scalar_t time);
    void updateOptimizedStates(scalar_t time, const torch::Tensor &envIds);
    void optimizeTrajectories(scalar_t time);
    void optimizeTrajectories(scalar_t time, const torch::Tensor &envIds);
    void setCurrentCommand(const torch::Tensor &command, const torch::Tensor &envIds);
    void updateDesiredContacts(scalar_t time, const torch::Tensor &envIds);
    void updateTimeLeftInPhase(scalar_t time, const torch::Tensor &envIds);
    void updateDesiredJointAnglesPerceptive(scalar_t time, const torch::Tensor &envIds);
    void updateCurrentDesiredJointAnglesPerceptive(scalar_t time, const torch::Tensor &envIds);
    void updateNextOptimizationTime(scalar_t time, const torch::Tensor &envIds);
    void updateDesiredBasePerceptive(scalar_t time, const torch::Tensor &envIds);
    void updateDesiredFootPositionsAndVelocitiesPerceptive(scalar_t time, const torch::Tensor &envIds);
    void moveDesiredBaseToGpu();

    /** Getters **/
    torch::Tensor &getOptimizedStates() { return optimizedStates_; }
    torch::Tensor &getUpdatedInSeconds() { return updateInSeconds_; }
    torch::Tensor &getConsistencyReward() { return consistencyRewards_; }
    torch::Tensor &getPlanarFootHolds() { return desiredFootholds_; }
    torch::Tensor &getDesiredJointPositions() { return desiredJointAngles_; }
    torch::Tensor &getDesiredContacts() { return desiredContacts_; }
    torch::Tensor &getTimeLeftInPhase() { return timeLeftInPhase_; }
    torch::Tensor &getCurrentDesiredJointPositions() { return currentDesiredJointAngles_; }
    torch::Tensor &getDesiredBasePositions() { return desiredBasePositions_; }
    torch::Tensor &getDesiredBaseOrientations() { return desiredBaseOrientations_; }
    torch::Tensor &getDesiredBaseLinearVelocities() { return desiredBaseLinearVelocities_; }
    torch::Tensor &getDesiredBaseAngularVelocities() { return desiredBaseAngularVelocities_; }
    torch::Tensor &getDesiredBaseLinearAccelerations() { return desiredBaseLinearAccelerations_; }
    torch::Tensor &getDesiredBaseAngularAccelerations() { return desiredBaseAngularAccelerations_; }
    torch::Tensor &getDesiredFootPositions() { return desiredFootPositions_; }
    torch::Tensor &getDesiredFootVelocities() { return desiredFootVelocities_; }

    void visualize(scalar_t time, torch::Tensor &state, int envId, torch::Tensor &obs);

    PrimalSolution getCurrentOptimalTrajectory(int envId) const;

    BaseReferenceHorizon getBaseReferenceHorizon(scalar_t time, int envId) { return {0.1, 10}; }

    BaseReferenceState getBaseReferenceState(scalar_t time, int envId) {
        scalar_t observationTime = time;
        Eigen::Vector3d positionInWorld = currentStatesPerceptiveCpu_.row(envId).segment<3>(3);
        Eigen::Vector3d eulerXyz = currentStatesPerceptiveCpu_.row(envId).head<3>();
        return {observationTime, positionInWorld, eulerXyz};
    }

    BaseReferenceCommand getBaseReferenceCommand(scalar_t time, int envId) {
        auto command = currentCommandsCpu_.row(envId);
        scalar_t vx = command(0);
        scalar_t vy = command(1);
        scalar_t wz = command(2);
        scalar_t comHeight = 0.53;
        return {vx, vy, wz, comHeight};
    }

    // Make sure to call this function only after updateDesiredContacts has been called
    torch::Tensor getBobnetPhases(scalar_t time, const torch::Tensor &envIds);

    // Move relevant tensors to CPU and convert them to Eigen data types
    void toCpu();

    // Move relevant tensors to GPU
    void toGpu();

    void setMapsFromFlattened(const torch::Tensor &flattenedMaps, scalar_t lengthX, scalar_t lengthY,
                              scalar_t resolution, const torch::Tensor &xCoords, const torch::Tensor &yCoords,
                              const torch::Tensor &envIds);

    void setMapFromFlattened(const torch::Tensor &flattenedMap, scalar_t length_x, scalar_t length_y,
                             scalar_t resolution, scalar_t x, scalar_t y);

   private:
    void allocateInterfaceBuffers();
    void allocateEigenBuffers();
    void allocateTorchBuffers();

    void loadModeSequenceTemplates(const std::string &gaitFile, const std::string &gaitName);

    void createInterfaces(const std::string &taskFile, const std::string &urdfFile, const std::string &referenceFile);

    void updateNextOptimizationTimeImpl(scalar_t time, int envId);
    scalar_t computeConsistencyReward(const PrimalSolution &previousSolution, const PrimalSolution &currentSolution);

    TargetTrajectories getTargetTrajectoryPerceptive(scalar_t initTime, int envIdx);

    int numEnvs_;
    int numThreads_;

    matrix_t currentStatesCpu_;
    matrix_t currentStatesPerceptiveCpu_;
    vector_t lastYawCpu_;
    matrix_t currentCommandsCpu_;
    torch::Tensor optimizedStates_;
    torch::Tensor consistencyRewards_;
    std::vector<PrimalSolution> solutionsPerceptive_;

    std::vector<std::unique_ptr<GridmapInterface>> gridmapInterfacesPtrs_;
    std::vector<std::unique_ptr<switched_model::QuadrupedInterface>> quadrupedInterfacePtrs_;
    std::vector<std::unique_ptr<SqpSolver>> quadrupedSolverPtrs_;
    std::vector<std::unique_ptr<switched_model::ComModelBase<scalar_t>>> comModelPtrs_;
    std::vector<std::unique_ptr<switched_model::KinematicsModelBase<scalar_t>>> kinematicsPtrs_;
    switched_model::Gait gait_;

    torch::Tensor currentStates_;
    torch::Tensor currentCommands_;

    torch::Tensor desiredContacts_;
    torch::Tensor timeLeftInPhase_;
    torch::Tensor desiredJointAngles_;
    torch::Tensor currentDesiredJointAngles_;
    torch::Tensor desiredFootholds_;
    torch::Tensor updateInSeconds_;

    matrix_t desiredBasePositionsCpu_;
    matrix_t desiredBaseOrientationsCpu_;
    matrix_t desiredBaseLinearVelocitiesCpu_;
    matrix_t desiredBaseAngularVelocitiesCpu_;
    matrix_t desiredBaseLinearAccelerationsCpu_;
    matrix_t desiredBaseAngularAccelerationsCpu_;

    matrix_t desiredContactsCpu_;
    matrix_t timeLeftInPhaseCpu_;
    matrix_t desiredJointAnglesCpu_;
    matrix_t desiredFootHoldsCpu_;
    matrix_t currentDesiredJointAnglesCpu_;
    matrix_t desiredStatesCpu_;

    matrix_t desiredFootPositionsCpu_;
    matrix_t desiredFootVelocitiesCpu_;

    torch::Tensor desiredBasePositions_;
    torch::Tensor desiredBaseOrientations_;
    torch::Tensor desiredBaseLinearVelocities_;
    torch::Tensor desiredBaseAngularVelocities_;
    torch::Tensor desiredBaseLinearAccelerations_;
    torch::Tensor desiredBaseAngularAccelerations_;

    torch::Tensor desiredFootPositions_;
    torch::Tensor desiredFootVelocities_;

    torch::Device device_;

    BS::thread_pool threadPool_;

    vector_t initialState_;

    bool visualize_;

    // MPC horizon in seconds
    scalar_t horizon_;

    std::unique_ptr<switched_model::ModeSequenceTemplate> modeSequenceTemplate_;
    std::unique_ptr<GridmapInterface> gridmapInterfacePtr_;

    ros::Publisher pub_;
};

}  // namespace bindings
}  // namespace tbai