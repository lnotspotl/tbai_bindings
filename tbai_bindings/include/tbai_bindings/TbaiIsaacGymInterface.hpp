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

#include <ros/ros.h>
#include <tbai_bindings/bindings_visualize.h>

namespace tbai {
namespace bindings {

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
    void updateCurrentStates(const torch::Tensor &newStates);
    void updateCurrentStates(const torch::Tensor &newStates, const torch::Tensor &envIds);
    void updateOptimizedStates(scalar_t time);
    void updateOptimizedStates(scalar_t time, const torch::Tensor &envIds);
    void optimizeTrajectories(scalar_t time);
    void optimizeTrajectories(scalar_t time, const torch::Tensor &envIds);
    void setCurrentCommand(const torch::Tensor &command, const torch::Tensor &envIds);
    void updateDesiredContacts(scalar_t time, const torch::Tensor &envIds);
    void updateTimeLeftInPhase(scalar_t time, const torch::Tensor &envIds);
    void updateDesiredJointAngles(scalar_t time, const torch::Tensor &envIds);
    void updateCurrentDesiredJointAngles(scalar_t time, const torch::Tensor &envIds);
    void updateNextOptimizationTime(scalar_t time, const torch::Tensor &envIds);
    void updateDesiredBase(scalar_t time, const torch::Tensor &envIds);
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

    void visualize(scalar_t time, torch::Tensor &state, int envId, torch::Tensor &obs);

    PrimalSolution getCurrentOptimalTrajectory(int envId) const;
    SystemObservation getCurrentObservation(scalar_t time, int envId) const;

    // Move relevant tensors to CPU and convert them to Eigen data types
    void toCpu();

    // Move relevant tensors to GPU
    void toGpu();

   private:
    void allocateInterfaceBuffers();
    void allocateEigenBuffers();
    void allocateTorchBuffers();

    const LeggedRobotInterface &getInterface(int i) const;

    void loadModeSequenceTemplates(const std::string &gaitFile, const std::string &gaitName);

    void createInterfaces(const std::string &taskFile, const std::string &urdfFile, const std::string &referenceFile);

    void updateNextOptimizationTimeImpl(scalar_t time, int envId);
    scalar_t computeConsistencyReward(const PrimalSolution &previousSolution, const PrimalSolution &currentSolution);

    TargetTrajectories getTargetTrajectory(scalar_t initTime, int envIdx);

    int numEnvs_;
    int numThreads_;

    matrix_t currentStatesCpu_;
    matrix_t currentCommandsCpu_;
    torch::Tensor optimizedStates_;
    torch::Tensor consistencyRewards_;
    std::vector<PrimalSolution> solutions_;

    std::vector<std::unique_ptr<LeggedRobotInterface>> interfacePtrs_;
    std::vector<std::unique_ptr<SqpSolver>> solverPtrs_;
    std::vector<std::unique_ptr<PinocchioInterface>> pinocchioInterfacePtrs_;
    std::vector<std::unique_ptr<PinocchioEndEffectorKinematics>> endEffectorKinematicsPtrs_;
    std::vector<std::unique_ptr<CentroidalModelPinocchioMapping>> centroidalModelMappingPtrs_;
    std::vector<std::unique_ptr<CentroidalModelRbdConversions>> centroidalModelRbdConversionsPtrs_;

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

    torch::Tensor desiredBasePositions_;
    torch::Tensor desiredBaseOrientations_;
    torch::Tensor desiredBaseLinearVelocities_;
    torch::Tensor desiredBaseAngularVelocities_;
    torch::Tensor desiredBaseLinearAccelerations_;
    torch::Tensor desiredBaseAngularAccelerations_;

    torch::Device device_;

    BS::thread_pool threadPool_;

    vector_t initialState_;

    bool visualize_;

    // MPC horizon in seconds
    scalar_t horizon_;

    std::unique_ptr<ocs2::legged_robot::ModeSequenceTemplate> modeSequenceTemplate_;

    ros::Publisher pub_;
};

}  // namespace bindings
}  // namespace tbai