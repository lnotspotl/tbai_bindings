#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "tbai_bindings/Utils.hpp"
#include <BS_thread_pool.hpp>
#include <Eigen/Dense>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/PinocchioCentroidalDynamics.h>
#include <tbai_bindings/Types.hpp>
#include <ocs2_core/misc/LinearInterpolation.h>
#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_legged_robot/LeggedRobotInterface.h>
#include <ocs2_legged_robot/gait/ModeSequenceTemplate.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_sqp/SqpSolver.h>
#include <torch/extension.h>
#include <torch/torch.h>

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
                          torch::Device device = torch::kCPU);

    void resetAllSolvers(scalar_t time);
    void resetSolvers(scalar_t time, const torch::Tensor &envIds);

    void updateCurrentStates(const torch::Tensor &newStates);
    void updateCurrentStates(const torch::Tensor &newStates, const torch::Tensor &envIds);

    void updateOptimizedStates(scalar_t time);
    void updateOptimizedStates(scalar_t time, const torch::Tensor &envIds);

    void optimizeTrajectories(scalar_t time);
    void optimizeTrajectories(scalar_t time, const torch::Tensor &envIds);

    void setCurrentCommand(const torch::Tensor &command, const torch::Tensor &envIds);

    torch::Tensor &getOptimizedStates();
    torch::Tensor &getUpdatedInSeconds();
    torch::Tensor &getConsistencyReward();

    torch::Tensor &getPlanarFootHolds();
    torch::Tensor &getDesiredJointPositions();
    torch::Tensor &getDesiredContacts();
    torch::Tensor &getTimeLeftInPhase();
    torch::Tensor &getCurrentDesiredJointPositions();

    torch::Tensor &getDesiredBasePositions();
    torch::Tensor &getDesiredBaseOrientations();
    torch::Tensor &getDesiredBaseLinearVelocities();
    torch::Tensor &getDesiredBaseAngularVelocities();
    torch::Tensor &getDesiredBaseLinearAccelerations();
    torch::Tensor &getDesiredBaseAngularAccelerations();

    const LeggedRobotInterface &getInterface(int i) const;
    void updateDesiredContacts(scalar_t time, const torch::Tensor &envIds);
    void updateTimeLeftInPhase(scalar_t time, const torch::Tensor &envIds);
    void updateDesiredJointAngles(scalar_t time, const torch::Tensor &envIds);
    void updateCurrentDesiredJointAngles(scalar_t time, const torch::Tensor &envIds);
    void updateNextOptimizationTime(scalar_t time, const torch::Tensor &envIds);

    void updateDesiredBase(scalar_t time, const torch::Tensor &envIds);
    void moveDesiredBaseToGpu();

    PrimalSolution getCurrentOptimalTrajectory(int envId) const;
    SystemObservation getCurrentObservation(scalar_t time, int envId) const;

   private:
    void updateNextOptimizationTimeImpl(scalar_t time, int envId);
    scalar_t computeConsistencyReward(const PrimalSolution &previousSolution, const PrimalSolution &currentSolution);

    TargetTrajectories getTargetTrajectory(scalar_t initTime, int envIdx);

    int numEnvs_;
    int numThreads_;

    matrix_t currentStates_;
    matrix_t currentCommands_;
    torch::Tensor optimizedStates_;
    torch::Tensor consistencyRewards_;
    std::vector<PrimalSolution> solutions_;

    std::vector<std::unique_ptr<LeggedRobotInterface>> interfaces_;
    std::vector<std::unique_ptr<SqpSolver>> solvers_;
    std::vector<std::unique_ptr<PinocchioInterface>> pinocchioInterfaces_;
    std::vector<std::unique_ptr<PinocchioEndEffectorKinematics>> endEffectorKinematics_;
    std::vector<std::unique_ptr<CentroidalModelPinocchioMapping>> centroidalModelMappings_;
    std::vector<std::unique_ptr<PinocchioCentroidalDynamics>> centroidalDynamics_;

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

    // MPC horizon in seconds
    scalar_t horizon_;

    std::vector<std::unique_ptr<ocs2::legged_robot::ModeSequenceTemplate>> modeSequenceTemplates_;
};

}  // namespace bindings
}  // namespace tbai