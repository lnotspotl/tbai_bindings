// clang-format off
#include <pinocchio/fwd.hpp>
// clang-format on

#include "tbai_bindings/TbaiIsaacGymInterface.hpp"
#include "tbai_bindings/Utils.hpp"
#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_centroidal_model/PinocchioCentroidalDynamics.h>
#include <ocs2_core/Types.h>
#include <ocs2_legged_robot/common/utils.h>
#include <ocs2_legged_robot/gait/LegLogic.h>
#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <ocs2_ros_interfaces/common/RosMsgConversions.h>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <ros/ros.h>
#include <tbai_bindings/Macros.hpp>

// This is a bit of a hack
#define protected public
#define private public
#include <ocs2_ros_interfaces/mpc/MPC_ROS_Interface.h>

using namespace ocs2;

namespace tbai {
namespace bindings {

TbaiIsaacGymInterface::TbaiIsaacGymInterface(const std::string &taskFile, const std::string &urdfFile,
                                             const std::string &referenceFile, const std::string &gaitFile,
                                             const std::string &gaitName, int numEnvs, int numThreads,
                                             torch::Device device, bool visualize)
    : numEnvs_(numEnvs), numThreads_(numThreads), threadPool_(numThreads), device_(device), visualize_(visualize) {
    // Perform simple checks
    if (numEnvs < 1) throw std::runtime_error("Number of environments must be at least 1");
    if (numThreads < 1) throw std::runtime_error("Number of threads must be at least 1");

    TBAI_BINDINGS_PRINT("Loading mode sequence template for gait" << gaitName);

    // Load mode sequence from config file
    loadModeSequenceTemplates(gaitFile, "trot");

    TBAI_BINDINGS_PRINT("Allocating interface buffers and creating interfaces");

    // Create all the necessary interfaces
    allocateInterfaceBuffers();
    createInterfaces(taskFile, urdfFile, referenceFile);

    TBAI_BINDINGS_PRINT("Allocating eigen buffers");

    // Allocate memory for buffers
    allocateEigenBuffers();

    TBAI_BINDINGS_PRINT("Allocating torch buffers");

    allocateTorchBuffers();

    TBAI_BINDINGS_PRINT("Loading MPC prediction horizon");

    // Get prediction horizon
    horizon_ = interfacePtrs_[0]->mpcSettings().timeHorizon_;

    TBAI_BINDINGS_PRINT("Horizon: " << horizon_ << " seconds");

    if (visualize_) {
        int argc = 1;
        char *argv[] = {(char *)"tbai_bindings"};
        ros::init(argc, argv, "tbai_bindings");
        ros::NodeHandle nodeHandle;
        pub_ = nodeHandle.advertise<tbai_bindings::bindings_visualize>("visualize", 1);
    }
}

void TbaiIsaacGymInterface::resetAllSolvers(scalar_t time) {
    resetSolvers(time, torch::arange(0, numEnvs_));
}

void TbaiIsaacGymInterface::loadModeSequenceTemplates(const std::string &gaitFile, const std::string &gaitName) {
    auto temp = loadModeSequenceTemplate(gaitFile, gaitName, false);  // last argument is verbose
    modeSequenceTemplate_ = std::make_unique<ModeSequenceTemplate>(temp.switchingTimes, temp.modeSequence);
}

void TbaiIsaacGymInterface::createInterfaces(const std::string &taskFile, const std::string &urdfFile,
                                             const std::string &referenceFile) {
    TBAI_BINDINGS_ASSERT(interfacePtrs_.size() == numEnvs_, "Interface pointers not allocated correctly");

    // Create a dummy interface to compile all necessary libraries
    interfacePtrs_[0] = std::make_unique<LeggedRobotInterface>(taskFile, urdfFile, referenceFile);
    interfacePtrs_[0].reset();

    // Create the rest of the interfaces
    auto createInterface = [&](size_t i) {
        // Create legged interface
        interfacePtrs_[i] = std::make_unique<LeggedRobotInterface>(taskFile, urdfFile, referenceFile);
        auto &interface = *interfacePtrs_[i];

        // Create SQP solver
        solverPtrs_[i] = std::make_unique<SqpSolver>(interface.sqpSettings(), interface.getOptimalControlProblem(),
                                                     interface.getInitializer());
        auto &solver = *solverPtrs_[i];
        solver.setReferenceManager(interface.getReferenceManagerPtr());

        // Set gait
        auto *referenceManager = dynamic_cast<SwitchedModelReferenceManager *>(&solver.getReferenceManager());
        referenceManager->getGaitSchedule()->insertModeSequenceTemplate(*modeSequenceTemplate_, -horizon_, horizon_);

        // Setup pinocchio interface
        pinocchioInterfacePtrs_[i] = std::make_unique<PinocchioInterface>(interface.getPinocchioInterface());
        auto &pinocchioInterface = *pinocchioInterfacePtrs_[i];

        // Setup centroidal model mapping
        centroidalModelMappingPtrs_[i] =
            std::make_unique<CentroidalModelPinocchioMapping>(interface.getCentroidalModelInfo());
        auto &centroidalModelMapping = *centroidalModelMappingPtrs_[i];
        centroidalModelMapping.setPinocchioInterface(pinocchioInterface);

        // Setup end effector kinematics for the feet
        endEffectorKinematicsPtrs_[i] = std::make_unique<PinocchioEndEffectorKinematics>(
            pinocchioInterface, centroidalModelMapping, interface.modelSettings().contactNames3DoF);
        auto &endEffectorKinematics = *endEffectorKinematicsPtrs_[i];
        endEffectorKinematics.setPinocchioInterface(pinocchioInterface);

        // Setup centroidal model RBD conversions
        PinocchioInterface pinint_temp(interface.getPinocchioInterface());
        centroidalModelRbdConversionsPtrs_[i] =
            std::make_unique<CentroidalModelRbdConversions>(pinint_temp, interface.getCentroidalModelInfo());
    };

    // Create all interfaces
    threadPool_.submit_loop(0, numEnvs_, createInterface).wait();
}

void TbaiIsaacGymInterface::allocateInterfaceBuffers() {
    interfacePtrs_.resize(numEnvs_);
    solverPtrs_.resize(numEnvs_);
    pinocchioInterfacePtrs_.resize(numEnvs_);
    endEffectorKinematicsPtrs_.resize(numEnvs_);
    centroidalModelMappingPtrs_.resize(numEnvs_);
    centroidalModelRbdConversionsPtrs_.resize(numEnvs_);
    solutions_.resize(numEnvs_);
}

void TbaiIsaacGymInterface::allocateEigenBuffers() {
    TBAI_BINDINGS_ASSERT(interfacePtrs_[0] != nullptr, "Interfaces not initialized. Call createInterfaces first");

    currentStatesCpu_ = matrix_t::Zero(numEnvs_, 6 + 6 + 12);  // 6 momentum, 6 base pose, 12 joint angles
    currentCommandsCpu_ = matrix_t::Zero(numEnvs_, 3);         // v_x, v_y, w_z

    desiredBasePositionsCpu_ = matrix_t::Zero(numEnvs_, 3);
    desiredBaseOrientationsCpu_ = matrix_t::Zero(numEnvs_, 4);
    desiredBaseLinearVelocitiesCpu_ = matrix_t::Zero(numEnvs_, 3);
    desiredBaseAngularVelocitiesCpu_ = matrix_t::Zero(numEnvs_, 3);
    desiredBaseLinearAccelerationsCpu_ = matrix_t::Zero(numEnvs_, 3);
    desiredBaseAngularAccelerationsCpu_ = matrix_t::Zero(numEnvs_, 3);

    desiredContactsCpu_ = matrix_t::Zero(numEnvs_, 4);       // LF, RF, LH, RH
    timeLeftInPhaseCpu_ = matrix_t::Zero(numEnvs_, 4);       // LF, RF, LH, RH
    desiredJointAnglesCpu_ = matrix_t::Zero(numEnvs_, 12);   // 3 joints per leg
    desiredFootHoldsCpu_ = matrix_t::Zero(numEnvs_, 4 * 2);  // x, y per foot
    currentDesiredJointAnglesCpu_ = matrix_t::Zero(numEnvs_, 12);
    desiredStatesCpu_ = matrix_t::Zero(numEnvs_, 12 + 12);

    initialState_ = interfacePtrs_[0]->getInitialState();
}

void TbaiIsaacGymInterface::allocateTorchBuffers() {
    optimizedStates_ = torch::empty({numEnvs_, 12 + 12}).to(device_);
    consistencyRewards_ = torch::zeros({numEnvs_}).to(device_);
    desiredContacts_ = torch::zeros({numEnvs_, 4}, torch::kBool).to(device_);
    timeLeftInPhase_ = torch::zeros({numEnvs_, 4}).to(device_);
    desiredJointAngles_ = torch::zeros({numEnvs_, 12}).to(device_);
    desiredFootholds_ = torch::zeros({numEnvs_, 4 * 2}).to(device_);
    updateInSeconds_ = torch::zeros({numEnvs_}).to(device_);
    currentDesiredJointAngles_ = torch::zeros({numEnvs_, 12}).to(device_);
    desiredBasePositions_ = torch::zeros({numEnvs_, 3}).to(device_);
    desiredBaseOrientations_ = torch::zeros({numEnvs_, 4}).to(device_);
    desiredBaseLinearVelocities_ = torch::zeros({numEnvs_, 3}).to(device_);
    desiredBaseAngularVelocities_ = torch::zeros({numEnvs_, 3}).to(device_);
    desiredBaseLinearAccelerations_ = torch::zeros({numEnvs_, 3}).to(device_);
    desiredBaseAngularAccelerations_ = torch::zeros({numEnvs_, 3}).to(device_);
}

void TbaiIsaacGymInterface::resetSolvers(scalar_t time, const torch::Tensor &envIds) {
    auto resetSolver = [&](int i) {
        int id = envIds[i].item<int>();
        solverPtrs_[id]->reset();
        solutions_[id].clear();
        dynamic_cast<SwitchedModelReferenceManager *>(&solverPtrs_[id]->getReferenceManager())
            ->getGaitSchedule()
            ->insertModeSequenceTemplate(*modeSequenceTemplate_, time - horizon_, time + horizon_);
        updateInSeconds_[id] = 0.0;
    };

    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), resetSolver).wait();
}

void TbaiIsaacGymInterface::updateCurrentStates(const torch::Tensor &newStates) {
    currentStatesCpu_ = torch2matrix(newStates.to(torch::kCPU));
}

void TbaiIsaacGymInterface::updateCurrentStates(const torch::Tensor &newStates, const torch::Tensor &envIds) {
    auto state = torch2matrix(newStates.to(torch::kCPU));
    auto updateState = [&](int i) { currentStatesCpu_.row(envIds[i].item<int>()) = state.row(i); };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), updateState).wait();
}

void TbaiIsaacGymInterface::updateDesiredContacts(scalar_t time, const torch::Tensor &envIds) {
    auto impl = [&](int i) {
        int id = envIds[i].item<int>();
        auto referenceManagerPtr = interfacePtrs_[id]->getSwitchedModelReferenceManagerPtr();
        auto contactFlags = referenceManagerPtr->getContactFlags(time);
        for (int j = 0; j < 4; ++j) {
            desiredContactsCpu_(id, j) = static_cast<scalar_t>(contactFlags[j]);
        }
    };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), impl).wait();
    desiredContacts_ = tbai::bindings::matrix2torch(desiredContactsCpu_).to(device_);
}

void TbaiIsaacGymInterface::updateTimeLeftInPhase(scalar_t time, const torch::Tensor &envIds) {
    // Make sure that updateDesiredContacts has been called before
    auto impl = [&](int i) {
        int id = envIds[i].item<int>();
        auto &solution = solutions_[id];
        auto &modeSchedule = solution.modeSchedule_;

        auto currentContacts = desiredContactsCpu_.row(id).cast<bool>();
        auto contactPhases = ocs2::legged_robot::getContactPhasePerLeg(time, modeSchedule);
        auto swingPhases = ocs2::legged_robot::getSwingPhasePerLeg(time, modeSchedule);

        for (int j = 0; j < 4; ++j) {
            if (currentContacts(j)) {
                auto legPhase = contactPhases[j];
                timeLeftInPhaseCpu_(id, j) = (1.0 - legPhase.phase) * legPhase.duration;
            } else {
                auto legPhase = swingPhases[j];
                timeLeftInPhaseCpu_(id, j) = (1.0 - legPhase.phase) * legPhase.duration;
            }

            if (std::isnan(timeLeftInPhaseCpu_(id, j))) timeLeftInPhaseCpu_(id, j) = 0.0;
        }
    };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), impl).wait();
    timeLeftInPhase_ = tbai::bindings::matrix2torch(timeLeftInPhaseCpu_).to(device_);
}

void TbaiIsaacGymInterface::updateOptimizedStates(scalar_t time) {
    updateOptimizedStates(time, torch::arange(0, numEnvs_));
}

void TbaiIsaacGymInterface::updateNextOptimizationTime(scalar_t time, const torch::Tensor &envIds) {
    auto impl = [&](int i) {
        int id = envIds[i].item<int>();
        updateNextOptimizationTimeImpl(time, id);
    };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), impl).wait();
}

void TbaiIsaacGymInterface::updateNextOptimizationTimeImpl(scalar_t time, int id) {
    auto modeSchedule = interfacePtrs_[id]->getSwitchedModelReferenceManagerPtr()->getModeSchedule();
    auto contactTimings = extractContactTimingsPerLeg(modeSchedule);

    scalar_t nextOptimizationTime = time + horizon_;

    for (int j = 0; j < 4; ++j) {
        auto contactTiming = contactTimings[j];

        auto nextSwingTime = getTimeOfNextLiftOff(time, contactTiming);
        auto nextContactTime = getTimeOfNextTouchDown(time, contactTiming);

        if (nextSwingTime < nextContactTime) {
            nextOptimizationTime = nextSwingTime;
        }

        if (nextContactTime < nextSwingTime) {
            nextOptimizationTime = nextContactTime;
        }
    }
    updateInSeconds_.index({id}) = nextOptimizationTime - time;
}

void TbaiIsaacGymInterface::updateOptimizedStates(scalar_t time, const torch::Tensor &envIds) {
    auto updateOptimizedStates = [&](int i) {
        int id = envIds[i].item<int>();
        const auto &p = this->solutions_[id];
        desiredStatesCpu_.row(id) = LinearInterpolation::interpolate(time, p.timeTrajectory_, p.stateTrajectory_);
    };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), updateOptimizedStates).wait();
    optimizedStates_ = tbai::bindings::matrix2torch(desiredStatesCpu_).to(device_);
}

void TbaiIsaacGymInterface::optimizeTrajectories(scalar_t time) {
    optimizeTrajectories(time, torch::arange(0, numEnvs_));
}

void TbaiIsaacGymInterface::optimizeTrajectories(scalar_t time, const torch::Tensor &envIds) {
    auto optimize = [&](int i) {
        int id = envIds[i].item<int>();
        const auto &interface = *(this->interfacePtrs_[id].get());
        const scalar_t initTime = time;
        const scalar_t finalTime = initTime + horizon_;
        auto initState = currentStatesCpu_.row(id);
        // auto initState = initialState_;
        // initState.segment<6>(0) = currentStatesCpu_.row(id).segment<6>(0);
        // initState.segment<6>(6) = currentStatesCpu_.row(id).segment<6>(6); // copy base position and orientation from
        // current state initState(8) = 0.54; // z position initState(10) = 0.0; // roll initState(11) = 0.0; // pitch

        // auto &prevSolution = this->solutions_[id];
        // if(prevSolution.timeTrajectory_.size() != 0) {
        //     auto prevJointAngles = ocs2::LinearInterpolation::interpolate(time, prevSolution.timeTrajectory_,
        //     prevSolution.stateTrajectory_).segment<12>(12); initState.segment<12>(12) = prevJointAngles;
        // }

        // auto targetTrajectory = TargetTrajectories({initTime}, {initState}, {vector_t::Zero(12+12)});
        auto targetTrajectory = getTargetTrajectory(time, id);
        // std::cout << "Target trajectory" << std::endl;
        // std::cout << targetTrajectory << std::endl;
        // std::cout << std::endl;
        this->solverPtrs_[id]->getReferenceManager().setTargetTrajectories(targetTrajectory);
        this->solverPtrs_[id]->run(initTime, initState, finalTime);
        PrimalSolution temp;
        this->solverPtrs_[id]->getPrimalSolution(finalTime, &temp);
        this->consistencyRewards_.index({id}) = computeConsistencyReward(temp, this->solutions_[id]);
        this->solutions_[id].swap(temp);
        temp.clear();
        updateNextOptimizationTimeImpl(time, id);
    };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), optimize).wait();
}

void TbaiIsaacGymInterface::updateCurrentDesiredJointAngles(scalar_t time, const torch::Tensor &envIds) {
    auto impl = [&](int i) {
        int id = envIds[i].item<int>();
        auto &solution = solutions_[id];
        auto &modeSchedule = solution.modeSchedule_;
        currentDesiredJointAnglesCpu_.row(i) = LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.stateTrajectory_).segment<12>(12);
    };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), impl).wait();
    currentDesiredJointAngles_ = tbai::bindings::matrix2torch(currentDesiredJointAnglesCpu_).to(device_);
}

void TbaiIsaacGymInterface::updateDesiredJointAngles(scalar_t time, const torch::Tensor &envIds) {
    // This function should only be called after updateTimeLeftInPhase
    auto impl = [&](int i) {
        int id = envIds[i].item<int>();
        auto &solution = solutions_[id];
        auto &modeSchedule = solution.modeSchedule_;
        for (int j = 0; j < 4; ++j) {
            scalar_t timeLeft = timeLeftInPhaseCpu_(id, j);

            auto state =
                LinearInterpolation::interpolate(time + timeLeft, solution.timeTrajectory_, solution.stateTrajectory_);
            auto jointAngles = state.segment<12>(12);

            // Update desired joint angles
            desiredJointAnglesCpu_.row(id).segment<3>(3 * j) = jointAngles.segment<3>(3 * j);
            torch::indexing::Slice slice(3 * j, 3 * (j + 1));

            // Compute forward kinematics
            auto &pinocchioMapping = *centroidalModelMappingPtrs_[id];
            auto &interface = *pinocchioInterfacePtrs_[id];
            auto q = pinocchioMapping.getPinocchioJointPosition(state);
            pinocchio::forwardKinematics(interface.getModel(), interface.getData(), q);
            pinocchio::updateFramePlacements(interface.getModel(), interface.getData());

            // Update end effector kinematics
            auto &endEffector = *endEffectorKinematicsPtrs_[id];
            auto positions = endEffector.getPosition(vector_t());

            // Update desired footholds
            desiredFootHoldsCpu_.row(id).segment<2>(2 * j) = positions[j].head<2>();
        }
    };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), impl).wait();
    desiredJointAngles_ = tbai::bindings::matrix2torch(desiredJointAnglesCpu_).to(device_);
    desiredFootholds_ = tbai::bindings::matrix2torch(desiredFootHoldsCpu_).to(device_);
}

const LeggedRobotInterface &TbaiIsaacGymInterface::getInterface(int i) const {
    TBAI_BINDINGS_ASSERT(i >= 0 && i < numEnvs_, "Index out of bounds");
    TBAI_BINDINGS_ASSERT(interfacePtrs_[i] != nullptr, "Interface not initialized");
    return *interfacePtrs_[i];
}

void TbaiIsaacGymInterface::setCurrentCommand(const torch::Tensor &command_tensor, const torch::Tensor &envIds) {
    auto command = torch2matrix(command_tensor.to(torch::kCPU));
    auto updateCommand = [&](int i) { currentCommandsCpu_.row(envIds[i].item<int>()) = command.row(i); };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), updateCommand).wait();
}

scalar_t TbaiIsaacGymInterface::computeConsistencyReward(const PrimalSolution &previousSolution,
                                                         const PrimalSolution &currentSolution) {
    scalar_t reward = 0.0;

    // Nothing to compute
    if (previousSolution.timeTrajectory_.size() == 0 || currentSolution.timeTrajectory_.size() == 0) {
        return reward;
    }

    // Find common time interval
    scalar_t startTime = std::max(previousSolution.timeTrajectory_.front(), currentSolution.timeTrajectory_.front());
    scalar_t endTime = std::min(previousSolution.timeTrajectory_.back(), currentSolution.timeTrajectory_.back());

    // Compute reward
    constexpr scalar_t dt = 0.01;
    scalar_t time = startTime;
    while (time < endTime) {
        vector_t previousState =
            LinearInterpolation::interpolate(time, previousSolution.timeTrajectory_, previousSolution.stateTrajectory_);
        vector_t currentState =
            LinearInterpolation::interpolate(time, currentSolution.timeTrajectory_, currentSolution.stateTrajectory_);
        reward += scalar_t(-1.0) * dt * (previousState - currentState).squaredNorm();
        time += dt;
    }

    // Return consistency reward
    return reward;
}

TargetTrajectories TbaiIsaacGymInterface::getTargetTrajectory(scalar_t initTime, int envIdx) {
    const scalar_t finalTime = initTime + horizon_;
    // auto initState = initialState_;
    auto currentState = currentStatesCpu_.row(envIdx);
    auto initState = currentState;

    auto currentCommand = currentCommandsCpu_.row(envIdx);
    const scalar_t v_x = currentCommand(0);
    const scalar_t v_y = currentCommand(1);
    const scalar_t w_z = currentCommand(2);

    auto &referenceManager = *(interfacePtrs_[envIdx]->getSwitchedModelReferenceManagerPtr());
    auto &centroidalModelInfo = interfacePtrs_[envIdx]->getCentroidalModelInfo();

    scalar_array_t timeTrajectory;
    vector_array_t stateTrajectory;
    vector_array_t inputTrajectory;

    // Insert initial time, state and input
    timeTrajectory.push_back(initTime);
    stateTrajectory.push_back(initState);
    inputTrajectory.push_back(weightCompensatingInput(centroidalModelInfo, referenceManager.getContactFlags(initTime)));

    // Complete the rest of the trajectory
    constexpr scalar_t dt = 0.1;
    scalar_t time = initTime;
    vector_t nextState = initState;
    nextState.tail<12>() = initialState_.tail<12>();
    nextState.segment<6>(0) = currentState.segment<6>(0);
    nextState.segment<6>(6) = currentState.segment<6>(6);
    nextState(8) = 0.54;  // z position
    nextState(10) = 0.0;  // roll
    nextState(11) = 0.0;  // pitch
    nextState(2) = 0.0;   // z velocity
    while (time < finalTime) {
        time += dt;

        const scalar_t yaw = nextState(9);
        const scalar_t cy = std::cos(yaw);
        const scalar_t sy = std::sin(yaw);

        const scalar_t v_x = currentCommand(0);
        const scalar_t v_y = currentCommand(1);

        const scalar_t dx = (cy * v_x - sy * v_y) * dt;
        const scalar_t dy = (sy * v_x + cy * v_y) * dt;
        const scalar_t dw = w_z * dt;

        nextState(0) = dx / dt;
        nextState(1) = dy / dt;

        nextState(6) += dx;
        nextState(7) += dy;
        nextState(9) += dw;

        timeTrajectory.push_back(time);
        stateTrajectory.push_back(nextState);
        inputTrajectory.push_back(weightCompensatingInput(centroidalModelInfo, referenceManager.getContactFlags(time)));
    }

    return TargetTrajectories(timeTrajectory, stateTrajectory, inputTrajectory);
}

PrimalSolution TbaiIsaacGymInterface::getCurrentOptimalTrajectory(int envId) const {
    return solutions_[envId];
}

SystemObservation TbaiIsaacGymInterface::getCurrentObservation(scalar_t time, int envId) const {
    vector_t currentState = currentStatesCpu_.row(envId);
    vector_t currentInput = vector_t::Zero(12 + 12);
    scalar_t currentTime = time;

    auto &solution = solutions_[envId];
    auto &modeSchedule = solution.modeSchedule_;
    size_t current_mode = modeSchedule.modeAtTime(currentTime);

    SystemObservation observation;
    observation.state = currentState;
    observation.input = currentInput;
    observation.time = currentTime;
    observation.mode = current_mode;

    return observation;
}

void TbaiIsaacGymInterface::updateDesiredBase(scalar_t time, const torch::Tensor &envIds) {
    auto impl = [&](int i) {
        int id = envIds[i].item<int>();
        auto &solution = solutions_[id];

        // compute desired MPC state and input
        vector_t desiredState =
            LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.stateTrajectory_);
        vector_t desiredInput =
            LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.inputTrajectory_);

        // Calculate desired base kinematics and dynamics
        using Vector6 = Eigen::Matrix<scalar_t, 6, 1>;
        Vector6 basePose, baseVelocity, baseAcceleration;
        vector_t jointAccelerations = vector_t::Zero(12);
        centroidalModelRbdConversionsPtrs_[id]->computeBaseKinematicsFromCentroidalModel(
            desiredState, desiredInput, jointAccelerations, basePose, baseVelocity, baseAcceleration);

        // Unpack data
        vector3_t desiredBasePosition = basePose.head<3>();
        vector3_t desiredBaseOrientation = basePose.tail<3>();  // zyx euler angles

        vector3_t desiredBaseLinearVelocity = baseVelocity.head<3>();
        vector3_t desiredBaseAngularVelocity = baseVelocity.tail<3>();

        vector3_t desiredBaseLinearAcceleration = baseAcceleration.head<3>();
        vector3_t desiredBaseAngularAcceleration = baseAcceleration.tail<3>();

        // Update desired base
        desiredBasePositionsCpu_.row(id) = desiredBasePosition;
        desiredBaseOrientationsCpu_.row(id) =
            ocs2::getQuaternionFromEulerAnglesZyx<scalar_t>(desiredBaseOrientation).coeffs();
        desiredBaseLinearVelocitiesCpu_.row(id) = desiredBaseLinearVelocity;
        desiredBaseAngularVelocitiesCpu_.row(id) = desiredBaseAngularVelocity;
        desiredBaseLinearAccelerationsCpu_.row(id) = desiredBaseLinearAcceleration;
        desiredBaseAngularAccelerationsCpu_.row(id) = desiredBaseAngularAcceleration;
    };

    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), impl).wait();
}

void TbaiIsaacGymInterface::moveDesiredBaseToGpu() {
    desiredBasePositions_ = tbai::bindings::matrix2torch(desiredBasePositionsCpu_).to(device_);
    desiredBaseOrientations_ = tbai::bindings::matrix2torch(desiredBaseOrientationsCpu_).to(device_);
    desiredBaseLinearVelocities_ = tbai::bindings::matrix2torch(desiredBaseLinearVelocitiesCpu_).to(device_);
    desiredBaseAngularVelocities_ = tbai::bindings::matrix2torch(desiredBaseAngularVelocitiesCpu_).to(device_);
    desiredBaseLinearAccelerations_ = tbai::bindings::matrix2torch(desiredBaseLinearAccelerationsCpu_).to(device_);
    desiredBaseAngularAccelerations_ = tbai::bindings::matrix2torch(desiredBaseAngularAccelerationsCpu_).to(device_);
}

void TbaiIsaacGymInterface::visualize(scalar_t time, torch::Tensor &state, int envId, torch::Tensor &obs) {
    if (!visualize_) {
        TBAI_BINDINGS_PRINT("Visualize is disabled!");
        return;
    }

    if (!ros::ok()) return;

    // Get solution
    auto &solution = solutions_[envId];
    auto &perfIndices = solverPtrs_[envId]->getPerformanceIndeces();

    // Generate system observation
    SystemObservation observation;
    observation.state = torch2vector(state);
    observation.input = LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.inputTrajectory_);
    observation.time = time;
    CommandData commandData = {observation, solverPtrs_[envId]->getReferenceManager().getTargetTrajectories()};

    tbai_bindings::bindings_visualize msg;
    msg.observation = ocs2::ros_msg_conversions::createObservationMsg(observation);
    auto &targetTrajectory = solverPtrs_[envId]->getReferenceManager().getTargetTrajectories();
    msg.target_trajectories = ocs2::ros_msg_conversions::createTargetTrajectoriesMsg(targetTrajectory);
    msg.flattened_controller = MPC_ROS_Interface::createMpcPolicyMsg(solution, commandData, perfIndices);

    // Copy obs to msg
    auto obsCpu = tbai::bindings::torch2vector(obs.to(torch::kCPU));
    std::copy(obsCpu.data(), obsCpu.data() + obsCpu.size(), std::back_inserter(msg.obs));

    pub_.publish(msg);
}

}  // namespace bindings
}  // namespace tbai