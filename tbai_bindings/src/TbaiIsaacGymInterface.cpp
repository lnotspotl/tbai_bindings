// clang-format off
#include <pinocchio/fwd.hpp>
// clang-format on
// This is a bit of a hack
#define protected public
#define private public

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

    if (visualize_) {
        int argc = 1;
        char *argv[] = {(char *)"tbai_bindings"};
        ros::init(argc, argv, "tbai_bindings");
        ros::NodeHandle nodeHandle;
        pub_ = nodeHandle.advertise<tbai_bindings::bindings_visualize>("visualize", 1);
        // gridmapInterfacePtr_ = std::make_unique<GridmapInterface>(true);
    }
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
    horizon_ = 1.0;

    TBAI_BINDINGS_PRINT("Horizon: " << horizon_ << " seconds");
}

void TbaiIsaacGymInterface::updateCurrentStatesPerceptive(const torch::Tensor &newStates, const torch::Tensor &envIds) {
    auto state = torch2matrix(newStates.to(torch::kCPU));
    auto updateImpl = [&](int i) {
        int id = envIds[i].item<int>();
        vector_t ocs2State = state.row(i);
        vector_t zyx_euler = ocs2State.segment<3>(0).reverse();

        // Convert to xyz euler
        vector_t xyz_ocs2_euler = tbai::bindings::mat2oc2rpy(tbai::bindings::rpy2quat(zyx_euler).toRotationMatrix(), lastYawCpu_[id]);
        lastYawCpu_[id] = xyz_ocs2_euler(2);

        vector_t position = ocs2State.segment<3>(3);
        vector_t localAngularVelocity = ocs2State.segment<3>(6);
        vector_t localLinearVelocity = ocs2State.segment<3>(9);
        vector_t jointAngles = ocs2State.segment<12>(12);
        vector_t jointVelocities = vector_t::Zero(12);

        // Flip FH and RF
        auto flipFHRF = [](vector_t &v) {
            std::swap(v(3 + 0), v(3 + 3));
            std::swap(v(3 + 1), v(3 + 4));
            std::swap(v(3 + 2), v(3 + 5));
        };
        flipFHRF(jointAngles);
        flipFHRF(jointVelocities);

        vector_t currentState = (vector_t(6 + 6 + 12 + 12) << xyz_ocs2_euler, position, localAngularVelocity,
                                 localLinearVelocity, jointAngles, jointVelocities)
                                    .finished();

        currentStatesPerceptiveCpu_.row(id) = currentState;
    };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), updateImpl).wait();
}

void TbaiIsaacGymInterface::resetAllSolvers(scalar_t time) {
    resetSolvers(time, torch::arange(0, numEnvs_));
}

void TbaiIsaacGymInterface::loadModeSequenceTemplates(const std::string &gaitFile, const std::string &gaitName) {
    auto temp = switched_model::loadModeSequenceTemplate(gaitFile, gaitName, false);  // last argument is verbose
    modeSequenceTemplate_ =
        std::make_unique<switched_model::ModeSequenceTemplate>(temp.switchingTimes, temp.modeSequence);
}

void TbaiIsaacGymInterface::createInterfaces(const std::string &taskFile, const std::string &urdfFile,
                                             const std::string &referenceFile) {
    TBAI_BINDINGS_ASSERT(quadrupedInterfacePtrs_.size() == numEnvs_, "Interface pointers not allocated correctly");

    quadrupedInterfacePtrs_[0] = createQuadrupedInterface(urdfFile, taskFile);
    quadrupedInterfacePtrs_[0].reset();

    // Create the rest of the interfaces
    auto createInterface = [&](size_t i) {
        // Setup gridmap interface
        bool visualize = (i == 0 && visualize_);
        gridmapInterfacesPtrs_[i] = std::make_unique<GridmapInterface>(visualize);
        quadrupedInterfacePtrs_[i] = createQuadrupedInterface(urdfFile, taskFile);

        // Quadruped interface
        auto &quadrupedInterface = *quadrupedInterfacePtrs_[i];
        comModelPtrs_[i].reset(quadrupedInterface.getComModel().clone());
        kinematicsPtrs_[i].reset(quadrupedInterface.getKinematicModel().clone());

        // Create SQP solver
        const std::string sqpPath = taskFile + "/multiple_shooting.info";
        const auto sqpSettings = ocs2::sqp::loadSettings(sqpPath);
        quadrupedSolverPtrs_[i] = std::make_unique<SqpSolver>(
            sqpSettings, quadrupedInterface.getOptimalControlProblem(), quadrupedInterface.getInitializer());
        auto &quadrupedSolver = *quadrupedSolverPtrs_[i];
        quadrupedSolver.setReferenceManager(quadrupedInterface.getReferenceManagerPtr());
        quadrupedSolver.setSynchronizedModules(quadrupedInterface.getSynchronizedModules());

        // Set gait
        auto *quadrupedReference =
            dynamic_cast<switched_model::SwitchedModelModeScheduleManager *>(&quadrupedSolver.getReferenceManager());
        quadrupedReference->getGaitSchedule().lock()->setGaitAtTime(switched_model::toGait(*modeSequenceTemplate_),
                                                                    0.0);
    };
    gait_ = switched_model::toGait(*modeSequenceTemplate_);

    // Create all interfaces
    threadPool_.submit_loop(0, numEnvs_, createInterface).wait();
}

void TbaiIsaacGymInterface::allocateInterfaceBuffers() {
    solutionsPerceptive_.resize(numEnvs_);
    gridmapInterfacesPtrs_.resize(numEnvs_);
    quadrupedInterfacePtrs_.resize(numEnvs_);
    quadrupedSolverPtrs_.resize(numEnvs_);
    comModelPtrs_.resize(numEnvs_);
    kinematicsPtrs_.resize(numEnvs_);
}

// self.tbai_ocs2_interface.set_maps_from_flattened(flattened_maps, length_x, length_y, resolution, x_coords, y_coords)
void TbaiIsaacGymInterface::setMapsFromFlattened(const torch::Tensor &flattenedMaps, scalar_t lengthX, scalar_t lengthY,
                                                 scalar_t resolution, const torch::Tensor &xCoords,
                                                 const torch::Tensor &yCoords, const torch::Tensor &envIds) {
    TBAI_BINDINGS_ASSERT(flattenedMaps.size(0) == envIds.numel(), "Flattened maps and yCoords must have the same size");

    auto updateImpl = [&](int i) {
        int id = envIds[i].item<int>();
        float x = xCoords[i].item<float>();
        float y = yCoords[i].item<float>();
        Eigen::VectorXf heights =
            Eigen::Map<Eigen::VectorXf>(flattenedMaps[i].data_ptr<float>(), flattenedMaps.size(1));

        gridmapInterfacesPtrs_[id]->updateFromFlattened(heights, lengthX, lengthY, resolution, x, y);

        auto t1 = std::chrono::high_resolution_clock::now();
        gridmapInterfacesPtrs_[id]->computeSegmentedPlanes();
        auto t2 = std::chrono::high_resolution_clock::now();

        // Set map for the quadruped solver
        auto &quadrupedSolver = *quadrupedSolverPtrs_[id];
        auto *quadrupedReference =
            dynamic_cast<switched_model::SwitchedModelModeScheduleManager *>(&quadrupedSolver.getReferenceManager());
        auto &planarTerrain = gridmapInterfacesPtrs_[id]->getPlanarTerrain();
        std::unique_ptr<switched_model::TerrainModel> terrain =
            std::make_unique<switched_model::SegmentedPlanesTerrainModel>(planarTerrain);
        quadrupedReference->getTerrainModel().swap(terrain);

    };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), updateImpl).wait();
}

void TbaiIsaacGymInterface::allocateEigenBuffers() {
    TBAI_BINDINGS_ASSERT(quadrupedInterfacePtrs_[0] != nullptr,
                         "Interfaces not initialized. Call createInterfaces first");

    currentStatesCpu_ = matrix_t::Zero(numEnvs_, 6 + 6 + 12 + 12);  // 6 momentum, 6 base pose, 12 joint angles
    currentStatesPerceptiveCpu_ =
        matrix_t::Zero(numEnvs_, 6 + 6 + 12 + 12);  // xyz euler angles, positoon, local angular velocity, local linear
                                                    // velocity, joint angles, joint velocities
    lastYawCpu_ = vector_t::Zero(numEnvs_);
    currentCommandsCpu_ = matrix_t::Zero(numEnvs_, 3);  // v_x, v_y, w_z

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

    desiredFootPositionsCpu_ = matrix_t::Zero(numEnvs_, 4 * 3);   // x, y, z per foot
    desiredFootVelocitiesCpu_ = matrix_t::Zero(numEnvs_, 4 * 3);  // x, y, z per foot

    initialState_ = quadrupedInterfacePtrs_[0]->getInitialState();
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
        solutionsPerceptive_[id].clear();
        quadrupedSolverPtrs_[id]->reset();
        dynamic_cast<switched_model::SwitchedModelModeScheduleManager *>(
            &quadrupedSolverPtrs_[id]->getReferenceManager())
            ->getGaitSchedule()
            .lock()
            ->setGaitAtTime(switched_model::toGait(*modeSequenceTemplate_), time);
        updateInSeconds_[id] = 0.0;
        lastYawCpu_[id] = 0.0;
    };

    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), resetSolver).wait();
}

void TbaiIsaacGymInterface::updateDesiredContacts(scalar_t time, const torch::Tensor &envIds) {
    auto impl = [&](int i) {
        int id = envIds[i].item<int>();
        auto referenceManagerPtr = quadrupedInterfacePtrs_[id]->getSwitchedModelModeScheduleManagerPtr();
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
        quadrupedInterfacePtrs_[id]->getSwitchedModelModeScheduleManagerPtr()->getGaitSchedule()->advanceToTime(time);
        scalar_t phase =
            quadrupedInterfacePtrs_[id]->getSwitchedModelModeScheduleManagerPtr()->getGaitSchedule()->getCurrentPhase();
        scalar_t timeLeftInMode = switched_model::timeLeftInMode(phase, gait_);

        for (int j = 0; j < 4; ++j) {
            timeLeftInPhaseCpu_(id, j) = timeLeftInMode;
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
    scalar_t phase =
        quadrupedInterfacePtrs_[id]->getSwitchedModelModeScheduleManagerPtr()->getGaitSchedule()->getCurrentPhase();
    scalar_t timeLeftInMode = switched_model::timeLeftInMode(phase, gait_);
    updateInSeconds_.index({id}) = timeLeftInMode;
}

void TbaiIsaacGymInterface::updateOptimizedStates(scalar_t time, const torch::Tensor &envIds) {
    auto updateOptimizedStates = [&](int i) {
        int id = envIds[i].item<int>();
        const auto &p = this->solutionsPerceptive_[id];
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
        const scalar_t initTime = time;
        const scalar_t finalTime = initTime + horizon_;

        // Optimize perceptive
        auto &quadrupedInterface = *quadrupedInterfacePtrs_[id];
        auto initStatePerceptive = currentStatesPerceptiveCpu_.row(id).head<6 + 6 + 12>();
        auto targetTrajectoryPerceptive = getTargetTrajectoryPerceptive(time, id);
        quadrupedSolverPtrs_[id]->getReferenceManager().setTargetTrajectories(targetTrajectoryPerceptive);
        auto t1 = std::chrono::high_resolution_clock::now();
        quadrupedSolverPtrs_[id]->run(initTime, initStatePerceptive, finalTime);
        auto t2 = std::chrono::high_resolution_clock::now();
        PrimalSolution tempPerceptive;
        quadrupedSolverPtrs_[id]->getPrimalSolution(finalTime, &tempPerceptive);
        scalar_t consistencyRewardPerceptive = computeConsistencyReward(tempPerceptive, solutionsPerceptive_[id]);
        this->consistencyRewards_.index({id}) = consistencyRewardPerceptive;
        solutionsPerceptive_[id].swap(tempPerceptive);
        tempPerceptive.clear();

        updateNextOptimizationTimeImpl(time, id);
    };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), optimize).wait();
}

void TbaiIsaacGymInterface::updateCurrentDesiredJointAnglesPerceptive(scalar_t time, const torch::Tensor &envIds) {
    auto impl = [&](int i) {
        int id = envIds[i].item<int>();
        auto &solution = solutionsPerceptive_[id];
        auto modeSchedule =
            quadrupedInterfacePtrs_[id]->getSwitchedModelModeScheduleManagerPtr()->getGaitSchedule()->getModeSchedule(
                horizon_);

        auto state = LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.stateTrajectory_);
        auto *quadcomPtr = dynamic_cast<anymal::QuadrupedCom *>(comModelPtrs_[id].get());
        auto &quadcom = *quadcomPtr;
        auto &kin = *kinematicsPtrs_[id];

        auto basePoseOcs2 = switched_model::getBasePose(state);
        auto jointAnglesOcs2 = switched_model::getJointPositions(state);

        auto qPinocchio = quadcom.getPinnochioConfiguration(basePoseOcs2, jointAnglesOcs2);
        auto jointAngles = qPinocchio.tail<12>();
        currentDesiredJointAnglesCpu_.row(i) = jointAngles;
    };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), impl).wait();
    currentDesiredJointAngles_ = tbai::bindings::matrix2torch(currentDesiredJointAnglesCpu_).to(device_);
}

void TbaiIsaacGymInterface::updateDesiredJointAnglesPerceptive(scalar_t time, const torch::Tensor &envIds) {
    auto impl = [&](int i) {
        int id = envIds[i].item<int>();
        auto &solution = solutionsPerceptive_[id];
        auto modeSchedule =
            quadrupedInterfacePtrs_[id]->getSwitchedModelModeScheduleManagerPtr()->getGaitSchedule()->getModeSchedule(
                horizon_);
        for (int j = 0; j < 4; ++j) {
            scalar_t timeLeft = timeLeftInPhaseCpu_(id, j);

            auto state =
                LinearInterpolation::interpolate(time + timeLeft, solution.timeTrajectory_, solution.stateTrajectory_);

            auto *quadcomPtr = dynamic_cast<anymal::QuadrupedCom *>(comModelPtrs_[id].get());
            auto &quadcom = *quadcomPtr;
            auto &kin = *kinematicsPtrs_[id];

            auto basePoseOcs2 = switched_model::getBasePose(state);
            auto jointAnglesOcs2 = switched_model::getJointPositions(state);

            auto qPinocchio = quadcom.getPinnochioConfiguration(basePoseOcs2, jointAnglesOcs2);
            auto jointAngles = qPinocchio.tail<12>();

            // Update desired joint angles
            desiredJointAnglesCpu_.row(id).segment<3>(3 * j) = jointAngles.segment<3>(3 * j);

            auto position = kin.footPositionInOriginFrame(j, basePoseOcs2, jointAnglesOcs2);

            // Update desired footholds
            desiredFootHoldsCpu_.row(id).segment<2>(2 * j) = position.head<2>();
        }
    };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), impl).wait();
    desiredJointAngles_ = tbai::bindings::matrix2torch(desiredJointAnglesCpu_).to(device_);
    desiredFootholds_ = tbai::bindings::matrix2torch(desiredFootHoldsCpu_).to(device_);
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

TargetTrajectories TbaiIsaacGymInterface::getTargetTrajectoryPerceptive(scalar_t initTime, int envIdx) {
    switched_model::BaseReferenceTrajectory baseReferenceTrajectory = generateExtrapolatedBaseReference(
        getBaseReferenceHorizon(initTime, envIdx), getBaseReferenceState(initTime, envIdx),
        getBaseReferenceCommand(initTime, envIdx), gridmapInterfacesPtrs_[envIdx]->getPlanarTerrain().gridMap, 0.53,
        0.3);

    constexpr size_t STATE_DIM = 6 + 6 + 12;
    constexpr size_t INPUT_DIM = 12 + 12;

    // Generate target trajectory
    ocs2::scalar_array_t desiredTimeTrajectory = std::move(baseReferenceTrajectory.time);
    const size_t N = desiredTimeTrajectory.size();
    ocs2::vector_array_t desiredStateTrajectory(N);
    ocs2::vector_array_t desiredInputTrajectory(N, ocs2::vector_t::Zero(INPUT_DIM));
    for (size_t i = 0; i < N; ++i) {
        ocs2::vector_t state = ocs2::vector_t::Zero(STATE_DIM);

        // base orientation
        state.head<3>() = baseReferenceTrajectory.eulerXyz[i];

        auto Rt = switched_model::rotationMatrixOriginToBase(baseReferenceTrajectory.eulerXyz[i]);

        // base position
        state.segment<3>(3) = baseReferenceTrajectory.positionInWorld[i];

        // base angular velocity
        state.segment<3>(6) = Rt * baseReferenceTrajectory.angularVelocityInWorld[i];

        // base linear velocity
        state.segment<3>(9) = Rt * baseReferenceTrajectory.linearVelocityInWorld[i];

        // joint angles
        state.segment<12>(12) = quadrupedInterfacePtrs_[envIdx]->getInitialState().segment<12>(12);

        desiredStateTrajectory[i] = std::move(state);
    }

    return TargetTrajectories(std::move(desiredTimeTrajectory), std::move(desiredStateTrajectory),
                              std::move(desiredInputTrajectory));
}

void TbaiIsaacGymInterface::updateDesiredBasePerceptive(scalar_t time, const torch::Tensor &envIds) {
    auto impl = [&](int i) {
        int id = envIds[i].item<int>();
        auto &solution = solutionsPerceptive_[id];

        // compute desired MPC state and input
        vector_t desiredState =
            LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.stateTrajectory_);
        vector_t desiredInput =
            LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.inputTrajectory_);

        auto *quadcomPtr = dynamic_cast<anymal::QuadrupedCom *>(comModelPtrs_[id].get());
        auto &quadcom = *quadcomPtr;
        auto &kin = *kinematicsPtrs_[id];

        auto basePoseOcs2 = switched_model::getBasePose(desiredState);
        auto jointAnglesOcs2 = switched_model::getJointPositions(desiredState);
        auto baseVelocityOcs2 = switched_model::getBaseLocalVelocities(desiredState);
        auto jointVelocitiesOcs2 = switched_model::getJointVelocities(desiredState);

        auto qPinocchio = quadcom.getPinnochioConfiguration(basePoseOcs2, jointAnglesOcs2);
        auto vPinocchio = quadcom.getPinnochioVelocity(baseVelocityOcs2, jointVelocitiesOcs2);

        // Desired base orinetation as a quaternion
        quaternion_t desiredBaseOrientationQuat = ocs2rpy2quat(basePoseOcs2.head<3>());  // ocs2 xyz to quaternion
        matrix3_t rotationWorldBase = desiredBaseOrientationQuat.toRotationMatrix();

        const vector_t &basePosett = desiredState.head<6>();
        const vector_t &baseVelocitytt = desiredState.segment<6>(6);
        const vector_t &jointPositionstt = desiredState.tail<switched_model::JOINT_COORDINATE_SIZE>();
        const vector_t &jointVelocitiestt = desiredInput.tail<switched_model::JOINT_COORDINATE_SIZE>();
        const vector_t &jointAccelerationstt = vector_t::Zero(switched_model::JOINT_COORDINATE_SIZE);

        // forcesOnBaseInBaseFrame = [torque (3); force (3)]
        vector_t forcesOnBaseInBaseFrame = vector_t::Zero(6);
        for (size_t i = 0; i < 4; ++i) {
            // force at foot expressed in base frame
            const vector3_t &forceAtFoot = desiredInput.segment<3>(3 * i);

            // base force
            forcesOnBaseInBaseFrame.tail<3>() += forceAtFoot;

            // base torque
            vector3_t footPosition = kin.positionBaseToFootInBaseFrame(i, jointPositionstt);
            forcesOnBaseInBaseFrame.head<3>() += footPosition.cross(forceAtFoot);
        }

        vector_t baseAccelerationLocal =
            quadcom.calculateBaseLocalAccelerations(basePosett, baseVelocitytt, jointPositionstt, jointVelocitiestt,
                                                    jointAccelerationstt, forcesOnBaseInBaseFrame);

        // Unpack data
        vector3_t desiredBasePosition = basePoseOcs2.tail<3>();
        vector_t desiredBaseOrientation = desiredBaseOrientationQuat.coeffs();  // zyx euler angles

        vector3_t desiredBaseLinearVelocity = rotationWorldBase * baseVelocityOcs2.tail<3>();
        vector3_t desiredBaseAngularVelocity = rotationWorldBase * baseVelocityOcs2.head<3>();

        vector3_t desiredBaseLinearAcceleration = rotationWorldBase * baseAccelerationLocal.tail<3>();
        vector3_t desiredBaseAngularAcceleration = rotationWorldBase * baseAccelerationLocal.head<3>();

        // Update desired base
        desiredBasePositionsCpu_.row(id) = desiredBasePosition;
        desiredBaseOrientationsCpu_.row(id) = desiredBaseOrientation;
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
    auto &solution = solutionsPerceptive_[envId];
    auto &perfIndices = quadrupedSolverPtrs_[envId]->getPerformanceIndeces();

    vector_t stateCpu = torch2vector(state);
    vector3_t zyx_euler = stateCpu.segment<3>(0).reverse();
    stateCpu.segment<3>(0) = tbai::bindings::mat2oc2rpy(tbai::bindings::rpy2quat(zyx_euler).toRotationMatrix(), lastYawCpu_[envId]);

    // Generate system observation
    SystemObservation observation;
    observation.state = stateCpu;
    observation.input = LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.inputTrajectory_);
    observation.time = time;
    CommandData commandData = {observation, quadrupedSolverPtrs_[envId]->getReferenceManager().getTargetTrajectories()};

    tbai_bindings::bindings_visualize msg;
    msg.observation = ocs2::ros_msg_conversions::createObservationMsg(observation);
    auto &targetTrajectory = quadrupedSolverPtrs_[envId]->getReferenceManager().getTargetTrajectories();
    msg.target_trajectories = ocs2::ros_msg_conversions::createTargetTrajectoriesMsg(targetTrajectory);
    msg.flattened_controller = MPC_ROS_Interface::createMpcPolicyMsg(solution, commandData, perfIndices);

    // Copy obs to msg
    auto obsCpu = tbai::bindings::torch2vector(obs.to(torch::kCPU));
    std::copy(obsCpu.data(), obsCpu.data() + obsCpu.size(), std::back_inserter(msg.obs));

    pub_.publish(msg);

    gridmapInterfacesPtrs_[envId]->computeSegmentedPlanes();
    auto &planarTerrain = gridmapInterfacesPtrs_[envId]->getPlanarTerrain();
    gridmapInterfacesPtrs_[envId]->visualizePlanarTerrain(planarTerrain);
}

void TbaiIsaacGymInterface::setMapFromFlattened(const torch::Tensor &flattenedMap, scalar_t length_x, scalar_t length_y,
                                                scalar_t resolution, scalar_t x, scalar_t y) {
    Eigen::VectorXf heights = tbai::bindings::torch2vector(flattenedMap).cast<float>();
    const size_t Nx = length_x / resolution;
    const size_t Ny = length_y / resolution;
    Eigen::MatrixXf mapMatrix = Eigen::Map<Eigen::MatrixXf>(heights.data(), Nx, Ny);
    auto &map = gridmapInterfacePtr_->getMap();
    grid_map::Length length(length_x, length_y);
    grid_map::Position position(x, y);
    map.setGeometry(length, resolution, position);
    map.setFrameId("odom");
    map.add("elevation", mapMatrix);
}

void TbaiIsaacGymInterface::updateDesiredFootPositionsAndVelocitiesPerceptive(scalar_t time,
                                                                              const torch::Tensor &envIds) {
    auto impl = [&](int i) {
        int id = envIds[i].item<int>();
        auto &solution = solutionsPerceptive_[id];
        auto modeSchedule =
            quadrupedInterfacePtrs_[id]->getSwitchedModelModeScheduleManagerPtr()->getGaitSchedule()->getModeSchedule(
                horizon_);
        auto *quadcomPtr = dynamic_cast<anymal::QuadrupedCom *>(comModelPtrs_[id].get());
        auto &quadcom = *quadcomPtr;
        auto &kin = *kinematicsPtrs_[id];
        auto state = LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.stateTrajectory_);
        auto basePoseOcs2 = switched_model::getBasePose(state);
        auto baseTwistOcs2 = switched_model::getBaseLocalVelocities(state);
        auto jointAnglesOcs2 = switched_model::getJointPositions(state);
        auto jointVelocitiesOcs2 = switched_model::getJointVelocities(state);

        for (int legidx = 0; legidx < 4; ++legidx) {
            auto position = kin.footPositionInOriginFrame(legidx, basePoseOcs2, jointAnglesOcs2);
            auto velocity = kin.footVelocityInOriginFrame(legidx, basePoseOcs2, baseTwistOcs2, jointAnglesOcs2,
                                                          jointVelocitiesOcs2);

            desiredFootPositionsCpu_.row(id).segment<3>(3 * legidx) = position;
            desiredFootVelocitiesCpu_.row(id).segment<3>(3 * legidx) = velocity;
        }
    };

    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), impl).wait();
    desiredFootPositions_ = tbai::bindings::matrix2torch(desiredFootPositionsCpu_).to(device_);
    desiredFootVelocities_ = tbai::bindings::matrix2torch(desiredFootVelocitiesCpu_).to(device_);
}

torch::Tensor TbaiIsaacGymInterface::getBobnetPhases(scalar_t time, const torch::Tensor &envIds) {
    // Make sure that updateDesiredContacts has been called before
    matrix_t phases = matrix_t::Zero(envIds.numel(), 4);
    auto impl = [&](int i) {
        int id = envIds[i].item<int>();
        scalar_t phase =
            quadrupedInterfacePtrs_[id]->getSwitchedModelModeScheduleManagerPtr()->getGaitSchedule()->getCurrentPhase();
        // LH, RF - phase in [0, PI]
        // LF, RH - phase in [PI, 2*PI]
        // Basically when LF lifts off the phase is 0
        constexpr scalar_t PI = 3.14159265358979323846;
        phases(id, 0) = phase * 2 * PI;       // LF
        phases(id, 1) = phase * 2 * PI + PI;  // LH
        phases(id, 2) = phase * 2 * PI + PI;  // RF
        phases(id, 3) = phase * 2 * PI;       // RH

        for (int j = 0; j < 4; ++j) {
            if (phases(id, j) > 2 * PI) phases(i, j) -= 2 * PI;
            if (phases(id, j) < 0) phases(i, j) += 2 * PI;
        }
    };
    threadPool_.submit_loop(0, static_cast<int>(envIds.numel()), impl).wait();
    return tbai::bindings::matrix2torch(phases).to(device_);
}

}  // namespace bindings
}  // namespace tbai