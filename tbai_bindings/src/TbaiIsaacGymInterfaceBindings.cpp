#include <pybind11/pybind11.h>
#include <tbai_bindings/TbaiIsaacGymInterface.hpp>
#include <torch/extension.h>
#include <torch/torch.h>

using namespace pybind11::literals;

PYBIND11_MODULE(tbai_ocs2_interface, m) {
    using namespace tbai::bindings;

    m.doc() = "Interface for ocs2's trajectory optimization module for legged robots.";

    py::class_<TbaiIsaacGymInterface>(m, "TbaiIsaacGymInterface")
        .def(py::init<const std::string &, const std::string &, const std::string &, const std::string &,
                      const std::string &, int, int, torch::Device, bool>(),
             "taskFile"_a, "urdfFile"_a, "referenceFile"_a, "gaitFile"_a, "gaitName"_a, "numEnvs"_a, "numThreads"_a,
             "device"_a, "visualize"_a)
        .def("reset_solvers", &TbaiIsaacGymInterface::resetSolvers, "time"_a, "envIds"_a)
        .def("reset_all_solvers", &TbaiIsaacGymInterface::resetAllSolvers, "time"_a)
        .def("optimize_trajectories", py::overload_cast<scalar_t>(&TbaiIsaacGymInterface::optimizeTrajectories))
        .def("optimize_trajectories",
             py::overload_cast<scalar_t, const torch::Tensor &>(&TbaiIsaacGymInterface::optimizeTrajectories))
        .def("update_optimized_states", py::overload_cast<scalar_t>(&TbaiIsaacGymInterface::updateOptimizedStates))
        .def("update_optimized_states",
             py::overload_cast<scalar_t, const torch::Tensor &>(&TbaiIsaacGymInterface::updateOptimizedStates))
        .def("get_optimized_states", &TbaiIsaacGymInterface::getOptimizedStates)
        .def("set_current_command", &TbaiIsaacGymInterface::setCurrentCommand)
        .def("update_desired_contacts", &TbaiIsaacGymInterface::updateDesiredContacts)
        .def("update_time_left_in_phase", &TbaiIsaacGymInterface::updateTimeLeftInPhase)
        .def("update_desired_joint_angles", &TbaiIsaacGymInterface::updateDesiredJointAnglesPerceptive)
        .def("updated_in_seconds", &TbaiIsaacGymInterface::getUpdatedInSeconds)
        .def("get_consistency_reward", &TbaiIsaacGymInterface::getConsistencyReward)
        .def("get_planar_footholds", &TbaiIsaacGymInterface::getPlanarFootHolds)
        .def("get_desired_joint_positions", &TbaiIsaacGymInterface::getDesiredJointPositions)
        .def("get_desired_contacts", &TbaiIsaacGymInterface::getDesiredContacts)
        .def("get_time_left_in_phase", &TbaiIsaacGymInterface::getTimeLeftInPhase)
        .def("update_current_desired_joint_angles", &TbaiIsaacGymInterface::updateCurrentDesiredJointAnglesPerceptive)
        .def("get_current_desired_joint_positions", &TbaiIsaacGymInterface::getCurrentDesiredJointPositions)
        .def("get_desired_base_positions", &TbaiIsaacGymInterface::getDesiredBasePositions)
        .def("get_desired_base_orientations", &TbaiIsaacGymInterface::getDesiredBaseOrientations)
        .def("get_desired_base_linear_velocities", &TbaiIsaacGymInterface::getDesiredBaseLinearVelocities)
        .def("get_desired_base_angular_velocities", &TbaiIsaacGymInterface::getDesiredBaseAngularVelocities)
        .def("get_desired_base_linear_accelerations", &TbaiIsaacGymInterface::getDesiredBaseLinearAccelerations)
        .def("get_desired_base_angular_accelerations", &TbaiIsaacGymInterface::getDesiredBaseAngularAccelerations)
        .def("update_desired_base", &TbaiIsaacGymInterface::updateDesiredBasePerceptive)
        .def("move_desired_base_to_gpu", &TbaiIsaacGymInterface::moveDesiredBaseToGpu)
        .def("visualize", &TbaiIsaacGymInterface::visualize, "time"_a, "state"_a, "envId"_a, "obs"_a)
        .def("get_bobnet_phases", &TbaiIsaacGymInterface::getBobnetPhases)
        .def("update_desired_foot_positions_and_velocities",
             &TbaiIsaacGymInterface::updateDesiredFootPositionsAndVelocitiesPerceptive)
        .def("get_desired_foot_positions", &TbaiIsaacGymInterface::getDesiredFootPositions)
        .def("get_desired_foot_velocities", &TbaiIsaacGymInterface::getDesiredFootVelocities)
        .def("set_map_from_flattened", &TbaiIsaacGymInterface::setMapFromFlattened)
        .def("set_maps_from_flattened", &TbaiIsaacGymInterface::setMapsFromFlattened)
        .def("update_states_perceptive", &TbaiIsaacGymInterface::updateCurrentStatesPerceptive);

    // Helper class
    py::class_<SystemObservation>(m, "SystemObservation");

    // Helper class
    py::class_<PrimalSolution>(m, "PrimalSolution");

    // Helper class
    py::class_<LeggedRobotInterface>(m, "LeggedRobotInterface");
}