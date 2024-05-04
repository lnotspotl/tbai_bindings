#include <iostream>
#include <string>
#include <tbai_bindings/TbaiIsaacGymInterface.hpp>

int main(int argc, char *argv[]) {

    std::cout << "Beginning testing of tbai_bindings" << std::endl;

    std::cout << "Setting up interface parameters" << std::endl;
    std::string taskFile = "/home/kuba/fun/tbai_bindings/src/tbai_bindings/dependencies/ocs2/ocs2_robotic_examples/ocs2_legged_robot/config/mpc/task.info";
    std::string urdfFile = "/home/kuba/fun/tbai_bindings/src/tbai_bindings/dependencies/ocs2_robotic_assets/resources/anymal_d/urdf/anymal.urdf";
    std::string referenceFile = "/home/kuba/fun/tbai_bindings/src/tbai_bindings/dependencies/ocs2/ocs2_robotic_examples/ocs2_legged_robot/config/command/reference.info";
    std::string gaitFile = "/home/kuba/fun/tbai_bindings/src/tbai_bindings/dependencies/ocs2/ocs2_robotic_examples/ocs2_legged_robot/config/command/gait.info";
    std::string gaitName = "trot";
    int numEnvs = 3;
    int numThreads = 2;

    std::cout << "Creating tbaiIsaacGymInterface object" << std::endl;
    tbai::bindings::TbaiIsaacGymInterface tbaiIsaacGymInterface(taskFile, urdfFile, referenceFile, gaitFile, gaitName, numEnvs, numThreads);

    return 0;
}