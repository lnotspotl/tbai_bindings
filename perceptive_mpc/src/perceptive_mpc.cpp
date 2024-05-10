#include "perceptive_mpc/perceptive_mpc.hpp"

std::string loadUrdf(const std::string &urdfFile) {
    std::ifstream stream(urdfFile.c_str());
    if (!stream) {
        throw std::runtime_error("File " + urdfFile + " does not exist");
    }

    std::string xml_str((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    return xml_str;
}

std::unique_ptr<switched_model::QuadrupedInterface> createQuadrupedInterface(const std::string &urdfFile,
                                                                             const std::string &taskFolder) {
    const std::string urdf = loadUrdf(urdfFile);
    return anymal::getAnymalInterface(urdf, taskFolder);
}