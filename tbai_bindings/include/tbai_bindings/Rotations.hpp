#pragma once

#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <pinocchio/math/rpy.hpp>
#include <tbai_bindings/Types.hpp>

namespace tbai {
namespace bindings {

/**
 * @brief Convert roll-pitch-yaw euler angles to quaternion
 *
 * @param rpy : roll-pitch-yaw euler angles
 * @return quaternion_t : quaternion
 */
inline quaternion_t rpy2quat(const vector3_t &rpy) {
    return angleaxis_t(rpy(2), vector3_t::UnitZ()) * angleaxis_t(rpy(1), vector3_t::UnitY()) *
           angleaxis_t(rpy(0), vector3_t::UnitX());
}

/**
 * @brief Convert quaternion to a 3x3 rotation matrix
 *
 * @param q : quaternion
 * @return matrix3_t : rotation matrix
 */
inline matrix3_t quat2mat(const quaternion_t &q) {
    return q.toRotationMatrix();
}

/**
 * @brief Convert a 3x3 rotation matrix to roll-pitch-yaw euler angles
 *
 * @param R : rotation matrix
 * @return vector3_t : roll-pitch-yaw euler angles
 */
inline vector3_t mat2rpy(const matrix3_t &R) {
    return pinocchio::rpy::matrixToRpy(R);
}

/**
 * @brief Convert a 3x3 rotation matrix to ocs2-style rpy euler angles, assume last yaw angle
 *
 * @param R : rotation matrix
 * @param lastYaw : previous yaw angle
 * @return vector3_t : roll-pitch-yaw euler angles
 */
inline vector3_t mat2oc2rpy(const matrix3_t &R, const scalar_t lastYaw) {
    // Taken from OCS2, see:
    // https://github.com/leggedrobotics/ocs2/blob/164c26b46bed5d24cd03d90588db8980d03a4951/ocs2_robotic_examples/ocs2_perceptive_anymal/ocs2_anymal_commands/src/TerrainAdaptation.cpp#L19
    vector3_t eulerXYZ = R.eulerAngles(0, 1, 2);
    ocs2::makeEulerAnglesUnique(eulerXYZ);
    eulerXYZ.z() = ocs2::moduloAngleWithReference(eulerXYZ.z(), lastYaw);
    return eulerXYZ;
}

/**
 * @brief Convert ocs2-style rpy angles to quaternion
 *
 * @param rpy : ocs2-style rpy angles
 * @return quaternion_t : quaternion
 */
inline quaternion_t ocs2rpy2quat(const vector3_t &rpy) {
    return angleaxis_t(rpy(0), vector3_t::UnitX()) * angleaxis_t(rpy(1), vector3_t::UnitY()) *
           angleaxis_t(rpy(2), vector3_t::UnitZ());
}

/**
 * @brief Convert roll-pitch-yaw euler angles to a 3x3 rotation matrix
 *
 * @param rpy : roll-pitch-yaw euler angles
 * @return matrix3_t : rotation matrix
 */
inline matrix3_t rpy2mat(const vector3_t &rpy) {
    return pinocchio::rpy::rpyToMatrix(rpy);
}

/**
 * @brief Convert a 3x3 rotation matrix to axis-angle representation
 *
 * @param R : rotation matrix
 * @return vector3_t : axis-angle representation
 */
inline vector3_t mat2aa(const matrix3_t &R) {
    angleaxis_t aa(R);
    return aa.axis() * aa.angle();
}

}  // namespace core
}  // namespace tbai
