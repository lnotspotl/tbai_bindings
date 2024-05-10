#pragma once

#include <ocs2_core/Types.h>

namespace tbai {
namespace bindings {

using scalar_t = ocs2::scalar_t;
using vector_t = ocs2::vector_t;
using matrix_t = ocs2::matrix_t;

using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
using matrix3_t = Eigen::Matrix<scalar_t, 3, 3>;
using quaternion_t = Eigen::Quaternion<scalar_t>;
using angleaxis_t = Eigen::AngleAxis<scalar_t>;


}  // namespace bindings
}  // namespace tbai