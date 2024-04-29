#pragma once

#include <Eigen/Dense>
#include <tbai_bindings/Types.hpp>
#include <torch/torch.h>

namespace tbai {
namespace bindings {

/** Torch -> Eigen*/
vector_t torch2vector(const torch::Tensor &t);
matrix_t torch2matrix(const torch::Tensor &t);

/** Eigen -> Torch */
torch::Tensor vector2torch(const vector_t &v);
torch::Tensor matrix2torch(const matrix_t &m);

}  // namespace bindings
}  // namespace tbai