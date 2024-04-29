#include <gtest/gtest.h>
#include <tbai_bindings/Utils.hpp>

using ocs2::matrix_t;
using ocs2::scalar_t;
using ocs2::vector_t;

static vector_t eigenArange(int n) {
    vector_t v(n);
    for (int i = 0; i < n; i++) {
        v(i) = i;
    }
    return v;
}

static matrix_t eigenArange(int rows, int cols) {
    matrix_t m(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m(i, j) = i * cols + j;
        }
    }
    return m;
}

static torch::Tensor torchArange(int n) {
    return torch::arange(n).toType(torch::kFloat32);
}

static torch::Tensor torchArange(int rows, int cols) {
    return torch::arange(rows * cols).reshape({rows, cols}).toType(torch::kFloat32);
}

TEST(Eigen2Torch, Vector2Torch) {
    vector_t v = eigenArange(10);
    torch::Tensor t = tbai::bindings::vector2torch(v);

    // Check dimensions
    EXPECT_EQ(t.size(0), v.rows());

    // Check values
    for (int i = 0; i < v.size(); i++) {
        EXPECT_NEAR(v(i), t[i].item<float>(), 1e-6);
    }
}

TEST(Eigen2Torch, SquareMat2Torch) {
    constexpr size_t N = 10;
    matrix_t m = eigenArange(N, N);
    torch::Tensor t = tbai::bindings::matrix2torch(m);

    // Check dimensions
    EXPECT_EQ(t.size(0), N);
    EXPECT_EQ(t.size(1), N);

    // Check values
    for (int i = 0; i < m.rows(); i++) {
        for (int j = 0; j < m.cols(); j++) {
            EXPECT_NEAR(m(i, j), t[i][j].item<float>(), 1e-6);
        }
    }
}

TEST(Eigen2Torch, RectMat2Torch) {
    constexpr size_t N = 4;
    constexpr size_t M = 7;
    matrix_t m = eigenArange(N, M);
    torch::Tensor t = tbai::bindings::matrix2torch(m);

    // Check dimensions
    EXPECT_EQ(t.size(0), N);
    EXPECT_EQ(t.size(1), M);

    // Check values
    for (int i = 0; i < m.rows(); i++) {
        for (int j = 0; j < m.cols(); j++) {
            EXPECT_NEAR(m(i, j), t[i][j].item<float>(), 1e-6);
        }
    }
}

TEST(Torch2Eigen, Torch2Vector) {
    constexpr size_t N = 9;
    torch::Tensor t = torchArange(N);
    vector_t v = tbai::bindings::torch2vector(t);

    // Check dimensions
    EXPECT_EQ(v.rows(), t.size(0));

    // Check values
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(t[i].item<float>(), v(i), 1e-6);
    }
}

TEST(Torch2Eigen, Torch2SquareMat) {
    constexpr size_t N = 3;
    torch::Tensor t = torchArange(N, N);
    matrix_t m = tbai::bindings::torch2matrix(t);

    // Check dimensions
    EXPECT_EQ(m.rows(), N);
    EXPECT_EQ(m.cols(), N);

    // Check values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            EXPECT_NEAR(t[i][j].item<float>(), m(i, j), 1e-6);
        }
    }
}

TEST(Torch2Eigen, Torch2RectMat) {
    constexpr size_t N = 10;
    constexpr size_t M = 5;
    torch::Tensor t = torchArange(N, M);
    matrix_t m = tbai::bindings::torch2matrix(t);

    // Check dimensions
    EXPECT_EQ(m.rows(), N);
    EXPECT_EQ(m.cols(), M);

    // Check values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            EXPECT_NEAR(t[i][j].item<float>(), m(i, j), 1e-6);
        }
    }
}

TEST(Combined, RoundTripEig2Torch2Eig) {
    constexpr size_t N = 3;
    constexpr size_t M = 2;
    matrix_t m = eigenArange(N, M);
    torch::Tensor t_m = tbai::bindings::matrix2torch(m);

    ASSERT_EQ(t_m.size(0), N);
    ASSERT_EQ(t_m.size(1), M);

    for (int i = 0; i < m.rows(); i++) {
        for (int j = 0; j < m.cols(); j++) {
            EXPECT_NEAR(m(i, j), t_m[i][j].item<float>(), 1e-6);
        }
    }

    matrix_t m2 = tbai::bindings::torch2matrix(t_m);

    ASSERT_EQ(m2.rows(), N);
    ASSERT_EQ(m2.cols(), M);

    for (int i = 0; i < m.rows(); i++) {
        for (int j = 0; j < m.cols(); j++) {
            EXPECT_NEAR(m(i, j), m2(i,j), 1e-6);
        }
    }
}

TEST(Combined, RoundTripTorch2Eig2Torch) {
    constexpr size_t N = 10;
    constexpr size_t M = 531;
    torch::Tensor t = torchArange(N, M);

    ASSERT_EQ(t.size(0), N);
    ASSERT_EQ(t.size(1), M);

    matrix_t m = tbai::bindings::torch2matrix(t);

    ASSERT_EQ(t.size(0), N);
    ASSERT_EQ(t.size(1), M);

    for (int i = 0; i < m.rows(); i++) {
        for (int j = 0; j < m.cols(); j++) {
            EXPECT_NEAR(t[i][j].item<float>(), m(i, j), 1e-6);
        }
    }

    torch::Tensor t2 = tbai::bindings::matrix2torch(m);

    ASSERT_EQ(t.size(0), N);
    ASSERT_EQ(t.size(1), M);

    for (int i = 0; i < t.size(0); i++) {
        for (int j = 0; j < t.size(1); j++) {
            EXPECT_NEAR(t[i][j].item<float>(), t2[i][j].item<float>(), 1e-6);
        }
    }
}

TEST(Storage, eig2torch) {
    constexpr size_t N = 10;
    vector_t v = eigenArange(N);
    torch::Tensor t = tbai::bindings::vector2torch(v);

    v(0) = 100;
    EXPECT_NEAR(t[0].item<float>(), 0, 1e-6);
}

TEST(Storage, torch2eig) {
    constexpr size_t N = 10;
    torch::Tensor t = torchArange(N);
    vector_t v = tbai::bindings::torch2vector(t);

    t[3] = 100;
    EXPECT_NEAR(v(3), 3, 1e-6);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}