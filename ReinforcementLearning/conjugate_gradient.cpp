#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

// Conjugate Gradient Method for matrix inversion
template <typename MatrixType, typename VectorType>
MatrixType conjugateGradient(const MatrixType &A, const VectorType &b, int maxIterations, double tolerance) {
    MatrixType x = MatrixType::Zero(b.rows(), b.cols());  // Initial guess
    VectorType r = b - A * x;  // Residual
    VectorType p = r;         // Initial search direction

    for (int k = 0; k < maxIterations; ++k) {
        double alpha = r.dot(r) / (p.transpose().dot(A * p));
        x = x + alpha * p;
        VectorType rNext = r - alpha * A * p;

        if (rNext.norm() < tolerance) {
            std::cout << "Convergence achieved after " << k + 1 << " iterations." << std::endl;
            return x;
        }

        double beta = rNext.dot(rNext) / r.dot(r);
        p = rNext + beta * p;
        r = rNext;
    }

    std::cerr << "Conjugate gradient method did not converge within the specified number of iterations." << std::endl;
    return x;
}

int main() {
    // Example usage
    MatrixXd A(3, 3);
    A << 4, 1, -2,
         1, 5, 1,
         -2, 1, 3;

    VectorXd b(3, 1);
    b << 1, 2, 3;

    // Solve Ax = b for x using the conjugate gradient method
    MatrixXd x = conjugateGradient(A, b, 100, 1e-6);

    // Display the result
    std::cout << "Solution x:\n" << x << std::endl;

    std::cout << b <<std::endl;
    std::cout << A*x <<std::endl;

    return 0;
}
