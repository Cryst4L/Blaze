////////////////////////////////////////////////////////////////////////////////
// File : Matrix.h
// Author: B. Halimi, 2017 - <bhalimi@outlook.fr>
// -----------------------------------------------------------------------------
// Basic matrix container embedding our data, and equiped with:
//
//         - addition and substraction
//         - naive matrix multiplication
//         - euclidean norm computation
//         - exact equality checking
//         - rows, cols, and size getters
//         - direct accessor to the data pointer
//
// Data are stored in a flat array, following the row-major convention:
//
//         M(row, col) = *(M.elements + row * M.cols + col)
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <iomanip>
#include <cmath>

class Matrix
{
  public :

	// Redefine default constructor (mandatory)
	Matrix();
	// Redefine copy constructor (mandatory)
	Matrix(const Matrix& rhs);
	// Construction by size
	Matrix(int rows, int cols);
	// Redefine destructor (mandatory)
	~Matrix();

	// Assignment operator
	Matrix& operator=(const Matrix& rhs);
	// Matrix addition
	Matrix operator+(Matrix& rhs);
	// Matrix substraction
	Matrix operator-(Matrix rhs);
	// Matrix multiplication :
	Matrix operator*(Matrix& rhs);
	// Equality checking
	bool operator==(Matrix& rhs);

	// Element accessing operator
	float& operator()(int row, int col);

	// Frobenius (L2) norm
	float norm();

	// Print matrix content
	void print();

	// Accessors
	int rows() { return m_rows; }
	int cols() { return m_cols; }
	float* data() { return m_data; }

  private :

	int m_rows;
	int m_cols;
	float * m_data;
};
