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
	Matrix()
	:	m_rows(0), m_cols(0), m_data(NULL) {}

	// Redefine copy constructor (mandatory)
	Matrix(const Matrix& rhs)
	:	m_rows(0), m_cols(0), m_data(NULL)
    {
        // Here : call the assignment operator
        *this = rhs;
    }

	// Construction by size
	Matrix(int rows, int cols)
	:	m_rows(rows), m_cols(cols)
	{
		m_data = new float[m_rows * m_cols];
	}

    // Redefine destructor (mandatory)
    ~Matrix() { delete [] m_data; }

	// Element accessing operator
	float& operator()(int row, int col)
	{
		return m_data[row * m_cols + col];
	}

	// Assignment operator
	Matrix& operator=(const Matrix& rhs)
	{
        if (this != &rhs) {
            m_rows = rhs.m_rows;
            m_cols = rhs.m_cols;

			delete [] m_data;
			m_data = new float[m_rows * m_cols];

//			std::copy(rhs.m_data, rhs.m_data + (m_rows * m_cols), m_data);
            for (int e = 0; e < m_rows * m_cols; e++)
                (*this).m_data[e] = rhs.m_data[e];
        }

        return *this;
	}

    // Matrix addition
    Matrix operator+(Matrix& rhs)
    {
        Matrix result(m_rows, m_cols);

        for (int e = 0; e < m_rows * m_cols; e++)
            result.m_data[e] = (*this).m_data[e] + rhs.m_data[e];

        return result;
    }

    // Matrix substraction
    Matrix operator-(Matrix rhs)
    {
        Matrix result(m_rows, m_cols);

        for (int e = 0; e < m_rows * m_cols; e++)
            result.m_data[e] = (*this).m_data[e] - rhs.m_data[e];

        return result;
    }

	// Matrix multiplication
    // It uses the transpose technique, which allows coalescent
    // memory accesses, and thereby halves the processing time.
	Matrix operator*(Matrix& rhs)
	{
        Matrix transposed(m_cols, m_rows);

        for (int i = 0; i < m_rows; i++)
            for (int j = 0; j < m_cols; j++)
                transposed(j, i) = rhs(i, j);

        Matrix result(m_rows, rhs.m_cols);

        for (int i = 0; i < result.m_rows; i++)
            for (int j = 0; j < result.m_cols; j++)
				for (int k = 0; k < m_cols; k++)
					result(i, j) += (*this)(i, k) * transposed(j, k);
/*
        // Use 16x16 tiles
        Matrix result(m_rows, rhs.m_cols);
        #define TILE 128
        // Loop over all the tiles, stride by tile size
        for ( int i=0; i<result.m_rows; i+=TILE )
            for ( int j=0; j<result.m_cols; j+=TILE )
                for ( int k=0; k<m_cols; k+=TILE )
                // Regular multiply inside the tiles
                    for ( int y=i; y<i+TILE; y++ )
                        for ( int x=j; x<j+TILE; x++ )
                            for ( int z=k; z<k+TILE; z++ )
                                result(y,x) += (*this)(y,z) * rhs(z,x);
*/
		return result;
	}

	// Equality checking
	bool operator==(Matrix& rhs)
	{
		bool same = true;

        for (int e = 0; e < m_rows * m_cols; e++)
            same &= (*this).m_data[e] == rhs.m_data[e];

		return same;
	}

    // Frobenius (L2) norm
    float norm()
    {
        float acc = 0.f;

        for (int e = 0; e < m_rows * m_cols; e++)
            acc += m_data[e] * m_data[e];

        return std::sqrt(acc);
    }

    // Print matrix content
    void print()
    {
        for (int i = 0; i < m_rows; i++) {
            std::cout << std::endl;
    	    for (int j = 0; j < m_cols; j++)
    			std::cout << std::setw(3) << (*this)(i, j) << " ";
    	}
        std::cout << std::endl;
    }

	// Accessors
	int rows() { return m_rows; }
	int cols() { return m_cols; }
    float* data() { return m_data; }

  private :

    int m_rows;
    int m_cols;
    float * m_data;
};
