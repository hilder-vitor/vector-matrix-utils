#ifndef ___MATRIX_UTILS_BASICS
#define ___MATRIX_UTILS_BASICS

#include <iostream>
#include <vector>
#include<iomanip>

using namespace std;

template <typename ELEMENT>
void operator+=(vector<vector<ELEMENT> >& A, const vector<vector<ELEMENT> >& B){
	if (A.size() != B.size() || A[0].size() != B[0].size())
		throw std::invalid_argument("It is impossible to add matrix with different sizes.");
	unsigned int n = A.size();
	unsigned int m = A[0].size();
	#pragma omp parallel for collapse(2)
	for (unsigned int i = 0; i < n; i++){
		for (unsigned int j = 0; j < m; j++){
			A[i][j] = A[i][j] + B[i][j];
		}
	}
}

template <typename ELEMENT>
void operator+=(vector<vector<ELEMENT> >& A, const ELEMENT& c){
	unsigned int n = A.size();
	unsigned int m = A[0].size();
	#pragma omp parallel for collapse(2)
	for (unsigned int i = 0; i < n; i++){
		for (unsigned int j = 0; j < m; j++){
			A[i][j] = A[i][j] + c;
		}
	}
}

template <typename ELEMENT>
void operator-=(const vector<vector<ELEMENT> >& A, const vector<vector<ELEMENT> >& B){
	if (A.size() != B.size() || A[0].size() != B[0].size())
		throw std::invalid_argument("It is impossible to subtract matrix with different sizes.");
	unsigned int n = A.size();
	unsigned int m = A[0].size();
	#pragma omp parallel for collapse(2)
	for (unsigned int i = 0; i < n; i++){
		for (unsigned int j = 0; j < m; j++){
			A[i][j] = A[i][j] - B[i][j];
		}
	}
}

template <typename ELEMENT>
void operator-=(vector<vector<ELEMENT> >& A, const ELEMENT& c){
	unsigned int n = A.size();
	unsigned int m = A[0].size();
	#pragma omp parallel for collapse(2)
	for (unsigned int i = 0; i < n; i++){
		for (unsigned int j = 0; j < m; j++){
			A[i][j] = A[i][j] - c;
		}
	}
}

template <typename ELEMENT>
void operator*=(vector<vector<ELEMENT> >& A, const ELEMENT& c){
	unsigned int n = A.size();
	unsigned int m = A[0].size();
	#pragma omp parallel for collapse(2)
	for (unsigned int i = 0; i < n; i++){
		for (unsigned int j = 0; j < m; j++){
			A[i][j] = A[i][j] * c;
		}
	}
}

template <typename ELEMENT>
void operator*=(ELEMENT& c, const vector<vector<ELEMENT> >& A){
	unsigned int n = A.size();
	unsigned int m = A[0].size();
	#pragma omp parallel for collapse(2)
	for (unsigned int i = 0; i < n; i++){
		for (unsigned int j = 0; j < m; j++){
			A[i][j] = c * A[i][j];
		}
	}
}

template <typename ELEMENT>
vector<ELEMENT> operator*(const vector<vector<ELEMENT> >& A, const vector<ELEMENT>& v){
	if (A[0].size() != v.size())
		throw std::invalid_argument("A and v don't have conformant dimensions to be multiplied.");

	unsigned int N = A.size();
	unsigned int M = A[0].size();

	vector<ELEMENT> resp;
	resp.reserve(M);

	#pragma omp parallel for
	for (unsigned int i = 0; i < N; i++){
		ELEMENT innerProd = A[i][0] * v[0];
		for (unsigned int j = 1; j < M; j++){
			innerProd = innerProd + A[i][j] * v[j];
		}
		resp.push_back(innerProd);
	}
	return resp;
}


template <typename ELEMENT, typename T2>
void operator/=(vector<vector<ELEMENT> >& A, const T2& m){
	unsigned int N = A.size();
	unsigned int M = A[0].size();
	#pragma omp parallel for collapse(2)
	for (unsigned int i = 0; i < N; i++){
		for (unsigned int j = 0; j < M; j++){
			A[i][j] = A[i][j] / m;
		}
	}
}

template <typename ELEMENT>
vector<vector<ELEMENT> > operator+(const vector<vector<ELEMENT> >& A, const vector<vector<ELEMENT> >& B){
	vector<vector<ELEMENT> > mat(A);
	mat += B;
	return mat;
}

template <typename ELEMENT>
vector<vector<ELEMENT> > operator+(const vector<vector<ELEMENT> >& A, const ELEMENT& c){
	vector<vector<ELEMENT> > B(A);
	B += c;
	return B;
}
template <typename ELEMENT>
vector<vector<ELEMENT> > operator+(const ELEMENT& c, const vector<vector<ELEMENT> >& A){
	return A + c;
}

template <typename ELEMENT>
vector<vector<ELEMENT> > operator-(const vector<vector<ELEMENT> >& A, const vector<vector<ELEMENT> >& B){
	vector<ELEMENT> C(A);
	C -= B;
	return C;
}

template <typename ELEMENT>
vector<vector<ELEMENT> > operator-(const vector<vector<ELEMENT> >& A, const ELEMENT& c){
	vector<vector<ELEMENT> > B(A);
	B -= c;
	return B;
}

template <typename ELEMENT>
vector<vector<ELEMENT> > operator*(const vector<vector<ELEMENT> >& A, const vector<vector<ELEMENT> >& B){
	if (A[0].size() != B.size())
		throw std::invalid_argument("A and B don't have conformant dimensions to be multiplied.");

	unsigned int N = A.size();
	unsigned int P = A[0].size();
	unsigned int M = B[0].size();

	vector<vector<ELEMENT> > C(N);
	
	#pragma omp parallel for
	for (unsigned int i = 0; i < N; i++){
		for (unsigned int j = 0; j < M; j++){
			ELEMENT innerProduct = A[i][0] * B[0][j];
			for (unsigned int k = 1; k < P; k++){
				innerProduct = innerProduct + A[i][k] * B[k][j];
			}
			C[i].push_back(innerProduct);
		}
	}
	return C;
}

template <typename ELEMENT>
vector<vector<ELEMENT> > operator*(const vector<vector<ELEMENT> >& A, const ELEMENT& c){
	vector<vector<ELEMENT> > B(A);
	B *= c;
	return B;
}


template <typename ELEMENT>
vector<vector<ELEMENT> > operator*(const ELEMENT& c, const vector<vector<ELEMENT> >& A){
	return A * c;
}

template <typename ELEMENT, typename T2>
vector<vector<ELEMENT> > operator/(const vector<vector<ELEMENT> >& A, const T2& m){
	vector<vector<ELEMENT> > B(A);
	B /= m;
	return B;
}

template <typename ELEMENT>
std::ostream& operator<<(std::ostream& os, const vector<vector<ELEMENT> >& A){
	unsigned int N = A.size();
	unsigned int M = A[0].size();
	for (unsigned int i = 0; i < N; i++){
		for (unsigned int j = 0; j < M; j++){
			os << A[i][j] << " ";
		}
		os << endl;
	}
	return os;
}

template <class ELEMENT> class SymmetricMatrix{
	private:
		unsigned int N;
		vector<vector<ELEMENT> > A;

	public:

		SymmetricMatrix(unsigned int __N, const ELEMENT& any_value) : N(__N), A(__N) {
			#pragma omp parallel for
			for (unsigned int i = 0; i < N; i++){
				for (unsigned int j = 0; j <= i; j++){
					A[i].push_back(any_value);
				}
			}
		}

		SymmetricMatrix(const vector<vector<ELEMENT> >& matrix) : N(matrix.size()), A(matrix.size()){
			#pragma omp parallel for
			for (unsigned int i = 0; i < N; i++){
				for (unsigned int j = 0; j <= i; j++){
					A[i].push_back(matrix[i][j]);
				}
			}
		}

		ELEMENT get(unsigned int i, unsigned int j) const{
			if (j > i)
				return A[j][i];
			return A[i][j];
		}

		void set(const ELEMENT& x, unsigned int i, unsigned int j){
			if (j > i){
				A[j][i] = ELEMENT(x);
			}else{
				A[i][j] = ELEMENT(x);
			}
		}

		unsigned int size() const{
			return N;
		}

		/*
		 * This method adds the value c in the main diagonal of this symmetric matrix.
		 */
		template <typename T>
		void add_in_diagonal(const T& c){
			for (unsigned int i = 0; i < N; i++){
				A[i][i] = A[i][i] + c;
			}
		}

		SymmetricMatrix<ELEMENT> operator*(const SymmetricMatrix<ELEMENT>& S){
			if (S.size() != N){
				throw std::invalid_argument("It is impossible to multiply symmetric matrices with different dimensions");
			}
			SymmetricMatrix<ELEMENT> prod(S);
			#pragma omp parallel for
			for (unsigned int i = 0; i < N; i++){
				for (unsigned int j = 0; j <= i; j++){
					ELEMENT innerProd = get(0, i) * S.get(0, j);
					for (unsigned int k = 1; k < N; k++){
						innerProd = innerProd + get(k, i) * S.get(k, j);
					}
					prod.set(innerProd, i, j);
				}
			}
			return prod; 
		}

		template <typename T>
		vector<ELEMENT> operator*(const vector<T>& v){
			if (v.size() != N){
				throw std::invalid_argument("Dimensions of symmetric matrices and vectorwith different dimensions");
			}
			vector<ELEMENT> prod(N, get(0, 0)); // create a N-dimensional vector with dummy values
			#pragma omp parallel for
			for (unsigned int i = 0; i < N; i++){
				ELEMENT innerProd = get(i, 0) * v[0];
				for (unsigned int j = 1; j < N; j++){
					innerProd = innerProd + get(i, j) * v[j];
				}
				prod[i] = innerProd;
			}
			return prod; 
		}

		void operator*=(const ELEMENT& c){
			for (unsigned int i = 0; i < N; i++){
				for (unsigned int j = 0; j <= i; j++){
					A[i][j] = A[i][j] * c;
				}
			}
		}
		SymmetricMatrix<ELEMENT> operator*(const ELEMENT& c) const{
			SymmetricMatrix<ELEMENT> prod(*this);
			prod *= c;
			return prod;
		}

		void operator+=(const SymmetricMatrix<ELEMENT>& S){
			if (S.size() != N){
				throw std::invalid_argument("It is impossible to sum symmetric matrices with different dimensions.");
			}
			#pragma omp parallel for
			for (unsigned int i = 0; i < N; i++){
				for (unsigned int j = 0; j <= i; j++){
					A[i][j] = A[i][j] + S.get(i, j);
				}
			}
		}
	
		void operator+=(const ELEMENT& c){
			#pragma omp parallel for
			for (unsigned int i = 0; i < N; i++){
				for (unsigned int j = 0; j <= i; j++){
					A[i][j] = A[i][j] + c;
				}
			}
		}
		
		SymmetricMatrix<ELEMENT> operator+(const SymmetricMatrix<ELEMENT>& S) const{
			SymmetricMatrix<ELEMENT> sum(*this);
			sum += S;
			return sum;
		}

		SymmetricMatrix<ELEMENT> operator+(const ELEMENT& c){
			SymmetricMatrix<ELEMENT> sum(*this);
			sum += c;
			return sum;
		}

		void operator-=(const SymmetricMatrix<ELEMENT>& S){
			if (S.size() != N){
				throw std::invalid_argument("It is impossible to subtract matrices with different dimensions.");
			}
			#pragma omp parallel for
			for (unsigned int i = 0; i < N; i++){
				for (unsigned int j = 0; j <= i; j++){
					A[i][j] = A[i][j] - S.get(i, j);
				}
			}
		}

		void operator-=(const ELEMENT& c){
			#pragma omp parallel for
			for (unsigned int i = 0; i < N; i++){
				for (unsigned int j = 0; j <= i; j++){
					A[i][j] = A[i][j] - c;
				}
			}
		}

		SymmetricMatrix<ELEMENT> operator-(const ELEMENT& c) const{
			SymmetricMatrix<ELEMENT> sub(*this);
			sub -= c;
			return sub;
		}

		SymmetricMatrix<ELEMENT> operator-(const SymmetricMatrix<ELEMENT>& S) const{
			SymmetricMatrix<ELEMENT> sub(*this);
			sub -= S;
			return sub;
		}

		template <typename T2>
		void operator/=(const T2& scalar){
			#pragma omp parallel for
			for (unsigned int i = 0; i < N; i++){
				for (unsigned int j = 0; j <= i; j++){
					A[i][j] = A[i][j] / scalar;
				}
			}
		}
		
		template <typename T2>
		SymmetricMatrix<ELEMENT> operator/(const T2& scalar){
			SymmetricMatrix<ELEMENT> div(*this);
			div /= scalar;
			return div;
		}

		vector<vector<ELEMENT> > to_usual_matrix() const{
			vector<vector<ELEMENT> > mat(N);
			for (unsigned int i = 0; i < N; i++){
				for (unsigned int j = 0; j < N; j++){
					mat[i].push_back(get(i, j));
				}
			}
			return mat;
		}

		/*******************************
		 * 		This method is only defined for positive values of k!
		 * 	(for k > 0)
		 **************************************************************/
		void power(unsigned int k){
			if (k <= 1) // for k = 1, this^1 = this
				return;

			SymmetricMatrix<ELEMENT> resp(*this);
			k /= 4; 

			while(true){
				short int bit = k % 2;
				k /= 2;
				if (bit == 1)
					resp = resp * (*this);
				/* doing this to avoid the last squaring (which is useles) */
				if (k == 0)
					break;
				else{
					*this = (*this) * (*this);
				}
			}
			*this = resp;
		}

};

template <typename ELEMENT>
SymmetricMatrix<ELEMENT> power(const SymmetricMatrix<ELEMENT>& A, unsigned int k){
	SymmetricMatrix<ELEMENT> B(A);
	B.power(k);
	return B;
}

template <typename ELEMENT>
std::ostream& operator<<(std::ostream& os, const SymmetricMatrix<ELEMENT>& S){
	unsigned int N = S.size();
	for (unsigned int i = 0; i < N; i++){
		for (unsigned int j = 0; j < N; j++){
			os << S.get(i, j) << " ";
		}
		os << endl;
	}
	return os;
}


template <typename ELEMENT>
vector<vector<ELEMENT> > create_matrix(ELEMENT default_value, unsigned int num_rows, unsigned int num_cols){
	vector<vector<ELEMENT> > A(num_rows);
	for (unsigned int i = 0; i < num_rows; i++){
		for (unsigned int j = 0; j < num_cols; j++){
			A[i].push_back(default_value);
		}
	}
	return A;
}



/* This function returns the inner product of the i-th column of A by the j-th column of B */
template <typename ELEMENT>
ELEMENT inner_product_column_by_column(const vector<vector< ELEMENT> >& A, unsigned int i, const vector<vector<ELEMENT> >& B, unsigned int j){
	unsigned int N = A.size();
	ELEMENT inner_product = A[0][i] * B[0][j];

	for (unsigned int k = 1; k < N; k++){
		inner_product = inner_product + A[k][i] * B[k][j];
	}
	return inner_product;
}


template <typename ELEMENT>
vector<vector<ELEMENT> > transpose(const vector<vector<ELEMENT> >& A){
	unsigned int N = A.size();
	unsigned int P = A[0].size();

	vector<vector<ELEMENT> > T(create_matrix(A[0][0], P, N));
	for (unsigned int i = 0; i < P; i++){
		for (unsigned int j = 0; j < N; j++){
			T[i][j] = A[j][i];
		}
	}

	return T;
}


/**
 *	Receives a NxP matrix A and returns a PxP matrix equal to transpose(A) * A.
 *
*/
template <typename ELEMENT>
SymmetricMatrix<ELEMENT> multiply_transpose_matrix_by_matrix(const vector<vector< ELEMENT> >& A){
	unsigned int P = A[0].size();

	SymmetricMatrix<ELEMENT> C(P, A[0][0]);
	#pragma omp parallel for
	for (unsigned int i = 0; i < P; i++){
		for (unsigned int j = 0; j <= i; j++){
			C.set(inner_product_column_by_column(A, i, A, j), i, j);
		}
	}
	return C;
}


template <typename ELEMENT>
vector<vector<ELEMENT> > outer_product(const vector<ELEMENT>& u, const vector<ELEMENT>& v){
	if (u.size() != v.size()){
		throw std::invalid_argument("Outer product should be done with vectors of same dimension.");
	}
	unsigned int N = u.size();
	vector<vector<ELEMENT> > O(N);
	#pragma omp parallel for
	for (unsigned int i = 0; i < N; i++){
		O[i].reserve(N);
		for (unsigned int j = 0; j < N; j++){
			O[i].push_back(u[i] * v[j]);
		}
	}
	return O;
}
template <typename ELEMENT>
SymmetricMatrix<ELEMENT> outer_product(const vector<ELEMENT>& u){
	unsigned int N = u.size();
	SymmetricMatrix<ELEMENT> O(N, u[0]);
	#pragma omp parallel for
	for (unsigned int i = 0; i < N; i++){
		for (unsigned int j = 0; j <= i; j++){
			O.set(u[i] * u[j], i, j);
		}
	}
	return O;
}

template <typename ELEMENT>
ELEMENT mean_of_column(const vector<vector<ELEMENT> >& A, unsigned int j){
	unsigned int N = A.size();
	ELEMENT mean = A[0][j];
	for (unsigned int i = 1; i < N; i++){
		mean = mean + A[i][j];
	}
	return mean / (N - 1);
}

template <typename ELEMENT>
void subtract_mean_column_wise(vector<vector<ELEMENT> >& A){
	unsigned int N = A.size();
	unsigned int P = A[0].size();

	#pragma omp parallel for
	for (unsigned int j = 0; j < P; j++){
		ELEMENT mean = mean_of_column(A, j);
		for (unsigned int i = 0; i < N; i++){
			A[i][j] =  mean - A[i][j];
		}
	}
}

template <typename ELEMENT>
SymmetricMatrix<ELEMENT> calculate_covariance(vector<vector<ELEMENT> >& data){
	subtract_mean_column_wise(data);
	return multiply_transpose_matrix_by_matrix(data);
}

template <typename T>
SymmetricMatrix<T> diagonal(const T& diagonal_element, unsigned int N){
	SymmetricMatrix<T> D(N, diagonal_element - diagonal_element);
	for (unsigned int i = 0; i < N; i++){
		D.set(diagonal_element, i, i);
	}
	return D;
}

template <typename T>
vector<vector<T> > get_non_zero_rows(const vector<vector<T> >& A){
	unsigned int N = A.size();
	unsigned int P = A[0].size();
	T zero = A[0][0] - A[0][0];
	vector<vector<T> > B;
	for (unsigned int i = 0; i < N; i++){
		unsigned int j;
		for (j = 0; j < N && zero == A[i][j]; j++);
		// if some non-zero value was found in this column
		if (j < P){
			B.push_back(A[i]);
		}
	}
	return B;
}

template <typename T>
vector<unsigned int> get_indices_non_zero_columns(const vector<vector<T> >& A){
	unsigned int N = A.size();
	unsigned int P = A[0].size();
	T zero = A[0][0] - A[0][0];
	vector<unsigned int> indices;
	for (unsigned int j = 0; j < P; j++){
		// testing if j-th column is null
		unsigned int i;
		for (i = 0; i < N && zero == A[i][j]; i++);
		// if some non-zero value was found in this column
		if (i < N){
			indices.push_back(j);
		}
	}
	return indices;
}

template <typename T>
vector<vector<T> > remove_columns(const vector<vector<T> >& A, const vector<unsigned int>& indices_cols){
	unsigned int N = A.size();
	unsigned int P = A[0].size();
	T zero = A[0][0] - A[0][0];
	vector<vector<T> > B(N);
	for (unsigned int j = 0; j < P; j++){
		bool is_in_indices_cols = false;
		unsigned i = 0;
		while (i < indices_cols.size() && !is_in_indices_cols){
			is_in_indices_cols = (j == indices_cols[i]);
			i++;
		}
		if (is_in_indices_cols){
			for (unsigned int l = 0; l < N; l++){
				B[l].push_back(A[l][j]);
			}
		}
	}
	return B;
}

template <typename T>
vector<vector<T> > get_non_zero_columns(const vector<vector<T> >& A){
	vector<unsigned int> indices = get_indices_non_zero_columns(A);
	return remove_columns(A, indices);
}



template <typename ELEMENT>
vector<vector<ELEMENT> > get_lines(unsigned int line, unsigned int qnt, const vector<vector<ELEMENT> >& A){
	unsigned int N = A.size();
	vector<vector<ELEMENT> > B;
	for (unsigned int i = line; i < line + qnt && i < N; i++){
		B.push_back(A[i]);
	}
	return B;
}

template <typename ELEMENT>
void print_cofactor(const vector<vector<ELEMENT> >& A, unsigned int starting_line, const vector<unsigned int>& cols_off){
	unsigned int N = A.size();
	unsigned int P = A[0].size();
	for (unsigned int i = 0; i < starting_line; i++){
		for (unsigned int j = 0; j < P; j++){
			cout << "  * ";
		}
		cout << endl;
	}

	for (unsigned int i = starting_line; i < N; i++){
		for (unsigned int j = 0; j < P; j++){
			if (0 == cols_off[j])
				cout << setw(3) << A[i][j] << " ";
			else
				cout << "  * ";
		}
		cout << endl;
	}
	cout << endl;
}

template <typename ELEMENT>
void rec_print_cofactors_used_in_determinant(const vector<vector<ELEMENT> >& A, unsigned int starting_line, vector<unsigned int>& cols_off){
	unsigned int N = A.size();
	unsigned int P = A[0].size();


	for (unsigned int j = 0; j < P; j++){
		if (0 == cols_off[j]){
			cols_off[j] = 1;
			print_cofactor(A, starting_line, cols_off);

			if (starting_line < N-1){
				rec_print_cofactors_used_in_determinant(A, starting_line + 1, cols_off);
			}
			cols_off[j] = 0;
		}
	}

}

template <typename ELEMENT>
void print_cofactors_used_in_determinant(const vector<vector<ELEMENT> >& A){
	vector<unsigned int> cols_off(A[0].size(), 0);
	rec_print_cofactors_used_in_determinant(A, 1, cols_off);
}


template <typename ELEMENT>
void get_submatrix(const vector<vector<ELEMENT> >& A, vector<vector<ELEMENT> >& cofactor, unsigned int row, unsigned int col){
	unsigned int N = A.size();
	for(unsigned int i = 0; i < row; i++){
		for(unsigned int j = 0; j < col; j++){
			cofactor[i][j] = A[i][j];
		}
	}

	for(unsigned int i = row+1; i < N; i++){
		for(unsigned int j = 0; j < col; j++){
			cofactor[i-1][j] = A[i][j];
		}
	}

	for(unsigned int i = 0; i < row; i++){
		for(unsigned int j = col+1; j < N; j++){
			cofactor[i][j-1] = A[i][j];
		}
	}
	for(unsigned int i = row+1; i < N; i++){
		for(unsigned int j = col + 1; j < N; j++){
			cofactor[i-1][j-1] = A[i][j];
		}
	}
}

template <typename ELEMENT>
ELEMENT rec_det(const vector<vector<ELEMENT> >& A, unsigned int starting_line, vector<unsigned int>& cols_off){
	unsigned int N = A.size();

	if (starting_line == N-2){
		unsigned int first_col = 0;
		for (; first_col < N && cols_off[first_col] != 0; first_col++);

		unsigned int second_col = first_col+1;
		for (; second_col < N && cols_off[second_col] != 0; second_col++);

		return A[starting_line][first_col]*A[starting_line+1][second_col] - A[starting_line][second_col]*A[starting_line+1][first_col];
	}
	
	ELEMENT d = A[0][0] - A[0][0]; // d = 0 (of type ELEMENT)

	int sign = 1;
	for (unsigned int j = 0; j < N; j++){
		if (0 == cols_off[j]){
			cols_off[j] = 1;
			d += sign * A[starting_line][j] * rec_det(A, starting_line + 1, cols_off);

			cols_off[j] = 0;
			sign *= -1;
		}
	}
	return d;
}

template <typename ELEMENT>
ELEMENT rec_determinant(const vector<vector<ELEMENT> >& A){
	vector<unsigned int> cols_off(A[0].size(), 0);
	return rec_det(A, 0, cols_off);
}

template <typename ELEMENT>
void switch_lines(vector<vector<ELEMENT> >& A, unsigned int i, unsigned int k){
	if (i != k){
		unsigned int M = A[i].size();
		ELEMENT pivot = A[0][0];
		for (unsigned int l = 0; l < M; l++){
			pivot = A[i][l];
			A[i][l] = A[k][l];
			A[k][l] = pivot;
		}
	}
}


/********************************************************************
 *    Perform Gaussian elimination in order to diagonalize A.
 *    It switchs lines every time the current diagonal element is zero.
 *    The number of switched lines is returned.
 ***********************************************************************/
template <typename ELEMENT>
int diagonalize_switching_lines(vector<vector<ELEMENT> >& A){
	unsigned int number_switched_lines = 0;
	unsigned int N = A.size();
	ELEMENT zero = A[0][0] - A[0][0];
	for (unsigned int j = 0; j < N; j++){
		if (0 == A[j][j]){
			unsigned int largest = j;
			for (unsigned int k = j+1; k < N; k++){
				if (A[largest][j] < A[k][j]){
					largest = k;
				}
			}
			if (largest != j){
				switch_lines(A, j, largest);
				number_switched_lines++;
			}
		}
		if (0 != A[j][j]){ // maybe the sub-column has only zero elements...
			for (unsigned int i = j+1; i < N; i++){
				ELEMENT multiplier = A[i][j] / A[j][j];	 
				A[i] -= multiplier*A[j];
				A[i][j] = zero;
			}
		}
	}
	return number_switched_lines;
}

template <typename ELEMENT>
ELEMENT multiply_elements_main_diagonal(const vector<vector<ELEMENT> >& A){
	unsigned int N = A.size();
	ELEMENT result = A[0][0];
	for (unsigned int i = 1; i < N; i++){
		result *= A[i][i];
	}
	return result;
}


template <typename ELEMENT>
ELEMENT determinant(const vector<vector<ELEMENT> >& X){
	vector<vector<ELEMENT> > A(X);
	unsigned int N = A.size();

	unsigned int number_switched_lines = diagonalize_switching_lines(A);

	if (number_switched_lines % 2 == 0)
		return multiply_elements_main_diagonal(A);

	return -multiply_elements_main_diagonal(A);
}


template <typename ELEMENT>
vector<vector<ELEMENT> > get_adjugate_matrix(vector<vector<ELEMENT> > A){
	unsigned int N = A.size();
	vector<vector<ELEMENT> > adj(A);

	if (1 == N){
		adj[0][0] = 1;
		return adj;
	}
	
	vector<vector<ELEMENT> > sub_matrix(create_matrix(A[0][0], N-1, N-1));

	for (unsigned int i = 0; i < N; i++){
		for(unsigned int j = 0; j < N; j++){
			int sign = ((i+j)%2 == 0 ? 1 : -1);
			get_submatrix(A, sub_matrix, i, j);
			adj[j][i] = sign*determinant(sub_matrix);
		}
	}
	return adj;
}

template <typename T>
T find_first_non_zero(const vector<vector<T> >& A){
	unsigned int N = A.size();
	unsigned int M = A[0].size();

	T zero = A[0][0] - A[0][0];

	for (unsigned int i = 0; i < N; i++){
		for (unsigned int j = 0; j < M; j++){
			if (zero != A[i][j])
				return A[i][j];
		}
	}
	return zero;
}


template <typename T>
T absolute_value(T x){
	T zero = x - x;
	if (x < zero)
		return -x;
	
	return x;
}

template <typename T>
vector<unsigned int> PLU(const vector<vector<T> >& A, vector<vector<T> >& L, vector<vector<T> >& U){
	unsigned int N = A.size();
	vector<unsigned int> P(N);
	for (unsigned int j = 0; j < N; j++)
		P[j] = j;

	U = A;

	T non_zero = find_first_non_zero(A);
	T one = non_zero / non_zero;

	T zero = A[0][0] - A[0][0];
	for (unsigned int j = 0; j < N; j++){
		if (zero == U[j][j]){
			unsigned int largest = j;
			for (unsigned int k = j+1; k < N; k++){
				if (absolute_value(U[largest][j]) < absolute_value(U[k][j])){
					largest = k;
				}
			}
			if (largest != j){
				switch_lines(U, j, largest);
				switch_lines(L, j, largest);
				unsigned int pivot = P[j];
				P[j] = P[largest];
				P[largest] = pivot;
			}
		}
		L[j][j] = one;
		if (zero != U[j][j]){ // maybe the sub-column has only zero elements...
			for (unsigned int i = j+1; i < N; i++){
				T multiplier = U[i][j] / U[j][j];	 
				U[i] -= multiplier*U[j];
				U[i][j] = zero;
				L[i][j] = multiplier;
			}
		}
	}
	return P;
}

template <typename T1, typename T2>
void inverse_uper_triangular(const vector<vector<T1> >& upper_trian, vector<vector<T2> >& inv){
	unsigned int N = upper_trian.size();
	T2 zero = inv[0][0] - inv[0][0];
	bool has_zero_in_diagonal = false;

	for (unsigned int i = 0; i < N && !has_zero_in_diagonal; i++){
		if (zero == upper_trian[i][i]){
			has_zero_in_diagonal = true;
		}
	}

	if (has_zero_in_diagonal){
		throw std::invalid_argument("This upper triangular matrix A is not invertible.");
	}


	T1 one = upper_trian[0][0] / upper_trian[0][0];
	
	// lower triangule of inverse is zero
	for (unsigned int i = 1; i < N; i++){
		for (unsigned int j = 0; j < i; j++){
			inv[i][j] = zero;
		}
	}

	// diagonal of inverse matrix is easily calculated
	for (unsigned int i = 0; i < N; i++){
		inv[i][i] = one / upper_trian[i][i]; 
	}

	for (unsigned int i = 0; i < N; i++){
		for (unsigned int j = i+1; j < N; j++){
			T2 sum = inv[i][i] * upper_trian[i][j];
			for (unsigned int l = i+1; l < j; l++){
				sum += inv[i][l] * upper_trian[l][j];
			}
			inv[i][j] = sum / (-upper_trian[j][j]);
		}
	}
}

template <typename T>
vector<vector<T> > inverse_uper_triangular(const vector<vector<T> >& A){
	unsigned int N = A.size();
	vector<vector<T> > inv(A);

	inverse_uper_triangular(A, inv);

	return inv;
}

template <typename T>
void copy_column(vector<vector<T> >& A, const vector<vector<T> >& L, unsigned int j_A, unsigned int j_L){
	for(unsigned int i = 0; i < A.size(); i++){
		A[i][j_A] = L[i][j_L];
	}
}

template <typename T1, typename T2>
void inverse(const vector<vector<T1> >& A, vector<vector<T2> >& invA){
	vector<vector<T2> > L(invA);
	vector<vector<T2> > U(invA);

	vector<unsigned int> P = PLU(A, L, U);
	T2 zero = L[0][0]-L[0][0];
	T2 one =  L[0][0];

	inverse_uper_triangular(U, invA); // now invA = U^-1
	inverse_uper_triangular(transpose(L), U); // now U = L^-1
	
	L = invA * transpose(U); // now L = U^-1 * L^-1

	// apply the inverse permutation P to obtain the inverse of A
	for (unsigned int i = 0; i < P.size(); i++){
		copy_column(invA, L, P[i], i);
	}
}

template <typename T>
vector<vector<T> > inverse(const vector<vector<T> >& A){
	unsigned int N = A.size();
	vector<vector<T> > inv(A);
	
	inverse(A, inv);

	return inv;
}


#endif
