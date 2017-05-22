#include "vectorutils.hpp"
#include "matrixutils.hpp"

#include <random>

#define EPSILON 0.00000001

/******************************************************	
 * 	Author: Hilder Vitor Lima Pereira 
 *
 * 	e-mail: hilder.vitor@gmail.com
 *
 * 	compiling:  g++ test.cpp -std=c++11
 * ****************************************************/

vector<vector<double> > random_matrix(unsigned int N, unsigned int M){
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
    
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<> dis(-50, 50);
	
	vector<vector<double> > A = create_matrix(0.0, N, N);
		  
	for (int i=0; i < N; i++){	
		for (int j = 0; j < M; j++){
			A[i][j] = dis(gen) / 100.0;
		}
	}
		
	return A;
}

void print_double_matrix(const vector<vector<double> >& A){
	unsigned int N = A.size();
	unsigned int M = A[0].size();

	for (unsigned int i = 0; i < N; i++){
		for (unsigned int j = 0; j < M; j++){
			if (-EPSILON < A[i][j] && A[i][j] < EPSILON)
				cout << "0 ";
			else
				cout << A[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

bool test_identity(const vector<vector<double> >& A){
	unsigned int N = A.size();
	unsigned int M = A[0].size();

	for (unsigned int i = 0; i < N; i++){
		for (unsigned int j = 0; j < M; j++){
			if (i != j){
				if (!(-EPSILON < A[i][j] && A[i][j] < EPSILON))
					return false;
			}else{
				if (!(1.0 - EPSILON < A[i][j] && A[i][j] < 1.0 + EPSILON))
					return false;
			}
		}
	}
	return true;
}


int main (int argc, char** argv){
	int N = 100;
	
	vector<vector<double> > A = random_matrix(N, N);
	
	cout << "matrix A" << endl;
	cout << A << endl;

	cout << "determinant(A)" << endl;
	double d = determinant(A);
	cout << d << endl << endl;

	if (-EPSILON < d && d < EPSILON){
		cout << "matrix not invertible." << endl;
		return 0;
	}

	vector<vector<double> > inv_A(inverse(A));

	vector<vector<double> > id = A*inv_A;
	if (test_identity(id)){
		cout << "OK" << endl;
	}else {
		cout << "ERROR: this matrix is not the identity" << endl;
		return 1;
	}

	id = inv_A * A;
	if (test_identity(id)){
		cout << "OK" << endl;
	}else {
		cout << "ERROR: this matrix is not the identity" << endl;
		return 1;
	}

	return 0;
}
