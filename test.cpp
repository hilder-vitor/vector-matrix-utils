#include "vectorutils.hpp"
#include "matrixutils.hpp"


#define EPSILON 0.00000001

/******************************************************	
 * 	Author: Hilder Vitor Lima Pereira 
 *
 * 	e-mail: hilder.vitor@gmail.com
 * ****************************************************/
int main (int argc, char** argv){
	int N = 5;
	
	vector<vector<double> > A = create_matrix(0.0, N, N);

	for (int i=0; i < N; i++){
		for (int j=0; j < N; j++){
			if (j%3)
				A[i][j] = i - N*j;
			else if (j%3 == 1)
				A[i][j] = i + j;
			else
				A[i][j] = i - j;
		}
		A[i][i] = 10;
	}

	cout << A << endl;

	cout << determinant(A) << endl;
	
	vector<vector<double> > inv_A = inverse(A);
	cout << inv_A << endl;

	vector<vector<double> > id = A*inv_A;
	cout << "A * inv_A" << endl;
	for (unsigned int i = 0; i < N; i++){
		for (unsigned int j = 0; j < N; j++){
			if (-EPSILON < id[i][j] && id[i][j] < EPSILON)
				cout << "0 ";
			else
				cout << id[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	id = inv_A * A;
	cout << "inv_A * A" << endl;
	for (unsigned int i = 0; i < N; i++){
		for (unsigned int j = 0; j < N; j++){
			if (-EPSILON < id[i][j] && id[i][j] < EPSILON)
				cout << "0 ";
			else
				cout << id[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	return 0;
}
