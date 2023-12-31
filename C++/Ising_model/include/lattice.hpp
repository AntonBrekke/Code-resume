/*
Header file for initializing lattice
*/
#ifndef __Lattice_hpp__
#define __Lattice_hpp__

#define ARMA_DONT_USE_STD_MUTEX 
#include <armadillo>

arma::mat initialize_lattice(int L, double temp, double& E, double& M,
                             bool ordered);

#endif
