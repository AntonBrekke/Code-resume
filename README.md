# Code resume

## C++
 All C++-files where made in collaberation with a group during the FYS3150-course taken at UiO. 

 Authors of the C++-files include: 
 - Anton Andreas Brekke
 - Johan Luciano Jahre Carlsen 
 - Mads Saua Balto
 - Alireza Asghari

### Ising_model
In this project, we solved the Ising-model using Markov-chain Monte-Carlo (MCMC). 
We used the Metropolis-Hastings algorithm to pull samples from an unknown pdf, by 
using that thermal systems get states from the Boltzmann distribution. We calculated
energies and magnetization, and used OpenMP to parallelize multiple simulations. 

### Schrodinger_equation
In this project, we solved the Schrodinger equation as a PDE in 1+2 dimensions. We
initiate a potential that simulates slits, and simulate double and triple slit experiments. 
The slits were modelled as a large potential in the Schrodinger-equation. 

More detailed README-files is contained in each project.

## Python

### Jupyter

#### Modelling a guitar: Modelling_guitar.ipynb
This project was a computational essay, where the goal is to simulate the sound of a guitar. 
This was done by solving the wave-equation numerically, and then retrieve the amplitudes in the 
sound by doing the Fourier-inner products on the generated signal. From the amplitudes, the sound 
is constructed. This project was done mainly in two parts: Generating a sound, then building on the 
existing model to include damping of the sound. 

### Separating fluids
In this project, I were given data of gas and fluids in a pipe under the sea. From this I separated 
the gas and the fluid, and calculated several properties as curl, divergence, and circulation in some 
specific regions. 

### Single Python-files

#### MCI_multivar.py, MCI_multivar_parallel.py
Simple algoritms for calculating integrals using Monte-Carlo methods. Simple cases without importance 
sampling. Codes work for an arbitrary amount of variables. MCI_multivar_parallel.py runs multiple simulations 
of same integral, giving a distribution of values. 

#### convolution.py 
Calculates convolution between functions. I also made a class which allows to multiply functions before 
evaluation, making fuctions that can interact (mostly for fun). 

#### fourier_series.py
Animates the Fourier-series for any given function. Asks for function as input, which should be written in 
Python-syntax. Numpy functions are allowed written as "np.". Example: "np.cos(x)*np.exp(-x**2)". Asks for 
number of terms in series and basic interval to do the series.  

#### ising_model.py
Toy-model for the Ising_model.cpp files in the C++ folder. Simple and more readable code, but not as detailed. 
Solving the Ising-model in Python using the Markov-chain Monte-Carlo (Metropolis-Hastings) algorithm.

#### move_points_bezier.py
Making bezier-curves in matplotlib, with interactive points that can be dragged aorund to adjust curves. 

#### plasma_globe.py 
Simulation of a plasma-globe by solving Poissions equation numerically for the electric potential in the globe.

#### trebuchet.py 
Simulation of a trebuchet. Lagrangian of system is very complex, and I find the equations of motion by using 
Sympy to solve it symbolically. Then I solve the equations of motion using numerical integration from Scipy. 
Then trebuchet is animated. 

#### wavelet_transformation.py 
Doing a wavelet-transformation on tawny_owl.wav by using a Morlet-wavelet. First I find the frequency spectrum 
by doing a numerical fourier-transform, then I do a wavelet transform to gain where in time the frequencies occur. 
The wavelet-transformation is done quickly by using the fast-fourier-transform algorithm from numpy, and the 
convolution theorem for Fourier-transforms. 

 All Python-files is made by me. 
