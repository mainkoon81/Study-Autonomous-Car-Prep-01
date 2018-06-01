# Study-00-math

#### 1. Introduction to Kalman Filters
A Kalman Filter is an algorithm which uses noisy sensor measurements (and Bayes' Rule) to produce reliable estimates of unknown quantities (like where a vehicle is likely to be in 3 seconds).

#### 2. State and Object Oriented Programming
What is the "state" of a self driving car? What quantities do we need to keep track of when programming a car to drive itself? How roboticists think about "state" and how to use a programming tool called object oriented programming to manage that "state".

#### 3. Matrices and Transformations
When a problem can be framed in the language of matrices, it's often possible to find programmatic solutions which are effective and very fast.

------------------------------------------------------------------------------------------------------------

## 1. Kalman Filters
: Estimate future locations and velocities based on data.
 - CONTINUOUS
 - UNI-MODAL

Performing a **measurement** meant updating our belief by a multiplicative factor, while **moving** involved performing a convolution.
<img src="https://user-images.githubusercontent.com/31917400/40812124-13f41562-652c-11e8-9bae-b4731167c731.jpg" />

 - Markov Model says our world is divided into **discrete grids**, and we assign to each grid a certain probability. Such a representation over spaces is called **histogram**.
 - In Kalman filter, the distribution is given by 'Gaussian':
   - a continuous function over the space of locations
   - the area underneath sums up to 1 
   - Rather than estimating entire distribution as a histogram, we maintain the mu, sigma2 that is our best estimate of the location of our object we want to find.
   
   




















