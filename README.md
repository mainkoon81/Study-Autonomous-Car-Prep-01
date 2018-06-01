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
 - Markov Model says our world is divided into **discrete grids**, and we assign to each grid a certain probability. Such a representation over spaces is called **histogram**.

Performing a **measurement** meant updating our belief by a multiplicative factor, while **moving** involved performing a convolution.
<img src="https://user-images.githubusercontent.com/31917400/40812124-13f41562-652c-11e8-9bae-b4731167c731.jpg" />

 - In Kalman filter, the distribution is given by 'Gaussian':
   - CONTINUOUS: a continuous function over the space of locations
   - UNI-MODAL
   - the area underneath sums up to 1 
   - Rather than estimating entire distribution as a histogram, we maintain the 'mu', 'variance' that is our best estimate of the location of our object we want to find.

> Measurement & Motion
 - Kalman filter iterates 2 updates- 'measurement', 'motion'. This is identical to the situation before in localization where we got a measurement then we took a motion. 
   - **Measurement:** meant updating our belief (and renormalizing our distribution, using BayesRule; product).
   - **Motion:** meant keeping track of where all of our probability "went" when we moved (using the law of Total Probability; convolution).
<img src="https://user-images.githubusercontent.com/31917400/40843318-7ce1a914-65a8-11e8-942a-f0a1086f046c.jpg" />
   
   




















