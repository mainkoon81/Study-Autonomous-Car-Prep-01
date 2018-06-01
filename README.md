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
 
> Intro
Performing a **measurement** meant updating our belief by a multiplicative factor, while **moving** involved performing a convolution.
<img src="https://user-images.githubusercontent.com/31917400/40812124-13f41562-652c-11e8-9bae-b4731167c731.jpg" />

 - In Kalman filter, the distribution is given by 'Gaussian':
   - CONTINUOUS: a continuous function over the space of locations
   - UNI-MODAL
   - the area underneath sums up to 1 
   - Rather than estimating entire distribution as a histogram, we maintain the 'mu', 'variance' that is our best estimate of the location of our object we want to find.

> UPDATE: Measurement & Motion
 - Kalman filter iterates 2 updates- 'measurement', 'motion'. This is identical to the situation before in localization where we got a measurement then we took a motion. 
   - **Measurement:** meant updating our belief (and renormalizing our distribution, using BayesRule; product).
   - **Motion:** meant keeping track of where all of our probability "went" when we moved (using the law of Total Probability; convolution).
<img src="https://user-images.githubusercontent.com/31917400/40843872-4ff26a7c-65aa-11e8-812f-f80f24a6597e.jpg" />

<img src="https://user-images.githubusercontent.com/31917400/40845114-9d457186-65ad-11e8-8e1a-545ee7c4e66f.jpg" />

<img src="https://user-images.githubusercontent.com/31917400/40846947-e0d91e8e-65b2-11e8-80f0-7199a081f09c.jpg" />

 - Localization: All self-driving cars go through the same series of steps to safely navigate through the world. The first step is localization. Before cars can safely navigate, they first use sensors and other collected data to best estimate where they are in the world.
 - The Kalman Filter simply repeats the sense and move (measurement and prediction) steps to localize the car as it’s moving!
<img src="https://user-images.githubusercontent.com/31917400/40847073-3ee63692-65b3-11e8-85ef-72febf5c43d6.png" />
 
 - [The Takeaway]: The beauty of Kalman filters is that they combine somewhat inaccurate sensor measurements with somewhat inaccurate predictions of motion to get a filtered location estimate that is better than any estimates that come from only sensor readings or only knowledge about movement.
 
> STATE
 - In order to actually make a Kalman Filter in a 2d or 3d world (or "state space" in the language of robotics), we will first need to learn more about what exactly we mean when we use this word "state".
 - 'the **state** of system: When we localize our car, we care about only the car's 'position(x)' and 'movement(v)', and they are a set of values.
```
x = 0
velocity = 50
initial_state = [x, velocity]

predicted_state = [150, 50]
```
 - the **state** gives us all the information we need to form predictions about a car's future location. But how to represent and how it changes over time?
<img src="https://user-images.githubusercontent.com/31917400/40849581-393d5822-65ba-11e8-90d0-dbe5439a8cbd.jpg" />

 - [The Takeaway]: In order to predict where a car will be at a future point in time, you rely on a **motion model**.
 - It’s important to note, that no motion model is perfect; it’s a challenge to account for outside factors like wind or elevation, or even things like tire slippage, and so on.

The `predict_state( )` should take in a state and a change in time, dt (ex. 3 for 3 seconds) and it should output a new, predicted state based on a constant motion model. This function also assumes that all units are in `m, m/s, s, etc`. `distance = x + velocity*time`.
```
def predict_state(state, dt):
    predicted_x = state[0] + dt*state[1]
    predicted_vel = state[1] 
    predicted_state = [predicted_x, predicted_vel]
    return predicted_state

test_state = [10, 3]
test_dt = 5
test_output = predict_state(test_state, test_dt)
```
> Two Motion Models(kinematic equations)
 - Constant Velocity(100m/sec): This model assumes that a car moves at a constant speed. This is the simplest model.
 - Constant Acceleration(10m/sec^2): This model assumes that a car is constantly accelerating; its velocity is changing at a constant rate.
<img src="https://user-images.githubusercontent.com/31917400/40864491-fea8438e-65eb-11e8-8c1b-c371faf23a16.png" />
 
#### How much the car has moved?
**Displacement in Constant Velocity Model:** 
 - Displacement can also be thought of as the area under the line within the given time interval.
   - `displacement = initial_velocity*dt`(where dt = t2-t1)
 
**Displacement in Constant Acceleration Model:** 
 - Velocity
   - Changing Velocity over time: `dv = acceleration*dt`
   - the current Velocity: `v = initial_velocity + dv`
 - Displacement can be calculated by finding the area under the line in between t1 and t2. This area can be calculated by breaking this area into two distinct shapes; A1 and A2.
   - A1 is the same area as in the constant velocity model:
     - `A1 = initial_velocity*dt `
   - In A2, the width is our change in time: `dt`, and the height is the **change in velocity over that time**: `dv`.
     - `A2 = 0.5*acceleration*dt*dt`
   - `displacement = initial_velocity*dt + 0.5*dv*dt` 
 
 




















