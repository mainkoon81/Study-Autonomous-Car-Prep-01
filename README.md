# Study-00-math

#### 1. Kalman Filters
A Kalman Filter is an algorithm which uses noisy sensor measurements (and Bayes' Rule) to produce reliable estimates of unknown quantities (like where a vehicle is likely to be in 3 seconds).

#### 2. State
What is the "state" of a self driving car? What quantities do we need to keep track of when programming a car to drive itself? How roboticists think about "state" and how to use a programming tool called object oriented programming to manage that "state".

#### 3. Matrices and Transformations of State
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
 
 - [Takeaway]: The beauty of Kalman filters is that they combine somewhat inaccurate sensor measurements with somewhat inaccurate predictions of motion to get a filtered location estimate that is better than any estimates that come from only sensor readings or only knowledge about movement.
 
## 2. STATE
 - In order to actually make a Kalman Filter in a 2d or 3d world (or "state space" in the language of robotics), we will first need to learn more about what exactly we mean when we use this word "state".
 - the **state** of system: When we localize our car, we care about only the car's 'position(x)' and 'movement(v)', and they are a set of values. `state = [position(x), velocity(v)]`
 - the **state** gives us all the information we need to form predictions about a car's future location. But how to predict(how it changes over time) and how to represent?
<img src="https://user-images.githubusercontent.com/31917400/40849581-393d5822-65ba-11e8-90d0-dbe5439a8cbd.jpg" />

The `predict_state( )` should take in a state and a change in time, dt and it should output a new, predicted state based on a constant motion model. This function also assumes that all units are in `m, m/s, s, etc`, and `distance = x + velocity*dt`.
```
def predict_state(state, dt):
    predicted_x = state[0] + dt*state[1]
    predicted_vel = state[1] 
    predicted_state = [predicted_x, predicted_vel]
    return(predicted_state)

test_state = [10, 3]
test_dt = 5
test_output = predict_state(test_state, test_dt)
```
 - [Takeaway]: In order to predict where a car will be at a future point in time, you rely on a **motion model**.
 - It’s important to note, that no motion model is perfect; it’s a challenge to account for outside factors like wind or elevation, or even things like tire slippage, and so on.
> Two Motion Models(kinematic equations)
 - Constant Velocity(100m/sec): This model assumes that a car moves at a constant speed. This is the simplest model.
 - Constant Acceleration(10m/sec^2): This model assumes that a car is constantly accelerating; its velocity is changing at a constant rate.
<img src="https://user-images.githubusercontent.com/31917400/40864491-fea8438e-65eb-11e8-8c1b-c371faf23a16.png" />
 
#### # How much the car has moved?
**Displacement in Constant Velocity Model:**
 - Velocity
   - the current Velocity: `v = initial_velocity`
 - Displacement can also be thought of as the area under the line within the given time interval.
   - `displacement = initial_velocity*dt`(where 'dt=t2-t1')

Predicted state after 3 seconds have elapsed: this state has a new value for x, but the same value for velocity.  
```
x = 0
initial_velocity = 50
initial_state = [x, velocity]

dt = 3
new_x = x + initial_velocity*dt
predicted_state = [new_x, initial_velocity]  
``` 
**Displacement in Constant Acceleration Model:**
 - Acceleration
 - Velocity
   - Changing Velocity over time: `dv = acceleration*dt`
   - the current Velocity: `v = initial_velocity + dv`
 - Displacement can be calculated by finding the area under the line in between t1 and t2. This area can be calculated by breaking this area into two distinct shapes; A1 and A2.
   - A1 is the same area as in the constant velocity model:
     - `A1 = initial_velocity*dt `
   - In A2, the width is our change in time: `dt`, and the height is the **change in velocity over that time**: `dv`.
     - `A2 = 0.5*acceleration*dt*dt`
   - `displacement = initial_velocity*dt + 0.5*dv*dt` 

Predicted State after 3 seconds have elapsed: this state has a new value for x, and a new value for velocity (but the acceleration stays the same).   
```
x = 0
initial_velocity = 50
acc = -20
initial_state = [x, initial_velocity, acc]

dt = 3
new_x = x + initial_velocity*dt + 0.5*acc*dt**2
new_vel = velocity + acc*dt
predicted_state = [new_x, new_vel, acc] 
```
#### # Then back to the question: 
 - How to represent **State**? : object-oriented programming
   - Using variables to represent State values
   - Using customized function to change those values
 - How to predict **State**?: Linear Algebra  
   - Using vector, matices to keep track of State and change it.
#### # Always moving
Self-driving cars constantly monitor their **state**. So, movement(v) and localization(x) have to occur in parallel. If we use a Kalman filter for localization, this means that as a car moves, the Kalman filter has to keep coming up with new state estimates.

Here, `predict_state( )` we wrote previously takes in a **current state** and a change in time, **dt**, and returns the new state estimate(based on a constant velocity model): [10, 60] -> [130, 60] -> [310, 60] -> [370, 60] -> [610, 60]
```
initial_state = [10, 60]
state_est1 = predict_state(initial_state, 2)
state_est2 = predict_state(state_est1, 3)
state_est3 = predict_state(state_est2, 1)
state_est4 = predict_state(state_est3, 4)
```
#### # Object
Objects hold a state; they hold a group of variables/properties and functions.
<img src="https://user-images.githubusercontent.com/31917400/40880031-e6666272-66a1-11e8-998b-79bff37b4b29.jpg" />

Step_1: import the statement(car-'class' file) and make a 2D world of 0's, then declare a car's **initial state** variable.
 - initial_position: [y, x] (top-left corner)
 - initial_velocity: [vy, vx] (moving to the right)
```
import numpy
import car

height = 4
width = 6

world = np.zeros((height, width))
initial_position = [0, 0] 
initial_velocity = [0, 1] 
```
Step_2: initializes the object, and pass in the initial state variables.
```
car_object = car.Car(initial_position, initial_velocity, world)
```
 - `car`: the name of the file
 - `Car()`: initializing function
 - `self` means the object
 
Step_3: interact with the object...`car_object.move()` , `car_object.turn_left()`, `car_object.display_world()`...

### A. How to represent State?
Have you ever seen the package inside `import car`? What is class?
<img src="https://user-images.githubusercontent.com/31917400/40885090-bafb1800-6717-11e8-8408-9934121743a8.jpg" />

 - **class** allows us a bunch of codes like `car.Car()`, `__init__`, etc.
 - `__init__` stands for initialize(it frees up memory) and allows us **to create a specific object**. The object can then access all of the functions that are inside the class like `move()` or `turn_left()`. The code right below `__init__` describe what will happen when we creat the object. 
 - Detail:
   - `class **Car**(object)`: this looks a bit like a function declaration, but the word "class" let Python know that the code that follows should describe the **state and functionality** of the object. Objects are always capitalized, like 'Car'. 
   - `__init__` function is responsible for creating space in memory to make a specific object, and it is where **initial state variable** are set with statements like `self.state = [position, velocity]`. 
   - `move()` function uses a constant velocity model to move the car in the direction of its velocity, vx, and vy, and it **updates the state**. It mainly offers **'dt'**.
   
For example, in the `move()`  
```   
    def move(self, dt=1):
        height = len(self.world)
        width = len(self.world[0])
        
        position = self.state[0]
        velocity = self.state[1]

        predicted_position = [(position[0] + velocity[0]*dt) % height, (position[1] + velocity[1]*dt) % width]
        
        # Update the state..where "velocity = self.state[1]" ## **[vy, vx]** always ##
        self.state = [predicted_position, velocity]
        
        # Every time the robot moves, add the new position to the path
        self.path.append(predicted_position)
```
   - `turn_left()`: ㄱ(vy -> -vx), and (vx -> vy) function rotates the velocity values to the left 90 degrees, and it **updates the state**. `turn_right()`: r(vy -> vx), and (vx -> -vy) to the right 90 degrees, and it **updates the state**.  
```
    def turn_left(self):
        velocity = self.state[1]
        predicted_velocity = [-velocity[1], velocity[0]]
        
        self.state[1] = predicted_velocity
    
    def turn_right(self):
        velocity = self.state[1]
        predicted_velocity = [velocity[1], -velocity[0]]
        
        self.state[1] = predicted_velocity
```
> FYI, Overloading:
   - The **double underscore** function: (`__init__`, `__repr__`, `__add__`, etc) https://docs.python.org/3/reference/datamodel.html#special-method-names These are special functions that are used by Python in a specific way. We typically don't call these functions directly. Instead, Python calls them automatically based on our use of keywords and operators. For example, `__init__` is called when we create a new object and `__repr__` is called when we tell Python to print the string representation of a specific object.
   - We can define what happens when we add two car objects together using a `**+**` symbol by defining the `__add__` function. 

For example, when we add up two '**object**'s, this below will happen).
```
def __add__(self, other):
    added_state = []
    for i in range(self.state):
        added_value = self.state[i] + other.state[i]
        added_state.append(added_value)

    return(added_state)
```    
Or..Print an error message? and return the unchanged, first state. 
```
def __add__(self, other):
    print('Adding two cars is an invalid operation!')
    return self.state
```
This is called **operator overloading**. And, in this case, overloading just means: giving **more than one meaning** to a standard operator like (+,-,/,%,stc). It is useful for writing classes.  

For example, overloading 'color addition'. The color **class** creates a color from 3 values, r, g, and b (red, green, and blue).
<img src="https://user-images.githubusercontent.com/31917400/40889632-5f10cfa8-6762-11e8-8916-7a44ba9d80fc.jpg" />

This will give..
<img src="https://user-images.githubusercontent.com/31917400/40889751-14146b66-6764-11e8-8081-4cc4a8c48356.jpg" />

### B. State vector and Matrix
<img src="https://user-images.githubusercontent.com/31917400/40891523-da44fc78-677e-11e8-960b-d04c9afebfd3.jpg" />

<img src="https://user-images.githubusercontent.com/31917400/40908445-ab1ce55e-67de-11e8-9a66-718c7047de98.gif" />
In the world of KalmanFilter(multivariate Gaussian)...we can build a 2-Dimensional Estimate because of the correlation b/w location and velocity.
 
## x_prime = x + x_dot*delta_t
<img src="https://user-images.githubusercontent.com/31917400/40915010-5bc1e540-67f2-11e8-97c3-0db98a7ce266.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/40919806-7d71cc0e-6802-11e8-9251-75e3358d0d5a.jpg" />

### So How are you gonna write object tracking code? How to design Kalman filter?
Kalman filtering, also known as linear quadratic estimation (LQE), is an algorithm that uses a series of measurements observed over time, containing statistical noise and other inaccuracies, and **produces estimates of unknown variables** by estimating a joint probability distribution over the variables for each timeframe.

> Representing State with Matrices








