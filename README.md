# 00-Theory

> **Sensors**
We gather data from the car's sensors. Self-driving cars mainly use three types of sensors to observe the world:
 - Camera, which records video,
 - Lidar, which is a light-based sensor, and
 - Radar, which uses radio waves.
All of these sensors detect surrounding objects and scenery. Autonomous cars also have lots of internal sensors that measure things like the speed and direction of the car's movement, the orientation of its wheels, and even the internal temperature of the car.

> **Reducing Uncertainty**
 - Self-D-car measures 'direction','speed','location','scenary','surrounding objects', etc...with sensors. And this sensor measurement are not perfect, so when the information they provide is combined(using BayesRule), we can form a reliable representation - movement, position, environment.
 - Probability Distributions are a mathematical way to represent the uncertainty across all possible outcomes(the system).
   - It can be visualized using a graph especially in 2-dimensional cases.
   - It makes it much easier to understand and summarize the probability of a system whether that system be a coin flip experiment or the location of a self-driving car.

--------------------------------------------------------------------------------------------------------------------------------------
#### 1. Kalman Filters
A Kalman Filter is an algorithm which uses noisy sensor measurements (and Bayes' Rule) to produce reliable estimates of unknown quantities (like where a vehicle is likely to be in 3 seconds).

#### 2. State
What is the "state" of a self driving car? What quantities do we need to keep track of when programming a car to drive itself? How roboticists think about "state" and how to use a programming tool called object oriented programming to manage that "state".

#### 3. Matrices and Transformations of State
When a problem can be framed in the language of matrices, it's often possible to find programmatic solutions which are effective and very fast.

#### 4. Problem Solving theory

#### 5. The Search Problem

#### 6. Intro to Computer Vision

------------------------------------------------------------------------------------------------------------

## 1. Kalman Filters
: Estimate future locations and velocities based on data.
 - Markov Model says our world is divided into **discrete grids**, and we assign to each grid a certain probability. Such a representation over spaces is called **histogram**.
 
> Intro
 - **Performing a measurement** meant updating our belief by a multiplicative factor, 
 - while **moving** involved performing a convolution.
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

<img src="https://user-images.githubusercontent.com/31917400/43492880-fc9d8e8a-9522-11e8-88ef-810170ab4cc0.jpg" />

 - The Kalman Filter simply repeats the sense and move (measurement and prediction) steps to localize the car as it’s moving!
<img src="https://user-images.githubusercontent.com/31917400/40847073-3ee63692-65b3-11e8-85ef-72febf5c43d6.png" />
 
 - [Takeaway]: The beauty of Kalman filters is that they **combine** somewhat inaccurate **sensor measurements** with somewhat inaccurate **predictions of motion** to get a filtered location estimate that is better than any estimates that come from only sensor readings or only knowledge about movement.
 
## 2. STATE

When you localize a car, you’re interested in only the car’s **position** and it’s **movement**. This is often called the **state** of the car. The **state of any system** is a **set of values** that we care about. In order to actually make a Kalman Filter in a 2d or 3d world (or "state space" in the language of robotics), we will first need to learn more about what exactly we mean when we use this word "state". In our case, the state of the car includes the car’s current position, x, and its velocity, v. (how about time????)  `state = [position_x, velocity_v]`. The **state** gives us all the information we need to form predictions about a car's future location. Self-driving cars constantly monitor their **State**. So, movement(v) and localization(x) have to occur in parallel. If we use a Kalman filter for localization, this means that as a car moves, the Kalman filter has to keep coming up with new state estimates.
# Q. But how to predict(how it changes over time) and how to represent?
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
 
**TYPE_1> Displacement in Constant Velocity Model:**
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
**Here comes the 't'**
 - Here, `predict_state( )` we wrote previously takes in a **current state** and a change in time, **dt**, and returns the new state estimate(based on a constant velocity model): [10, 60] -> [130, 60] -> [310, 60] -> [370, 60] -> [610, 60]
```
initial_state = [10, 60]
state_est1 = predict_state(initial_state, 2)
state_est2 = predict_state(state_est1, 3)
state_est3 = predict_state(state_est2, 1)
state_est4 = predict_state(state_est3, 4)
```

**TYPE_2> Displacement in Constant Acceleration Model:**
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
# Then back to the question: 
 - How to represent **State**? : [object-oriented programming]
   - Using [variables] to represent State values
   - Using [customized_function] to change those values
 - How to predict **State**?: Linear Algebra  
   - Using vector, matices to keep track of State and change it.

# So how to represent State?
#### # Object
Objects hold a state; they hold a group of variables/properties and functions.
<img src="https://user-images.githubusercontent.com/31917400/40880031-e6666272-66a1-11e8-998b-79bff37b4b29.jpg" />

#### # How it works? 
[Typical-Step_1]: import the car-'class' file(car.py) and make a 2D world of 0's(which is our map), then declare a car's **initial state** variable. A velocity also has vertical and horizontal components as a position does.
 - initial_position: `[y, x]` (let's say, top-left corner)
 - initial_velocity: `[v, v]` (let's say, moving to the right)
```
import numpy
import car

height = 4
width = 6
world = np.zeros((height, width))

initial_position = [0, 0] 
initial_velocity = [0, 1] 
```
[Typical-Step_2]: initializes the object, and pass in the initial state variables.
```
car_object = car.Car(initial_position, initial_velocity, world)
```
 - `car`: the name of the file
 - `Car()`: initializing function(defined by `class Car(object):` in the `car.py` file) 
 
[Typical-Step_3]: interact with the object...`car_object.move()` , `car_object.turn_left()`, `car_object.display_world()`...

#### # Let's see the detail. How to represent State?
How `import car` works? What is class?
<img src="https://user-images.githubusercontent.com/31917400/40885090-bafb1800-6717-11e8-8408-9934121743a8.jpg" />

 - `__init__` stands for initialize(it frees up memory) and allows us **to create a specific object**(we can call it 'car_object' as above, or 'carla', or 'your mom'). The object can then access all of the functions that are inside the class like `move()` or `turn_left()`. The code right below `__init__` describe what will happen when we creat the object. 
 - Detail:
   - `class Car(object):`: the word "class" let Python know that the code that follows should describe the **functionality** of the object. Objects are always capitalized, like 'Car'. 
   - `self`: the object
   - `__init__` function is responsible for creating space in memory to make a specific object, and it is where **initial variables**(state, world, color, path) are set. For example....
     - `self.state = [position, velocity]`: This is the moment where the **initial state** is created from the position vector and velocity vector that are passed in.   
     - `self.world = world`: initial world (will be passed in too). 
     - `self.color = 'r'`: initial color. That's why our car appears red in the grid world.
     - `self.path = []`, `self.path.append(position)`: it's gonna be a list of locations that our car visit. 

All of variables in here are something that our **car_object** keeps track of.  

Here, `move()` uses a constant velocity model(we discussed previously) to move the car in the direction of its velocity(v_x and v_y), and it updates the **state**. **'dt'** is offered?
For example, in the `move()`,   
```   
    def move(self, dt=1):
        height = len(self.world) ## rows in the grid
        width = len(self.world[0]) ## col in the grid
        
        position = self.state[0]
        velocity = self.state[1]

        predicted_position = [(position[0] + velocity[0]*dt) % height, (position[1] + velocity[1]*dt) % width]
        
        # Update the state..where "velocity = self.state[1]" ## **[v_y, v_x]** always ##
        self.state = [predicted_position, velocity]
        
        # Every time the robot moves, add the new position to the path
        self.path.append(predicted_position)
```
   - `turn_left()`: ㄱ(v_y -> -v_x), and (v_x -> v_y) function rotates the velocity values to the left 90 degrees, and it updates the **state**. `turn_right()`: r(v_y -> v_x), and (v_x -> -v_y) to the right 90 degrees, and it updates the **state**.  
```
    def turn_left(self):
        velocity = self.state[1]
        predicted_velocity = [-velocity[1], velocity[0]]
        
        self.state[1] = predicted_velocity
    
    def turn_right(self):
        velocity = self.state[1]
        predicted_velocity = [velocity[1], -velocity[0]]
        
        self.state[1] = predicted_velocity
        
    # Helper function for displaying the world + robot position assumes the world in a 2D numpy array and position is in the form [y, x]. path is a list of positions, and it's an optional argument..
    
    def display_world(self):
        
        # Store the current position of the car
        position = self.state[0]
        
        # Plot grid of values + initial ticks
        plt.matshow(self.world, cmap='gray')

        # Set minor axes in between the labels
        ax=plt.gca()
        rows = len(self.world)
        cols = len(self.world[0])

        ax.set_xticks([x-0.5 for x in range(1,cols)],minor=True )
        ax.set_yticks([y-0.5 for y in range(1,rows)],minor=True)

        # Plot grid on minor axes in gray (width = 2)
        plt.grid(which='minor',ls='-',lw=2, color='gray')

        # Create a 'x' character that represents the car
        # ha = horizontal alignment, va = verical
        ax.text(position[1], position[0], 'x', ha='center', va='center', color=self.color, fontsize=30)
            
        # Draw path if it exists
        if(len(self.path) > 1):
            # loop through all path indices and draw a dot (unless it's at the car's location)
            for pos in self.path:
                if(pos != position):
                    ax.text(pos[1], pos[0], '.', ha='center', va='baseline', color=self.color, fontsize=30)

        # Display final result
        plt.show()       
```
#### # FYI, Overloading:
 - The **double underscore** function: (`__init__`, `__repr__`, `__add__`, etc) https://docs.python.org/3/reference/datamodel.html#special-method-names These are special functions that are used by Python in a specific way. We typically don't call these functions directly. Instead, Python calls them automatically based on our use of keywords and operators. For example, 
   - `__init__` is called when we create a new object.
   - `__repr__` is called when we tell Python to print the string representation of a specific object.
   - We can define what happens when we add two car objects together using a `+` symbol by defining the `__add__` function. 

For example, when we add up two '**object**'s, this below will happen).
```
def __add__(self, other):
    added_state = []
    for i in range(self.state):
        added_value = self.state[i] + other.state[i]
        added_state.append(added_value)
    return(added_state)
```    
Or..you may choose to just print an error message? and return the unchanged-first state. 
```
def __add__(self, other):
    print('Adding two cars is an invalid operation!')
    return self.state
```
This is called **operator overloading**. And, in this case, overloading just means: giving **more than one meaning** to a standard operator like (+, -, /, %,etc). It is useful for writing classes.

For example, overloading 'color addition'. The color **class** creates a color from 3 values, r, g, and b (red, green, and blue).
<img src="https://user-images.githubusercontent.com/31917400/43522437-47cf54c4-9591-11e8-8787-cb4554d13be6.jpg" />

This will give..
<img src="https://user-images.githubusercontent.com/31917400/40889751-14146b66-6764-11e8-8081-4cc4a8c48356.jpg" />

# then how to predict State?
### B. State vector and Matrix
<img src="https://user-images.githubusercontent.com/31917400/40891523-da44fc78-677e-11e8-960b-d04c9afebfd3.jpg" />

<img src="https://user-images.githubusercontent.com/31917400/40908445-ab1ce55e-67de-11e8-9a66-718c7047de98.gif" />
In the world of KalmanFilter(multivariate Gaussian)...we can build a 2-Dimensional Estimate because of the correlation b/w location and velocity.
 
## x_prime = x + x_dot*delta_t
<img src="https://user-images.githubusercontent.com/31917400/40915010-5bc1e540-67f2-11e8-97c3-0db98a7ce266.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/40919806-7d71cc0e-6802-11e8-9251-75e3358d0d5a.jpg" />

### So How are you gonna write object tracking code? How to design Kalman filter?
Kalman filtering, also known as linear quadratic estimation (LQE), is an algorithm that uses a series of measurements observed over time, containing statistical noise and other inaccuracies, and **produces estimates of unknown variables** by estimating a joint probability distribution over the variables for each timeframe.
<img src="https://user-images.githubusercontent.com/31917400/40940038-68b1b900-683e-11e8-83a3-bf03de561fe0.jpg" />


> Representing State with Matrices
Matrices provide a very convenient and compact form for representing a vehicle's state. Matrices do all of the calculations in just one step. In the constant velocity motion model, and in a two-dimensional world, the state could have:
```
state = [distance_x, distance_y, velocity_x, velocity_y, steering_angle, angular_velocity]
```
### # vector calculation
Vectors are one part of the Kalman filter equations. In python, a vector can be a simple grid with one row and a column for each element. What if transposed? You can think of a vector as a simple list of values even if the vector is vertical.

1.Vector Math: **Addition**
 - You are tracking the other vehicle. Currently, the other vehicle is 5 meters ahead of you along your x-axis, 2 meters to your left along your y-axis, driving 10 m/s in the x direction and 0 m/s in the y-direction.
 - The vehicle has moved 3 meters forward in the x-direction, 5 meters forward in the y-direction, has increased its x velocity by 2 m/s and has increased its y velocity by 5 m/s.
```
x0 = [5,2,10,0]
xdelta = [3,5,2,5]

x1 = []
for i in range(len(x0)):
    add = x0[i] + xdelta[i]
    x1.append(add)
```
2.Scalar Math: multiplication
 - You have your current position in meters and current velocity in m/s. But you need to report your results at a company meeting where most people will only be familiar with working in ft rather than meters. Convert your position vector x1 to feet and feet/second.
```
meters_to_feet = 1.0 / 0.3048

x1feet =[]
for i in range(len(x1)):
    x1feet.append(meters_to_feet*x1[i])
```
3.Vector Math: **Dot-Product** 
<img src="https://user-images.githubusercontent.com/31917400/40938492-f084d0ec-6839-11e8-9ab0-a1aa387699f8.jpg" />

 - It involves mutliplying the vectors element by element and then taking the sum of the results: **element-wise multiplication**. 
 - The tracked vehicle is currently at the state: X1 = [**8**,**7**,12,5]. Where will the vehicle be in two seconds(assuming the constant velocity)?
   - the new x-position: 8 + 12*2sec = 32
   - the new y-position: 7 + 5*2sec = 17
   - if solving each of these equations using the dot product:
     - [8,7,12,5].[1,0,2,0] = 32
     - [8,7,12,5].[0,1,0,2] = 17
   - the final state vector would be: [**32**,**17**,12,5]
```
def dotproduct(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        print("error! Vectors must have same length")
    
    result = 0
    for i in range(len(vector_a)):
        result += vector_a[i]*vector_b[i]
    return(result)
    
x2 = [dotproduct([8, 7, 12, 5], [1, 0, 2, 0]), dotproduct([8, 7, 12, 5], [0, 1, 0, 2]), 12, 5]
```
### # matrix calculation
What is a matrix? Selection: in the second row last column? `matrix[1][-1]`
```
first_row = [17, 25, 6, 2, 16]
second_row = [6, 1, 8, 4, 22]
third_row = [17, 8, 54, 15, 65]
fourth_row = [11, 25, 68, 9, 2]

matrix = [first_row, second_row, third_row, fourth_row]
```
If you are treating the vector as a matrix, then you'd need to make a list of lists. And this is a `1 x 5` matrix. We can transpose to `5 x 1` as well.
```
first_matrix = [ [17, 25, 6, 2, 16] ]
first_matrix_T = [ [17],[25],[6],[2],[16] ]
```
Looping through Matrices: Because matrices are lists of lists, you will need to use a 'for-loop' inside another for loop. The **outside** for loop iterates over the **rows** and the **inside** for loop iterates over the **columns**.
```
for i in range(len(marix)):
    row = matrix[i]
    new_row = [] 
    for j in range(len(row)):
         new_ij = matrix[i][j]
```
1. Scalar Math: multiplication (try 'x5')
 - Note the location of `new_row = []`. Where it is initialized, When the first iteration ends and is stored up.  
```
new_m = []

for i in range(len(m)):
    row = m[i]
    new_row = [] ## Note ##
    for j in range(len(row)):
        new_ij = 5*m[i][j]
        new_row.append(new_ij)
    new_m.append(new_row) ## Note ##
    
def matrix_print(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(matrix[i][j], '\t', end='')
        print('\n')
    return
    
matrix_print(new_m)    
```
2. Matrix Math: **Addition**
 - In the Kalman filter, the equation for calculating the **error covariance matrix** after the prediction step includes matrix addition. It means they are all the same size. 
<img src="https://user-images.githubusercontent.com/31917400/40942772-1acbcdee-6847-11e8-929e-708326017a05.jpg" />

```
S[0][0] = A[0][0] + B[0][0]
S[0][1] = A[0][1] + B[0][1]
S[0][2] = A[0][2] + B[0][2]
S[0][3] = A[0][3] + B[0][3]
.... etc

S[1][0] = A[1][0] + B[1][0]
S[1][1] = A[1][1] + B[1][1]
...etc.
```
or
```
def matrix_addition(matrixA, matrixB):
    if len(matrixA) != len(matrixB):
        print("error! matrices must have same length")

    matrixSum = []
 
    for i in range(len(matrixA)):
        row = [] ## Note ##
        for j in range(len(matrixA[i])):
            r_ij = matrixA[i][j] + matrixB[i][j]
            row.append(r_ij)
        matrixSum.append(row) ## Note ##
    return(matrixSum)
```
3. Matrix Math: **Multiplication(using `get_row()`, `get_column()` for programming)**
 - In the Kalman filter, every equation involves a matrix multiplication operation. In matrix multiplication, we calculate the **dot-product** of row one of A and column one of B. 
```
def get_row(matrix, row):
    return(matrix[row])

def get_column(matrix, col):
    column = []
    for i in range(len(matrix)):
        column.append(matrix[i][col])
    return(column)

def dot_product(vector_one, vector_two):
    result = 0
    for i in range(len(vector_one)):
        result += vector_one[i]*vector_two[i]
    return(result)

def matrix_multiplication(matrixA, matrixB):
    m_rows = len(matrixA)
    p_col = len(matrixB[0])
    result = []

    for i in range(m_rows):
        row_result = []
        for j in range(p_col):
            vec_1 = get_row(matrixA, i)
            vec_2 = get_column(matrixB, j)
            dott = dot_product(vec_1, vec_2)
            row_result.append(dott)
        result.append(row_result)
    return(result)
```
4. Matrix Math: **Multiplication(using `transpose()` for programming)**
 - In the Kalman filter, there are equations that required the transpose of a matrix. We don't need `get_row()` and `get_column()` functions anymore because the tranpose essentially takes care of turning columns into row vectors.
<img src="https://user-images.githubusercontent.com/31917400/40970148-a678eee8-68b1-11e8-9105-8c9eb70b85b5.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/40974805-a21d2698-68c0-11e8-8f89-cc9153729400.jpg" />

```
def transpose(matrix):
    matrix_transpose = []

    for j in range(len(matrix[0])):
        new_row = []
        for i in range(len(matrix)):
            new_row.append(matrix[i][j])
        matrix_transpose.append(new_row)
    return(matrix_transpose)

def dot_product(vector_one, vector_two):
    result = 0
    for i in range(len(vector_one)):
        result += vector_one[i]*vector_two[i]
    return(result)


def matrix_multiplication(matrixA, matrixB):
    
    matrixBT = transpose(matrixB)
    mrows = len(matrixA)
    pcol = len(matrixBT)
    result = []
    
    for i in range(mrows):
        new_row =[]
        for j in range(pcol):
            new_val = dot_product(matrixA[i], matrixBT[j])
            new_row.append(new_val)
        result.append(new_row)
    return(result)
```
5. **Identity Matrix**
 - **I** is an `n x n` **sqr-matrix** with `1` across the main diagonal and `0` for all other elements:`np.eye(n)` 
 - Identity Matrix is like the '1' in scalar world: `AI = IA = A`where although the identity matrix is always square, matrix 'A' does not have to be square. 
```
def identity_matrix(n):
    m=[[0 for x in range(n)] for y in range(n)]
    
    for i in range(n):
        m[i][i] = 1
    return m
```
6. **Inverse Matrix** 
 - When calculating the Kalman filter gain matrix **K**, you will need to take the inverse of the **S** matrix.
 - In linear algebra, the inverse matrix is analogous to the scalar inverse: `np.linalg.inv(matrix)`
 - Only square matrices(nxn) have **inverses**(identity matrix is always a square matrix too), but at the same time, not all square matrices have inverses.  
<img src="https://user-images.githubusercontent.com/31917400/40980824-48bc6f44-68d1-11e8-8d3c-0303332435c4.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/40981273-7b550050-68d2-11e8-9ab2-8cd92959d606.jpg" />

```
def transposeMatrix(m):
    return(map(list,zip(*m)))

def getMatrixMinor(m,i,j):
    return([row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])])

def getMatrixDeternminant(m):
    if len(m) == 2:  ## base case for 2x2 matrix ##
        return(m[0][0]*m[1][1]-m[0][1]*m[1][0])
    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return(determinant)

def getMatrixInverse(m):
    if len(m) != len(m[0]):
        raise ValueError('The matrix must be square')
    determinant = getMatrixDeternminant(m)
    if len(m) == 2:   ## special case for 2x2 matrix ##
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return(cofactors)
```

## 4. Problem Solving


## 5. The Search Problem(Route-finding)
> 1. **Breadth-first Search**
 - **TreeSearch**
 - **GraphSearch**
 - Here, the optimal means finding the shortest path. 
<img src="https://user-images.githubusercontent.com/31917400/43773718-973e68a4-9a3e-11e8-8f7c-c16633974e2d.jpg" />

we are looking ahead to a modified algorithm that keeps track of **explored states** so that they aren't repeated. In the preliminary algorithm, **A(Arad) is repeated** since we are not keeping track of **explored states**. Ideally, we would not add duplicates from backtracking...(picking Arad is backtracking).
<img src="https://user-images.githubusercontent.com/31917400/43776822-fa6281ae-9a48-11e8-9a98-a3a10cfbe4c0.jpg" />

 - 1. As exploring the state, we keep track of the **frontier states**. 
 - 2. Behind the frontier is the set of the **explored states**. 
   - We tracking it coz when expanding, we need to detect duplicates.
 - 3. Ahead of the frontier is the set of the unexplored states. 

But we can avoid this repeated-path problem. The GraphSearch can eliminate the duplicates. 
<img src="https://user-images.githubusercontent.com/31917400/43787888-fdc8dcba-9a63-11e8-8607-072e0e64f00f.jpg" />

Then Fagaras leads to our destination and we don't add the path going back because they are in the 'explored_list'. We add the path that ends in Bucharest, but we don't terminate yet. The **`GoalTest()`** is not applied when we add a path to the [frontier_list]. It's applied **when we remove that path** from the [frontier_list]. 
 - Why doesn't our 'TreeSearch' or 'GraphSearch' stop when it adds a **goal_node** to the frontier_list? coz there is no guarantee that it is the best path to the goal. It needs **'optimization' in terms of**..?
   - If we want the shortest path in terms of `NO.of steps`?
   - If we want the shortest path in terms of `total cost`(by adding up the step costs)
   - etc...

> **2. Uniform Cost Search(cheapest-first-search)**
 - Here, optimal means finding the cheapest total-path cost and the shortest path
<img src="https://user-images.githubusercontent.com/31917400/43790205-ce291ba4-9a69-11e8-9f78-4e7a5e75a66a.jpg" />

 - When we've reached the goal-state, we put the **path** onto the **frontier_list** that reaches the destination, but do not stop yet. The algorithm continues to search to find the better path until we pop it off the **frontier_list**. Which path reaches first is not important.

> **3. Depth-first Search**
 - The opposite of Breadth-first Search: 
   - Always expand first the longest path, which is the path with the most links in it. 
   - Then backs up 
 - Here, optimal means finding the longest path? No, it's not optimal...then why we use it?
   - because of the **'storage requirement'**
     - if the tree is infinite, if our destination(goal) is placed at any finite level, Breadth-first Search and Uniform-Cost Search will march down and find it. But not so for Depth-first Search. 

> **4. A-Star Search**
 - In Uniform-Cost Search, just like a topological map, it expands out to a certain distance, then to the farther, farther..until meet up witht the goal. The search is not really directed at any way towards the goal, it is expanding out everywhere in the space. Depending on where the goal is, and the size of the space, it takes too much time to get the goal. If we want to find it faster, we need to add more knowledge - the **estimate of the distance** from the start state to the goal.
 - **Greedy-best-firtst Search**: 
   - It expands the path closest to the goal according to the estimate. (but this not always be the case if there are obstacles along the way. When it reached the barrier..
     - continuosly expand out along the barrier to get closer and closer to the goal (it is willing to accept the path longer than other path). 
     - or explore in the other direction to find much simpler path by popping over the barrier. ??????
<img src="https://user-images.githubusercontent.com/31917400/43803586-2eb3dfe6-9a91-11e8-91c4-d62815547720.jpg" />

**WTF A-Star Search ?**(Best Estimated total-path-cost First Search):
 - Always expands the path that has a minimum value of the function `f` defined as a sum of `g`+`h` 
<img src="https://user-images.githubusercontent.com/31917400/43833532-f78cad5a-9b02-11e8-8197-676f6731de1e.jpg" />
 
 - Does this search always find the lowest-cost path? No, it depends on `h`function. So `h`function should not over-estimate the distance to the goal. 
   - `h` is **optimistic**
   - `h` is **admissible**(never over-estimate). 
 - Why **optimistic**`h` finds the lowest-cost path? 
   - First, when the algorithm returns the final path with the estimated cost(C), we know that by this moment, `h=0`, thus the estimated cost(C) become the actual cost. 
   - All the paths on the frontier have an estimated cost greater than C(coz all frontiers were explored in cheapest-first order).   
<img src="https://user-images.githubusercontent.com/31917400/43834083-cab9d9d6-9b04-11e8-99de-4f3cd91cdfc3.jpg" />

## 6. Intro to Computer Vision
Our machine can visually perceive the world and respond to it. It gathers data through sensors, cameras. It extracts important information(color, shape, etc) from it. Sensors are broken down into active(RADAR, LiDAR) / passive(camera). These sensor details have serious repercussions with respect to the types of algorithms we end up using to analyze this data. Machine learning is used in combination with computer vision to give machines a way to learn from data and recognize patterns in images.
<img src="https://user-images.githubusercontent.com/31917400/43929930-d84950a6-9c2e-11e8-9f52-5b98545022f8.png" />

> LiDAR: 'Light Detection and Ranging' is a type of sensor that uses light (a laser) to measure the distance between itself and objects that reflect light. It sends out pulses of light...The longer the reflection takes, the farther and object is from the sensor. In this sense, LiDAR is spatially coherent data, and can be used to create a visual world representation. Since LiDAR uses laser light, which sends out a thin light beam, the data it collects ends up being many single points also called **point clouds**. These point clouds can tell us a lot about an object like its shape and surface texture. By clustering points and analyzing them, this data provides enough information to classify an object or track a cluster of points over time! 

> Image Classification Pipeline
<img src="https://user-images.githubusercontent.com/31917400/43930455-611e3156-9c31-11e8-98ce-aa507b51a88e.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/43965087-434528a8-9cb6-11e8-8f40-c4ca38f72177.jpg" />

### **Preprocessing**
Pre-processing images is all about **standardizing input images** so that you can move further along the pipeline and analyze images in the same way. Common pre-processing steps include:
 - Changing how an image looks spatially, by using geometric transforms which can scale an image, rotate it, or even change how far away an object appears
 - Changing color schemes, like choosing to use grayscale images over color images.
> Color Masking
 - Color can be used in image analysis and transformation. We'll be selecting an area of interest using a color threshold; a common use is with a green screen. A green screen is used to layer two images based on identifying and replacing a large green area. The first step is to isolate the green background, and then replace that green area with an image of your choosing.
```
# Define our color selection boundaries in RGB values
lower_green = np.array([0,180,0]) 
upper_green = np.array([100,255,100])

# Define the masked area
mask = cv2.inRange(image, lower_green, upper_green)

# Mask the image to let the car show through
masked_image = np.copy(image)
masked_image[mask != 0] = [0, 0, 0]
```
> Color Spaces
 - What if in the green screen color, the color field is not consistent (varying light, gradient, shadow, etc) ? RGB color selection will fail. This is where **Color spaces**(RGB, HSV, HLS) comes in. Here any color can be represented by a 3D coordinates. color spaces provide a way to categorize colors and represent them in digital images. 
   - **saturation** is a measurement of colorfulness. So, as colors get lighter and closer to white, they have a lower saturation value, whereas colors that are the most intense, like a bright primary color (imagine a bright red, blue, or yellow), have a high saturation value. **Hue** is the value that represents color independent of any change in brightness. So if you imagine a basic red color, then add some white to it or some black to make that color lighter or darker. In RGB, for example, **white** has the coordinate (255, 255, 255), which has the maximum value for red, green, and blue.
<img src="https://user-images.githubusercontent.com/31917400/43980371-d784a262-9ce5-11e8-83f1-8e44aca2203d.jpg" />

 - **HSV** isolates 'v'(value) component of each pixel. 'v' varies the most under different lighting conditions. 'H'(hue) channel stays consistent under shadow or brightness. If we discard 'v' and rely on 'H', we can detect colored object such as the green screen color.   
OpenCV provides a function `hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)` that converts images from one color space to another.

### **feature extraction**
A feature can easily be thought of as a "summarizer" of something. Furthermore, just as how images are really just a collection of numbers in an array, features are also just another collection of numbers in an array, although usually, they are much smaller than images. For example, if we wanted to place boxers into their weight class, we may want to do feature extraction on each fighter, and we would extract a two dimensional feature: height and weight (both of which are used to determine a weight class. Those are "features" in this sense, because they nicely ignore the irrelevant(skin color, or say, hair length, etc). But how to detect them?????? One of the breakthroughs in computer vision came from being able to automatically come up with features that are good.

In the world of image, There are two main types of features:
 - Color-based and
 - Shape-based

For example, say I wanted to classify a stop sign. Stop signs are supposed to stand out in color and shape! A stop sign is an octagon (it has 8 flat sides) and it is very red. It's red color is often enough to distinguish it, but the sign can be obscured by trees or other artifacts and the shape ends up being important, too. As a different example, say I want to avoid crashing into a car (a very important avoidance case!). I'll want to classify the object as a car, or at least recognize the car's boundaries, which are determined by shape. Specifically, I'll want to identify the edges of the vehicle, so that I can track the car and avoid it. Color is not very useful in this case, but shape is critical. **Selecting the right feature** is an important computer vision task.

> **filter**
In addition to taking advantage of color information, we also have knowledge about patterns of grayscale intensity in an image. Intensity is a measure of light and dark similar to brightness, and we can use this knowledge to detect other areas or objects of interest. For example, you can often **identify the edges** of an object by looking at an **abrupt change in intensity** which happens when an image changes from a very dark to light area.
 - To detect these changes, you’ll be using and creating **specific image filters** that look at groups of pixels and detect big changes in intensity in an image. These filters produce an output that shows these edges.

> **High-Pass Filter**
High-pass filters detect big changes in intensity over a small area, and patterns of intensity can be best seen in a grayscale image. The **filters** are in the form of matrices, often called **convolution kernels**, which are just grids of numbers that modify an image. Below is an example of a high-pass kernel that does edge detection:
<img src="https://user-images.githubusercontent.com/31917400/44003228-2e45f490-9e47-11e8-9cf6-844cc67f97b5.jpg" />

It’s important that, for edge detection, `all of the elements **sum to 0** `because edge filters compute the **difference between neighboring pixels and around a center pixel**; they are an approximation for the derivative of an image over space. During **kernel convolution**, the 3x3 kernel is slid over every pixel in the original, grayscale image. The weights in the kernel are multiplied pair-wise around a center pixel, and then added up. **This sum becomes the value of a pixel in a new, filtered, output image.** This operation is at the center of convolutional neural networks, which use multiple kernels to extract shape-based features and identify patterns that can accurately classify sets of images. These neural networks are trained on large sets of labelled data, and they learn the most effective kernel weights; the weights that help characterize each image correctly.

> High and low frequency
frequency in images is a rate of change. Well, images change in space, and a **high frequency image** is one where the intensity changes a lot, and the level of brightness changes quickly from one pixel to the next. A **low frequency image** may be one that is relatively uniform in brightness or changes very slowly. High-frequency components also correspond to the edges of objects in images, which can help us classify those objects.

### CNN
> Convolutional layer
 - A convolutional layer takes in an image array as input.
 - A convolutional layer can be thought of as a set of image **filters**.
   - Each filter extracts a specific kind of feature (like an edge, blur, etc).
 - The output of a given convolutional layer is a set of **feature maps**, which are differently filtered versions of the input image.

> Fully-connected layer
 - Its job is to connect the input to a desired form of output. 
   - As an example, say we are sorting images into two classes: day and night, you could give a fully-connected layer a set of feature maps and tell it to use a combination of these features (multiplying them, adding them, combining them, etc.) to output a prediction: whether a given image is likely taken during the "day" or "night." This output prediction is sometimes called the output layer.

Typically, CNN's are made of many convolutional layers and even include other processing layers whose job is to standardize data or reduce its dimensionality (for a faster network).
 
 








































