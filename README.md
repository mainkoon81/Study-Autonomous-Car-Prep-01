# Study-00-Theory

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


























































