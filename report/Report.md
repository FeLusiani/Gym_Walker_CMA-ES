# Project presentation

Here follows an in-depth presentation of the project.

## The model

A feed-forward neural network is trained to pilot a [Bipedal Walker](https://gym.openai.com/envs/BipedalWalker-v2/) (one of OpenAI Gym environments) using a [CMA Evolutionary Strategy](https://en.wikipedia.org/wiki/CMA-ES) (using [pycma](https://github.com/CMA-ES/pycma) package). The feed-forward NN is implemented through [PyTorch](https://github.com/pytorch/pytorch).

### The Bipedal Walker environment
Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There's no coordinates in the state vector.

![](./images/original_walker.gif)

### The Neural Network model
At every step of the environment, the action vector is computed from the agent observed state using a feed-forward NN with the following number of units per layer: 14, 25, 10, 4. Each layers has a Leaky ReLU activation function, expect for the output layer, which uses a [HardTanh](https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html#torch.nn.Hardtanh) to clip the values of the action vector.

### The CMA-ES method
The [Covariance Matrix Adaptation Evolution Strategy (CMA-ES)](https://en.wikipedia.org/wiki/CMA-ES) is a particular kind of evolution strategy. In an evolution strategy, we iteratively sample candidate points out of a multivariate distribution in the search space, evaluate the objective function in the points, and updated accordingly the distribution.

If the distribution is a [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution), then the update consists in changing the mean vector *u* and the covariance matrix *M*. The matrix *M* can be restricted to be an identity matrix multiplied by a scalar (keep track of just one std value for all the dimensions), or a diagonal matrix (keep track of a vector of std values, one for each dimension). In the CMA-ES method, no such restriction is applied. Therefore, the CMA-ES method is able to keep full track of the correlations between the different dimensions, affording better convergence over ill-conditioned objective functions, at the cost of updating and storing a matrix of dimension n<sup>2</sup> (with n the number of dimensions of the search-space).

In this project, the searching space corresponds to the weights of the NN we are training, and the objective function is set to the total reward obtained by the agent at the end of an episode (multiplied by -1, as to make it a minimization problem).

## User interface

### Training the model

The code offers two interface scripts to the user. The first one is the `run_training.py`.

```
~$ python ./Gym_Walker_CMA-ES/run_training.py --help

usage: run_training.py [-h] [--duration DURATION] [--n_gens N_GENS] [--std STD] [--name NAME] [--load LOAD] [--dir DIR] [--no_multiproc] [--logging]

Train model with cma-es

optional arguments:
  -h, --help           show this help message and exit
  --duration DURATION  duration of episode
  --n_gens N_GENS      n of generations
  --std STD            starting std
  --name NAME          filename to save model
  --load LOAD          file to load as starting solution
  --dir DIR            dir path to save model
  --no_multiproc       disable multiprocessing
  --logging            enable cma logging

```

Here follows an explanation of each flag:

- `--duration`: duration (n. of steps) of each training episode. Default is 100.
- `--n_gens`: n. of iterations (generations) to reach before stopping. Default is 50.
- `--std`: starting standard deviation (the covariance matrix is initialized as the identity matrix multiplied by this factor). Default is 0.3.
- `--name`: filename to use to save the model at the end of training. By default, the name is a formatted string showing the three training parameters (duration, n_gens and starting std).
- `--load`: filename of a previously saved model, to load as the starting solution (that is, the initial distribution mean). Useful to further train a saved model.
- `--dir`: directory where the trained model is to be saved at the end of training. Default is the `saved_models` directory inside of the project.
- `--no_multiproc`: disable multiprocessing. While this makes the episodes evaluation slower, it can be useful for debugging (and to correctly measure the training time).
- `--logging`: enables the logging of the `pycma` solver, which saves infos to a `outcma` directory.

If we run the script with no flags, it will train the agent using the default settings:

```
~$ python ./Gym_Walker_CMA-ES/run_training.py
/home/felusiani/Projects/gym_walker/gym/gym/logger.py:30: UserWarning: WARN: Box bound precision lowered by casting to float32
  warnings.warn(colorize('%s: %s'%('WARN', msg % args), 'yellow'))
(11_w,23)-aCMA-ES (mu_w=6.7,w_1=25%) in dimension 679 (seed=307371, Sat Apr 24 17:21:24 2021)
Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
    1     23 4.052863008997098e+00 1.0e+00 2.97e-01  3e-01  3e-01 0:01.3
    2     46 8.624444341815398e+00 1.0e+00 2.94e-01  3e-01  3e-01 0:02.5
    3     69 6.921273856472832e+00 1.0e+00 2.91e-01  3e-01  3e-01 0:03.9
    6    138 3.539996880124008e+00 1.0e+00 2.84e-01  3e-01  3e-01 0:07.9
    9    207 2.706086083090562e+00 1.0e+00 2.78e-01  3e-01  3e-01 0:12.1
   13    299 2.130009593710304e+00 1.0e+00 2.71e-01  3e-01  3e-01 0:17.9
   18    414 3.491434025784964e+00 1.0e+00 2.64e-01  3e-01  3e-01 0:24.5
   24    552 2.931648961782464e+00 1.0e+00 2.58e-01  3e-01  3e-01 0:32.9
   30    690 8.544348158209311e-01 1.0e+00 2.52e-01  3e-01  3e-01 0:41.0
   37    851 -8.306338359414836e-02 1.0e+00 2.47e-01  2e-01  2e-01 0:50.9
   45   1035 -1.888917951850775e-01 1.0e+00 2.43e-01  2e-01  2e-01 1:01.8
   50   1150 -9.646593723098380e-01 1.0e+00 2.41e-01  2e-01  2e-01 1:08.6

 Saving model at /home/felusiani/Projects/gym_walker/Gym_Walker_CMA-ES/saved_models/walker_D100_N50_STD3.00E-01.pth
```

The infos we read on the terminal are printed by the `pycma` solver. See the section **Visualizing the CMA-ES training** for an explanation of these values. At the end of the training, the model is saved in the `saved_models` directory, with a name showing the training parameters (**D** for duration, **N** for n. of generations, and **STD** for the starting sigma).

### Testing the model
To see the saved model in action, the `test.py` script can be used:
```
$ python ./Gym_Walker_CMA-ES/test.py -h
usage: test.py [-h] [--duration DURATION] model_file

Test model

positional arguments:
  model_file           file to load as model

optional arguments:
  -h, --help           show this help message and exit
  --duration DURATION  duration of episode
```

Aside from the model file to load, the script takes an optional argument for the duration of each test episode.

The script will show the agent acting in 10 episodes. For example, this is the behaviour of an agent trained for 50 generations on episodes of duration 200 step:

![](/report/images/walking_cropped.gif)

### Training a saved model

As described above, using the `run_training.py` script, it is possible to load a previously saved model to use as a starting solution (distribution mean) for the CMA-ES algorithm. You can see an [**example video here**](https://youtu.be/AwU9RbSOIP0).

## Visualizing the CMA-ES training
We start with an explanation of the values printed on terminal by the `pycma` solver.

The first column shows the number of **iterations** (generations). The second column shows the number of **function evaluations**; since the solver has chosen a population of 23 individuals, this column is equal to the previous one multiplied by 23.

The third column is the **max axis ratio**, defined as the max square root ratio between any two of the eigen-vectors of the covariance matrix *M*. This quantity can be visualized as the max ratio between two axis of the multi-dimensional ellipsoid shaped by the current distribution: it is therefore a measure of how much the distribution is "stretched" in some direction (or alternatively, of how much the current *M* differs from the identity matrix, eventually rotated in space).

The fourth column is **sigma**, the current scaling factor used for the covariance matrix, or "step-size". The next two columns are the **min and max standard deviation** of the current distribution over all the dimensions. Notice how this quantity relates closely to the max axis ratio and the current sigma (when the covariance matrix is diagonal, we have *max axis ratio = sigma * max_std / min_std*).

The last column is the current **running time** of the algorithm.

### Plotting the CMA-ES
As described above, using the `--logging` flag in the `run_training.py` script will enable the logging functionalities of the `pycma` solver, which saves the logged data in the `outcmaes` directory.

The `pycma` library presents its own functionality to plot the data logged in the `/outcmaes` directory. However, since this doesn't seem to be currently working, I have made the jupyter notebook `plot_cmaes.ipynb` to plot some of the data logged by the `pycma` module. Below are shown the results relative to the training of an agent for 500 generations.

![](/img/objective_function.svg)

This is the objective function at each generation. Specifically, the plot shows the total reward (multiplied by -1) for the worst, median and best individual at each generation.

![](/img/std.svg)

This is the sigma value at each generation. As the method converges to a solution, the sigma value decreases.

![](/img/axis_ratio.svg)

This is the max axis ratio at each generation. As the distribution is updated according to the objective function values, its shape becomes less spherical and more "stretched" in the direction of correlated dimensions.




