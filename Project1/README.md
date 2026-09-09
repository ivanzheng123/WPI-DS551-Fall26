# DS551/CS551 2026 Fall Individual Project 1
# Dynamic Programming of Markov Decision Process

#### Starting Date
* Week 2 Tuesday September 1, 2026(23:59)

#### Due Date
* Week 4 Tuesday September 15, 2026(23:59)

#### Total Points
* 100 (One Hundred)

## Goal

* In this assignment, you will be asked to implement policy iteration and value iteration for the Frozen Lake environment (FrozenLake-v1) from [OpenAI Gymnasium](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) and play the game with the algorithms you implemented. This project will be completed in Python 3.



<img src="../Project1/img/hw1.png" width="80%">

* See more details of FrozenLake on [OpenAI Gymnasium](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).


* If your program works, the command line output should look like this.
<img src="../Project1/img/UnitTest2023.png" width="80%">

## Deliverables

Submit your completed `mdp_dp.py` to Canvas. You can upload it directly --
**no need to zip it** -- or zip it as `firstName_lastName_hw1.zip` if you
prefer; the autograder accepts either.

## Grading
* policy evaluation (20 points)
* policy improvement (20 points)
* policy iteration (20 points)
* value iteration (20 points)
* rander game (20 points)

## Hints
* Policy Evaluation<br/>
<span style="color:red">**Please note that reward can be defined on (state), (state, action), (state, action, next_state). In this assignment, we define the reward on (state,action,next_state).** The following pseudocode is the general method.</span>
<img src="../Project1/img/pe.png" width="80%" >

* Policy Iteration<br/>
<img src="../Project1/img/pi.png" width="80%" >

* Value Iteration<br/>
<img src="../Project1/img/vi.png" width="80%" >


## Setup
* Recommended programming IDE (integrated development environment): VS code (See [install VS code](https://code.visualstudio.com/)) 
* Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* Create virtual environment and install Python 3.11: `conda create -n myenv python=3.11`. This will help you create a new conda environment named myenv.
* Activate your virtual environment: `conda activate myenv`
* Install gymnasium: `pip install "gymnasium[atari]==1.2.2"` (See [install gymnasium](https://github.com/Farama-Foundation/Gymnasium))
* Install nose: `pip install pynose==1.5.5` (See [install nose](https://pypi.org/project/pynose/))
* Install numpy: `pip install numpy`

```diff
- Note: our environment code mdp_dp.py and mdp_dp_test.py are verified for DS551/CS551 Fall 2026 with gymnasium==1.2.2.

- This is pinned deliberately -- Project 3 (which shares the same course environment) requires gymnasium==1.2.2 because ray[rllib] hard-pins that exact version. Please install with pip install "gymnasium[atari]==1.2.2", not an unpinned/newer version.
```


## Guidelines
* Implement functions in mdp_dp.py
* Evaluate functions by typing "python -m nose -v mdp_dp_test.py" in terminal (you need put mdp_dp.py and mdp_dp_test.py in the same folder)
* If you got error using "nosetests -v mdp_dp_test.py" due to python version (sometimes, nosetests will use python2.7 by default), use `python -m nose -v mdp_dp_test.py` instead (as above), or try `python3 -m nose -v mdp_dp_test.py`

## Tips for Python and OpenAI Gym
[Python Documentation](https://www.python.org/doc/)

[Python Tutorial](https://www.geeksforgeeks.org/python-programming-language/)

[OpenAI Gym Documentation 1](https://gymnasium.farama.org/)

[OpenAI Gym Documentation 2](https://github.com/Farama-Foundation/Gymnasium)
