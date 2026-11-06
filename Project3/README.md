# DS551/CS551 Inidividual Project 3
# Deep Q-learning Network(DQN)
Please don't revise test.py, environment.py, atari_wrapper.py, main.py, and agent.py

You work on the following files, including agent_dqn.py and dqn_model.py. 

You can optionally update argument.py to add your own arguments (if needed).

#### Starating Date
* Week 7, Tuesday Oct 6, 2026 (23:59)

#### Due Date
* Deadline: Week 10, Tuesday Oct 27, 2026 (23:59)

#### Total Points
* 100 (One Hundred)

## Leaderboard and Bonus Points
In this project, we will provide a leaderboard and give **10** bonus points to the **top 3** highest reward students! 
* Where to see the leaderboard 
  * The leaderboard is **live in Gradescope**, ranked by your average reward, and updates automatically every time you submit -- no need to post a screenshot anywhere. TA will read the top 3 off the Gradescope leaderboard after the deadline and apply the +10 bonus. <br>
  * The leaderboards of previous years are also posted at the end of this page, you can check it out.

  
* How to evaluate
  * You should submit your latest trained model (.pth file) and python code (agent_dqn.py, dqn_model.py, and optionally argument.py (if updated)). When you submit to Canvas/Gradescope, the autograder automatically runs your code (`python main.py --test_dqn`) and computes your Trained Model and Python Code scores right away -- make sure the result is consistent with the screenshot in your report. 
  
* How to grade
  * Top 3 students on the leaderboard can get 10 bonus points for project 3.
  
## Setup
* Recommended programming IDE (integrated development environment): VS code (See [install VS code](https://code.visualstudio.com/)) 
* Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* Create virtual environment and install Python 3.11: `conda create -n myenv python=3.11`. This will help you create a new conda environment named myenv.
* Activate your virtual environment: `conda activate myenv`
* install pytorch: See [install pytorch](https://pytorch.org/get-started/locally/) and pick the build matching your CUDA version (or CPU-only if you don't have a local GPU -- note training in practice requires a GPU; a CPU-only machine will be far too slow).
* For the Atari wrapper, install ray[rllib]: `pip install -U "ray[rllib]" ipywidgets`
* Install gymnasium: `pip install opencv-python-headless "gymnasium[atari]==1.2.2"` (See [install gymnasium](https://github.com/Farama-Foundation/Gymnasium))
  * **Important:** this is pinned to `1.2.2`, *not* the newest `1.3.0`. `ray[rllib]` (installed above, used by `atari_wrapper.py`) hard-pins `gymnasium==1.2.2` and fails to resolve against `1.3.0` -- this is a verified, real dependency conflict, not a typo. Don't try to "upgrade" this yourself.
  * `gymnasium[atari]` pulls in `ale-py`, which registers the `ALE/*` and legacy `*NoFrameskip-v4` environment IDs automatically on import -- `atari_wrapper.py` already does `import ale_py` for you.
* Download the Atari ROMs: `python -m pip install -U "autorom[accept-rom-license]"`, then `AutoROM --accept-license`.
* For successfully running code, you may also need to install the following item: `pip install --upgrade scipy numpy`.
* For video recording in testing, install the following: `pip install moviepy`, and make sure `ffmpeg` is available (e.g. `conda install -c conda-forge ffmpeg`).
* When testing, for nice output on the terminal, you need to install tqdm: `pip install tqdm`

## How to run :
training DQN:
* `$ python main.py --train_dqn`

testing DQN:
* `$ python main.py --test_dqn`

testing DQN while recording a video (recording video takes time, so usually you use this option when the number of testing episodes is small):
* `$ python main.py --test_dqn --record_video`

## Goal
In this project, you will be asked to implement DQN to play [Breakout](https://gymnasium.farama.org/environments/atari/breakout/). This project will be completed in Python 3 using [Pytorch](https://pytorch.org/). The goal of your training is to get averaging reward in 100 episodes over **40 points** in **Breakout** (each episode has 5 lives), with OpenAI's Atari wrapper & clipped reward. For more details, please see the [slides](https://github.com/UrbanIntelligence/WPI-DS551-Fall26/blob/main/Project3/materials/DS551_CS525%20FALL_Project%203%20-%20Deep%20Q-learning.pdf).

<img src="../Project3/materials/project3.png" width="80%" >

## Deliverables

Please submit the following files to Canvas:

* **Trained Model** (Will be autograded by Canvas GradeScope)
  * Model file (.pth). The autograder doesn't require any specific
    filename, but it must match whatever filename your own `agent_dqn.py`
    is coded to load (see "Submission checklist" below).


* **PDF Report** (To be graded by TA)
  * screenshot of the test results
  * Set of Experiments Performed: 
    * Include a section describing the set of experiments that you performed
    * what structures you experimented with (i.e., number of layers, number of neurons in each layer)
    * what hyperparameters you varied (e.g., number of epochs of training, batch size and any other parameter values, weight initialization schema, activation function)
    * what kind of loss function you used and what kind of optimizer you used.
  * Special skills: Include the skills which can improve the generation quality. Here are some [tips](https://arxiv.org/pdf/1710.02298.pdf) may help. (Optional)
  * Visualization: Learning curve of DQN. 
    * X-axis: number of episodes
    * Y-axis: average reward in last 30 episodes.
    
    <img src="../Project3/materials/plot.png" width="60%" >

* **Python Code** (Will be autograded by Canvas GradeScope)
  * All the code you implemented, including agent_dqn.py, dqn_model.py, and optionally argument.py (if updated)

### Submission checklist

**⚠️ Submitting only your `.pth` file is the #1 mistake and scores an
immediate 0/70 on the autograded portion.** Your submission (directly, or
zipped -- both work) must include **all** of:
- [ ] `agent_dqn.py`
- [ ] `dqn_model.py`
- [ ] `argument.py` -- only if you added custom arguments
- [ ] your trained model `.pth` file(s)
- [ ] your PDF report

The autograder reads only a subset of your submission. It does not read report.pdf, which is graded separately by the TA; therefore, you will not see feedback on the report in Gradescope.

For the files processed by the autograder, it looks for the required files by filename, regardless of where they are located within your submission. You may submit either a list of individual files or a compressed ZIP file containing all the required files.

* `agent_dqn.py` and `dqn_model.py` -- **required**; if either is missing,
  Gradescope reports "MISSING required file" and both the Trained Model
  and Python Code scores are 0 -- this is exactly what happens if you
  submit just the checkpoint by itself
* `argument.py` -- optional, only used if you added custom arguments
* your trained model `.pth` file(s) -- **required**; missing this also
  scores 0 on the automated portion. The filename doesn't need to be
  anything specific, but it **must match whatever filename your own
  `agent_dqn.py` is coded to load** (e.g. if your `__init__` does
  `torch.load('save_model/best_checkpoint.pth')`, submit a file named
  `best_checkpoint.pth`, not something else)

## Grading
Submitting to Canvas/Gradescope automatically runs `python main.py --test_dqn`
against your code and checkpoint (100 episodes, 5 lives each) and scores the
**Trained Model** and **Python Code** components immediately -- the same
command you'd run yourself, so what you see locally is what you get. The
**PDF Report** and the leaderboard **Bonus** are graded separately by the TA
after the deadline (not by the autograder).

* **Trained Model (50 points, automatic)**
  * Getting averaging reward in 100 episodes over **40 points** (with 5 lives) in Breakout will get full credits. 
  * For every average reward below 40, you will be taken off 2 points. i.e., you will be taken off 2 points, if getting averaging reward in 100 episodes is 39 and taken off 4 points, if averaging reward is 38, so on so forth.

* **PDF Report (30 points, manual/TA-graded)**
  * Set of parameters performed: 20 points
  * Visualization: 10 points
  
* **Python Code (20 points, automatic)**
  * You can get full credits if the scripts can run successfully, otherwise you may loss some points based on your error.

## Hints
* [Naive Pytorch Tutorial](https://github.com/UrbanIntelligence/WPI-DS551-Fall26/blob/main/Project3/Pytorch_tutorial.ipynb)
* [How to Save Model with Pytorch](https://github.com/yingxue-zhang/DS595CS525-RL-Projects/blob/master/Project3/materials/How%20to%20Save%20Model%20with%20Pytorch.pdf)
* [Official Pytorch Tutorial](https://pytorch.org/tutorials/)
* [Official DQN Pytorch Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* [Official DQN paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf)
* [DQN Tutorial on Medium](https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4)

## Tips for Using Turing GPUs or Google Cloud
* [How to use GPUs on WPI Turing](https://github.com/UrbanIntelligence/WPI-DS551-Fall26/blob/main/Project3/materials/Turing_Setup_Instructions_2025.pdf)
* [Google Cloud Platform](https://colab.google/)

  
## Leaderboard for Fall 2019** 
  
  | Top | Date | Name | Score |
  | :---: | :---:| :---: | :---: |
  | **1**   |10/22/2019| **Prathyush SP**          |  **142.77**    | 
  |         |10/18/2019| Prathyush SP          |  81.07     | 
  | **2**   |10/28/2019| **Sapan Agrawal**         |   **91.34**    |
  | 3   |11/1/2019| Hanshen Yu| 86.82 |
  | **4**   |10/31/2019| **Mohamed Mahdi Alouane** | **80.24**     | 
  | 5   |10/26/2019| Vamshi Krishna Uppununthala|  79.5   | 
  | 6   |10/31/2019| Sai Vineeth K V | 66.5 | 
  | 7   |11/14/2019| Cory neville | 59.96 | 
  | 8   |10/24/2019|Shreesha Narasimha Murthy  |56.79     | 
  | 9   |10/20/2019|Sinan Morcel            |53.26        |
  
## Leaderboard for Fall 2020** 

  | Top | Date | Name | Score |
  | :---: | :---:| :---: | :---: | 
  | 1  | 11/19/2020|Abhishek Jain  | 424.21  |
  | 2  | 11/19/2020|Akshay Sadanandan  | 403  |
  | 3  | 11/19/2020|Dhirajsinh Deshmukh  | 393.37  |
  |4 |11/19/2020 |   Daniel Jeswin Nallathambi      | 335.26  |
  | 5  | 11/18/2020|Sayali Shelke  | 334  |
  |6 | 11/19/2020|Varun Eranki  | 298  |
  | 7  | 11/5/2020|Apiwat Ditthapron  | 194.5  | 
  |8 | 11/18/2020|Panagiotis Argyrakis  | 156.09  |
  |9 | 11/20/2020|Scott Tang  | 153.89  |
  |10 | 11/18/2020|Xinyuan Yang  | 139.11  |
  
## Leaderboard for Spring 2022**
  
  | Top | Date | Name | Score |
  | :---: | :---:| :---: | :---: | 
  | 1  |4/6/2022 | Hongchao Zhang | 128 |
  | 2  |4/13/2022 | Apratim Mukherjee| 112 |
  | 3  | 4/6/2022 |  Puru Upadhyay | 82 |
  | 4  | 4/6/2022 |  Khai Yi Chin | 81 |
  | 4  | 4/6/2022 |  Karter Krueger | 81 |
  | 6  | 4/6/2022 |  Sailesh Rajagopalan | 78 |
  | 6  |4/6/2022 | Steven Hyland | 78 |
  | 8  |4/6/2022 | Yiran Fang | 74 |
  | 9  |4/6/2022 | Zhentian Qian | 67 |
  | 10  |4/6/2022 | Anujay Sharma | 66 |

## Leaderboard for Fall 2022**
  | Top | Date | Name | Score | Model |
  | :---: | :---:| :---: | :---: | :---: |
  | 1  | 10/24/2022 | Palawat Busaranuvong | 317 | Prioritized DQN |
  | 2  | 11/15/2022 | Amey Deshpande | 166.8 | ... |
  | 2  | 11/15/2022 | Rane, Bhushan | 166.8 | ... |
  | 3  | 11/15/2022 | Yash Patil | 113.14 | ... |
  | 4  | 11/06/2022 | Yiwei Jiang | 96.45 | DDQN |
  | 5  | 11/15/2022 | Aniket Patil | 92.18 | ... |
  | 6  | 11/14/2022 | Samarth Shah | 85.39 | DDQN with Prioritized Replay |
  | 7  | 11/15/2022 | Neet Mehulkumar Mehta | 80.39 | ... |
  | 8  | 11/15/2022 | Noopur Koshta | 79.68 | ... |  
  | 9  | 11/15/2022 | Kunal Nandanwar | 79.68 | ... |            
  | 10  | 11/14/2022 | Aadesh Varude | 71.65 | Vanilla DQN |
  | 11  | 11/1t/2022 | Rutwik Bonde | 69.52 | ... |  
  | 12  | 11/07/2022 | Brown, Galen | 69.01 | Basic DQP with reward shaping |
  | 13  | 11/5/2022  | Ryan Killea | 67.12 | ... |
  | 14  | 11/14/2022  | Rushabh Kheni | 65.51 | Vanilla DQN with Deepmind architecture |  
  | 15  | 10/30/2022 | Jack Ayvazian | 47.49 | Double DQN, DeepMind Architecture | 

## Leaderboard for Fall 2023**
  | Top | Date | Name | Score | Model |
  | :---: | :---:| :---: | :---: | :---: |
  | 1  | 11/7/2023 | Antony Garcia | 426 | DQN |
  | 2  | 11/15/2023 | Anas AlRifai | 402.72 | DDQN |
  | 3  | 11/14/2023 | Maanav Iyengar | 384.32 | ... |
  | 4  | 11/1/2023 | Daniel Moyer | 377.5 | ... |
  | 5  | 11/19/2023 | Xinyi Fang | 367.27 |  Dueling Double Deep Q-Network |
  | 6  | 11/15/2023 | Martha Cash | 363.57 | DQN |
  | 7  | 11/18/2023 | Zhuang Luo | 329.4 | DQN |
  | 8  | 11/14/2023 | Yiming Liu | 315.81 | DQN |
  | 9  | 11/14/2023 | Aikeremu Aixilafu | 193.8 | ... |
  | 10 | 11/15/2023 | Michael O'Connor | 158.68 | ... |

## Leaderboard for Fall 2024** 
  | Top | Date | Name | Score | Model |
  | :---: | :---:| :---: | :---: | :---: |
  | 1  | 11/03/2024 | Zhiyang Zhang | 493.09 | dueling DQN |
  | 2  | 11/02/2024 | Badrivishal Ajeet Paurana | 382.96 | Double DQN with Prioritized Experience Replay |
  | 3  | 11/12/2024 | Maya Flores | 371.58 | Standard DQN |
  | 4  | 10/26/2024 | Ningcong Chen | 334.48 | ... |
  | 5  | 11/11/2024 | Alessandra Savio Serpes | 320.57 | Standard DQN |
  | 6  | 11/03/2024 | Sumukh Porwa | 309.54 | Double DQN |
  | 7  | 11/12/2024 | Qiaochu Liu | 309.16 | Dueling DQN |
  | 8  | 11/12/2024 | Jake Watson | 300.22 | Vanilla DQN |
  | 9  | 11/08/2024 | Srikanth Natarajan | 281.62 | ... |
  | 10  | 11/12/2024 | Piyush Thapar | 276.57 | Standard DQN |

 ## Leaderboard for Fall 2025** 
  | Top | Date | Name | Score | Model |
  | :---: | :---:| :---: | :---: | :---: |
  | ✅ 1  | 11/03/2025 | Alex Ballentine | 399.53 | Noisy Dueling DQN |
  | ✅ 2   | 11/18/2025 | Tanveer Kaur | 359.01 | Double DQN |
  | ✅ 3   | 11/18/2025 | Paola Frunzio | 358.14 | Double DQN |
  | 4  | 11/18/2025 | Kang Zhang | 327.75 | Vanilla DQN |
  | 5  | 11/18/2025 | Gavin Hamburg | 265.86 | Double DQN |
  | 6  | 11/18/2025 | Suryansh Goyal | 255.2 | Double DQN |
  | 7   | 11/18/2025 | Bibhus Luitel | 227.26 | Vanilla DQN |
  | 8   | 11/18/2025 | Manideep Duggi | 186.38 | Vanilla DQN |
  | 9  | 11/18/2025 | Kevin O'Brien | 107.21 | Dueling Double DQN |
  | 10   | 11/03/2025 | Davud Azizov | 103.86 | Vanilla DQN |
