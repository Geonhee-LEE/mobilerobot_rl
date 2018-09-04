# Machine Learning
Machine learning is a data analysis technique that teaches computers to recognize what is natural for people and animals - learning through experience. There are three types of machine learning: supervised learning, unsupervised learning, reinforcement learning.

This application is reinforcement learning with DQN (Deep Q-Learning). The reinforcement learning is concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward.


This shows reinforcement learning with TurtleBot3 in gazebo. This reinforcement learning is applied DQN(Deep Q-Learning) algorithm with LDS.
We are preparing a four-step reinforcement learning tutorial.


## Installation

To do this tutorial, you need to install Tensorflow, Keras and Anaconda with Ubuntu 16.04 and ROS kinetic.
http://wiki.ros.org/kinetic/Installation/Ubuntu


### Anaconda

You can download Anaconda 5.2[https://www.anaconda.com/download/#linux] for Python 2.7 version.

After downloading Andaconda, go to the directory in located download file and enter the follow command.

<pre> bash Anaconda2-x.x.x-Linux-x86_64.sh </pre>

After installing Anaconda,

<pre> source ~/.bashrc </pre> 
<pre> python -V</pre> 

If Anaconda is installed, you can see Python 2.7.xx :: Anaconda, Inc..

### ROS dependency packages

To use ROS and Anaconda together, you must additionally install ROS dependency packages.

<pre>$ pip install -U rosinstall msgpack empy defusedxml netifaces </pre>

#### Tensorflow (https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package)

You can install TensorFlow.

<pre>$ conda create -n tensorflow pip python=2.7</pre>
This tutorial is used python 2.7(CPU only). If you want to use another python version and GPU, please refer to TensorFlow.

Activate virtual environment 
<pre> source activate tensorflow </pre>

conda(cpu) <pre>$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp27-none-linux_x86_64.whl</pre>

conda(gpu), only cuda9.0 supported <pre>$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.1-cp27-none-linux_x86_64.whl</pre>


#### Keras

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow.

conda <pre> $ pip install keras </pre>

#### Install requirements and clone turtlebot packages

<pre> sudo apt-get install ros-kinetic-joy ros-kinetic-teleop-twist-joy ros-kinetic-teleop-twist-keyboard ros-kinetic-laser-proc ros-kinetic-rgbd-launch ros-kinetic-depthimage-to-laserscan ros-kinetic-rosserial-arduino ros-kinetic-rosserial-python ros-kinetic-rosserial-server ros-kinetic-rosserial-client ros-kinetic-rosserial-msgs ros-kinetic-amcl ros-kinetic-map-server ros-kinetic-move-base ros-kinetic-urdf ros-kinetic-xacro ros-kinetic-compressed-image-transport ros-kinetic-rqt-image-view ros-kinetic-gmapping ros-kinetic-navigation ros-kinetic-interactive-markers </pre> 
 
<pre> cd ~/catkin_ws/src/ </pre> 
<pre> git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git </pre> 
<pre> git clone https://github.com/ROBOTIS-GIT/turtlebot3.git </pre>
<pre> git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git</pre> 
<pre> cd ~/catkin_ws && catkin_make </pre> 
<pre> export TURTLEBOT3_MODEL=burger </pre> 


#### Machine Learning packages

WARNING: Please install turtlebot3, turtlebot3_msgs and turtlebot3_simulations package before installing this package.

<pre> cd ~/catkin_ws/src/ </pre> 
<pre> git clone https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning.git </pre> 
<pre> cd ~/catkin_ws && catkin_make</pre> 

Set parameters
The goal of DQN Agent is to get the TurtleBot3 to the goal avoiding obstacles. When TurtleBot3 gets closer to the goal, it gets a positive reward, and when it gets farther it gets a negative reward. The episode ends when the TurtleBot3 crashes on an obstacle or after a certain period of time. During the episode, TurtleBot3 gets a big positive reward when it gets to the goal, and TurtleBot3 gets a big negative reward when it crashes on an obstacle.


Set state
State is an observation of environment and describes the current situation. Here, state_size is 26 and has 24 LDS values, distance to goal, and angle to goal.

Turtlebot3’s LDS default is set to 360. You can modify sample of LDS at turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro.

```
<xacro:arg name="laser_visual" default="false"/>   # Visualization of LDS. If you want to see LDS, set to `true`
<scan>
  <horizontal>
    <samples>360</samples>            # The number of sample. Modify it to 24
    <resolution>1</resolution>
    <min_angle>0.0</min_angle>
    <max_angle>6.28319</max_angle>
  </horizontal>
</scan>
```
  
sample = 360	sample = 24
Set action
Action is what an agent can do in each state. Here, turtlebot3 has always 0.15 m/s of linear velocity. angular velocity is determined by action.

Action	Angular velocity(rad/s)
```
--------------
| 0 : -1.5  |
| 1 : -0.75 |
| 2 : 0    | 
| 3 : 0.75  |
| 4 : 1.5   |
---------------
```
Set reward
When turtlebot3 takes an action in a state, it receives a reward. The reward design is very important for learning. A reward can be positive or negative. When turtlebot3 gets to the goal, it gets big positive reward. When turtlebot3 collides with an obstacle, it gets big negative reward. If you want to apply your reward design, modify setReward function at /turtlebot3_machine_learning/turtlebot3_dqn/src/turtlebot3_dqn/environment_stage_#.py.

Set hyper parameters
This tutorial has been learned using DQN. DQN is a reinforcement learning method that selects a deep neural network by approximating the action-value function(Q-value). Agent has follow hyper parameters at /turtlebot3_machine_learning/turtlebot3_dqn/nodes/turtlebot3_dqn_stage_#.

Hyper parameter	default	description
episode_step	6000	The time step of one episode.
target_update	2000	Update rate of target network.
discount_factor	0.99	Represents how much future events lose their value according to how far away.
learning_rate	0.00025	Learning speed. If the value is too large, learning does not work well, and if it is too small, learning time is long.
epsilon	1.0	The probability of choosing a random action.
epsilon_decay	0.99	Reduction rate of epsilon. When one episode ends, the epsilon reduce.
epsilon_min	0.05	The minimum of epsilon.
batch_size	64	Size of a group of training samples.
train_start	64	Start training if the replay memory size is greater than 64.
memory	1000000	The size of replay memory.

#### Run Machine Learning

<pre>  roslaunch turtlebot3_gazebo turtlebot3_stage_1.launch</pre> 
conda <pre>  roslaunch turtlebot3_dqn result_graph.launch </pre> 
conda <pre> roslaunch turtlebot3_dqn turtlebot3_dqn_stage_1.launch</pre> 


### Building PyQT5 with Python 2.7 on Ubuntu 16.04 (http://www.powerbrowsing.com/2017/12/building-pyqt5-with-python-2-7-on-ubuntu-16-04/)

So you want Qt5, but you have a lot of strict dependencies holding you back from going to Python 3. What to do? Go build everything yourself!

The following are steps based on a fresh install of Ubuntu 16.04, so adapt it per your needs.

First install the latest version of Qt. I was using the free open source edition. At the time of this post, the following was the set of commands given by Qt’s official website. I just downloaded the installation into my Downloads folder and ran it:

First download the installer:



<pre>  cd ~/Downloads </pre> 

<pre>  wget https://download.qt.io/official_releases/qt/5.9/5.9.2/qt-opensource-linux-x64-5.9.2.run </pre> 



Next, adjust the permissions and install Qt:


<pre>  chmod +x qt-opensource-linux-x64-5.9.2.run </pre> 
<pre> ./qt-opensource-linux-x64-5.9.2.run </pre> 

Now, that should have installed Qt5.x on your Ubuntu installation, it is time to install PyQt5 such that it uses Qt5 with Python 2.7. The key in this setup procedure is to build PyQt5 yourself.

First, SIP must be installed before proceeding with building PyQt5. You can download SIP from here. You can also just wget it!



<pre>  https://sourceforge.net/projects/pyqt/files/sip/sip-4.19.6/sip-4.19.6.tar.gz </pre> 


Now is where things get important, as I always prefer to use virtual environments when working with Python, rather than installing everything on the global Python installation. After extracting the file, make sure to run configure.py using your virtual environment’s Python!



<pre> tar xvf sip-4.19.6.tar.gz </pre> 
<pre> cd sip-4.19.6/ </pre> 
<pre> ~/path/to/virtualenv/python configure.py </pre> (build configure.py using python command in virtualenv )

If the configuration was successful, the output should tell you that things related to SIP will be installed in your virtual environment’s folder, and NOT in your global Python that resides in /usr.

Now, running make and sudo make install will install SIP into your virtual environment folder, even if you are using sudo.

<pre> make </pre> 
<pre> sudo make install </pre> 


Now it is time to download PyQt5 and build it!



<pre> wget https://sourceforge.net/projects/pyqt/files/PyQt5/PyQt-5.9.2/PyQt5_gpl-5.9.2.tar.gz </pre> 


Configuration of PyQt5 is needed for building it. Again, using the right interpreter and the right parameters are key.


<pre> ~/my_virtual_env_path/to/python configure.py -d ~/my_virtual_env_path/lib/python2.7/site-packages/ --sip=/my_virtual_env_path/bin/sip --sip-incdir=../sip-4.19.3/siplib/ --qmake ~/Qt5.9.0/5.9.0/gcc_64/bin/qmake</pre> 
(build configure.py using python command in virtualenv )
<pre> make</pre> 
<pre> make install</pre> 


Now, assuming there are no errors, you should be able to run ‘import PyQt5’ in your virtual environment running Python 2.7!


# reference 

[1]. http://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/#machine-learning

[2]. https://github.com/erlerobot/gym-gazebo

[3]. http://wiki.ros.org/openai_ros

[4]. http://scriptedonachip.com/pytorch-ros

[5]. https://github.com/lakehanne/soft-neuro-adapt


