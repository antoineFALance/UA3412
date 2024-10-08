import random
import tensorflow as tf
from heater_env_4 import heaterEnvRC
from collections import deque
import numpy as np
import math
import itertools
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
tf.random.set_seed(1234)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# PATH
PATH_TO_OUTPUT_MODELS = "C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\RL\\DQN_4\\models\\"

# HYPERPARAMETERS

EPOCHS=2000
hidden_units = [64,64]
eps_start = 1
eps_end = 0.000
eps_decay = 0.01
batch_size=32
gamma = 0.8
lr = 0.001
target_update = 25
losses=[]
ep_rewards = 0
total_rewards = []
memoryCapacity=2000

class EpsilonGreedyStrategy():
	"""
	Decaying Epsilon-greedy strategy
	"""
	def __init__(self, start, end, decay):
		self.start = start
		self.end = end
		self.decay = decay

	def get_exploration_rate(self, current_step):
		return self.end + (self.start - self.end) * math.exp(-1*current_step*self.decay)


class Model(tf.keras.Model):

    def __init__(self, num_states, hidden_units, num_actions):
        super(Model, self).__init__()

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(None,1,num_states))
        self.hidden_layers = []

        for hidden_unit in hidden_units:
            self.hidden_layers.append( tf.keras.layers.Dense(hidden_unit, activation='relu'))

        self.output_layer =  tf.keras.layers.Dense(num_actions, activation='linear')

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output

class MemoryBuffer:
    def __init__(self, capacity):
        self.memory=deque(maxlen=capacity)

    def push(self,experience):
        if not np.isnan(experience[0]).any() and not np.isnan(experience[0]).any():
            self.memory.append(experience)

    def sample(self, batch_size=32):
        memorySize=self.getSize()
        indices = random.sample(range(memorySize), batch_size)
        return [self.memory[index] for index in indices]

    def getSize(self):
        return len(self.memory)


class DQN_Agent():

    def __init__(self, strategy, num_actions):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions

    def select_action(self, state, policy_net,rate):
        # rate = max(self.strategy.get_exploration_rate(self.current_step),0.001)
        self.current_step += 1
        if rate > random.random():
            # return random.randrange(self.num_actions), rate, True
            return random.randrange(self.num_actions), True
        else:
            try:
                test =state
                # return np.argmax(policy_net(np.atleast_2d(np.atleast_2d(state).astype('float32')))), rate, False
                return np.argmax(policy_net(np.atleast_2d(np.atleast_2d(state).astype('float32')))), False
                # return np.argmin(policy_net(np.atleast_2d(np.atleast_2d(state).astype('float32')))), False
            except:
                pass


def copy_weights(Copy_from, Copy_to):
	"""
	Function to copy weights of a model to other
	"""
	variables2 = Copy_from.trainable_variables
	variables1 = Copy_to.trainable_variables
	for v1, v2 in zip(variables1, variables2):
		v1.assign(v2.numpy())


def save(model,fileName,directory):
    fullFileName=directory+fileName
    model.save(fullFileName,save_format="tf")

if __name__ == "__main__":
    env = heaterEnvRC()
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = DQN_Agent(strategy,env.action_space.n)
    memory=MemoryBuffer(capacity=memoryCapacity)
    ep_rewards = 0

    # Initialize the policy and target network
    policy_net = Model(env.observation_space.shape[0], hidden_units, env.action_space.n)
    target_net = Model(env.observation_space.shape[0], hidden_units, env.action_space.n)
    optimizer = tf.optimizers.Adam(lr)
    epoch=0
    timeStepList=[]
    avg_reward_timestep=[]
    ratio_reward_timeStep=[]
    rate=1
    lastReward=[-2500]
    while True:
        state=env.reset()
        done = False
        ep_rewards=0

        for timestep in itertools.count():
            action, flag = agent.select_action(state, policy_net,rate=rate)
            next_state, reward, done, _,__ = env.step(action)

            if timestep % 50 == 0:
                env.on_running()

            ep_rewards += reward

            if np.isnan(reward):
                print('ok')

            # Store the experience in Replay memory
            memory.push([state, action, next_state, reward, done])
            state = next_state
            test =np.isnan(state)


            if memory.getSize()>batch_size:
                past_experiences = memory.sample(batch_size)
                batch =list(zip(*past_experiences))

                states, actions, rewards, next_states, dones = np.asarray(batch[0]),\
                                                               np.asarray(batch[1]),\
                                                               np.asarray( batch[3]), \
                                                               np.asarray(batch[2]), \
                                                               np.asarray(batch[4])

                q_s_a_prime = np.max(target_net(np.atleast_2d(next_states).reshape(batch_size,env.observation_space.shape[0]).astype('float32')), axis=1)
                q_s_a_target = np.where(dones, rewards, rewards + gamma * q_s_a_prime)
                q_s_a_target = tf.convert_to_tensor(q_s_a_target, dtype='float32')

                with tf.GradientTape() as tape:
                    test1=np.atleast_2d(states).astype('float32')
                    q_s_a = tf.math.reduce_sum(
                    policy_net(np.atleast_2d(states).reshape((batch_size,env.observation_space.shape[0])).astype('float32')) * tf.one_hot(actions, env.action_space.n), axis=1)
                    loss = tf.math.reduce_mean(tf.square(q_s_a_target - q_s_a))

                    # Update the policy network weights using ADAM
                variables = policy_net.trainable_variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
                losses.append(loss.numpy())
            else:
                losses.append(0)

            if done :
                env.reset()
                break

        if ep_rewards>max(lastReward) and timestep>1400:
            copy_weights(policy_net, target_net)
            save(target_net, "target_net", PATH_TO_OUTPUT_MODELS)
            save(policy_net, "policy_net", PATH_TO_OUTPUT_MODELS)

        lastReward.append(ep_rewards)
        ratio_reward_timeStep.append(ep_rewards / (timestep+1))
        timeStepList.append(timestep)
        AvgTimeStep=np.mean(timeStepList)
        total_rewards.append(ep_rewards)
        copy_weights(policy_net, target_net)
        print('EPOCH: ' + str(epoch) +'-rate: '+str(rate)+'- Reward : '+str(total_rewards[-1])+'-timestep: '+str(timestep)+'-ratio Time step rward: '+str(ep_rewards/(timestep+1))+'Avg ratio: '+str(np.mean(ratio_reward_timeStep[-5:])))

        rate=max(eps_start * math.exp(-1*epoch*eps_decay),0.01)
        epoch+=1

        if epoch > 30000:
            break

CumulativeAvgReward = [np.mean(total_rewards[0:index]) for index in range(1,len(total_rewards))]
plt.plot(list(CumulativeAvgReward))
plt.show()
print('ok')
