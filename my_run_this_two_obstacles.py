from myenv_two_obstacles import Env
from hybrid import HybPrioritizedReplayDueling
import tensorflow as tf
import time

MEMORY_SIZE = 5000
J = 200
ts = 0.05

def run_maze(RL):
    step = 0
    for episode in range(20000):
        start = time.process_time()
        print('---------------epo:------------------  ', episode)
        # initial observation
        observation = env.reset()
        # print(observation)

        for j in range(1, J + 1):
            # fresh env
            # env.render()
            t = j * ts
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # print('j: ', j, ' t: ', t, '  step:  ', step, '  action: ', action)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(observation, action, t)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 300) and (step % 5 == 0):
                learn_st1 = time.process_time()
                RL.learn()
                learn_et1 = time.process_time()
                # print('learn_time: ', learn_et1 - learn_st1)
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                print('结束一轮eposide, j: ', j, ' t: ', t)
                break
            step += 1
        end = time.process_time()
        print('一轮耗时:', end - start)

    # end of game
    print('game over')
    # env.destroy()


if __name__ == "__main__":
    # maze game
    env = Env()
    sess = tf.Session()
    with tf.variable_scope('dueling'):
        dueling_DQN = HybPrioritizedReplayDueling(
            n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, sess=sess, prioritized=True, dueling=True, output_graph=True)

    sess.run(tf.global_variables_initializer())
    run_maze(dueling_DQN)
    dueling_DQN.plot_cost()

