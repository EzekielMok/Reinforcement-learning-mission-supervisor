import numpy as np
from math import *
from sympy import *
from matplotlib import pyplot as plt
from scipy.integrate import quad


class Env:
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['B', 'A']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.sPv = []
        self.sa = []
        self.sdvo = []
        self.sdvo1 = []
        self.sdvo2 = []
        self.sdvo3 = []
        self.sdvTv = []
        self.sxo = []
        self.syo = []
        self.J = 200
        self.ts = 0.05
        self.xv = 0
        self.yv = 1
        self.Pv = np.array([[self.xv], [self.yv]])
        self.xvg = 10
        self.yvg = 10
        self.Pvg = np.array([[self.xvg], [self.yvg]])
        self.xo = 0
        self.yo = 0
        self.xo1 = 3.5
        self.yo1 = 2.5
        self.Po1 = np.array([[self.xo1], [self.yo1]])
        self.xo2 = 7.5
        self.yo2 = 8.2
        self.Po2 = np.array([[self.xo2], [self.yo2]])
        self.Po = np.array([[self.xo], [self.yo]])
        self.d = 1
        self.A = 20
        self.B = 8
        self.id = 1

    def reset(self):
        x = self.xv
        y = self.yv
        return np.array([x, y])

    def plot(self, id):

        X = [x[0] for x in self.sPv]
        Y = [x[1] for x in self.sPv]

        Dvo = self.sdvo
        Dvo1 = self.sdvo1
        Dvo2 = self.sdvo2

        DvTv = self.sdvTv
        AA = self.sa
        if (id >= 1):
            file = open('I://dqn_analyse//dqn_data//' + str(id) + '.txt', 'a+')
            file.writelines('---------------------------------fig1---------------------------------' + '\n')
            for item in self.sPv:
                file.writelines(str(item[0][0]) + ' ' + str(item[1][0]) + '\n')
            file.writelines('---------------------------------fig21---------------------------------' + '\n')
            for item in self.sdvo1:
                file.writelines(str(item) + '\n')
            file.writelines('---------------------------------fig22---------------------------------' + '\n')
            for item in self.sdvo2:
                file.writelines(str(item) + '\n')
            file.writelines('---------------------------------fig3---------------------------------' + '\n')
            for item in self.sdvTv:
                file.writelines(str(item) + '\n')
            file.writelines('---------------------------------fig4---------------------------------' + '\n')
            for item in self.sa:
                file.writelines(str(item) + '\n')
            file.close()
            # 清空内容
        plt.figure(12)
        plt.subplot(221)
        plt.plot(X, Y, '-b', self.xo1, self.yo1, 'kd', self.xo2, self.yo2, 'kd', self.xvg,
                 self.yvg, 'kp')

        plt.subplot(222)
        plt.plot(Dvo1, 'b-')
        plt.plot(Dvo2, 'r-')

        # plt.plot(Dvo, 'b-')

        plt.subplot(223)
        plt.plot(DvTv, 'b-')

        plt.subplot(224)
        plt.plot(AA, '-.b')
        plt.savefig('I://dqn_analyse//dqn_data//' + str(id) + '.jpg')
        #plt.show()
        plt.close('all')
        # # 保存数据
        # file = open('D://PycharmProjects//5_Deep_Q_Network_new//data_point//res' + str(id) + '.txt', 'a+')
        # file.writelines('-----------------------fig1-----------------------' + '\n')
        # for item in self.sPv:
        #     file.writelines(str(item[0][0]) + ' ' + str(item[1][0]) + '\n')
        # file.writelines('-----------------------fig2-----------------------' + '\n')
        # for item in self.sdvo1:
        #     print(item)
        #     file.writelines(str(item) + '\n')
        # file.writelines('-----------------------fig3-----------------------' + '\n')
        # for item in self.sdvTv:
        #     file.writelines(str(item) + '\n')
        # file.writelines('-----------------------fig4-----------------------' + '\n')
        # for item in self.sa:
        #     file.writelines(str(item) + '\n')
        # file.close()
        # 清空内容
        self.sPv = []
        self.sa = []
        self.sdvo = []
        self.sdvTv = []
        self.sdvo1 = []
        self.sdvo2 = []

    def moveToTarget(self, Pv, t):
        # 当前位置到障碍物的距离
        dvo1 = sqrt(pow((Pv[0] - self.Po1[0]), 2) + pow((Pv[1] - self.Po1[1]), 2))
        dvo2 = sqrt(pow((Pv[0] - self.Po2[0]), 2) + pow((Pv[1] - self.Po2[1]), 2))

        self.sdvo1.append(dvo1)
        self.sdvo2.append(dvo2)

        if dvo1 <= dvo2:
            dvo = dvo1
            self.xo = self.Po1[0]
            self.yo = self.Po1[1]
        elif dvo2 < dvo1:
            dvo = dvo2
            self.xo = self.Po2[0]
            self.yo = self.Po2[1]

        self.sdvo.append(dvo)
        xTv = 1 + 0.9 * t
        yTv = 1 + 0.9 * t
        PTv = np.array([[xTv], [yTv]])
        # 当前位置到目标位置的距离
        dvTv = sqrt(pow((Pv[0] - PTv[0]), 2) + pow((Pv[1] - PTv[1]), 2))
        self.sdvTv.append(dvTv)
        JB = np.eye(2)
        dpBd = np.array([[0.9], [0.9]])
        VB = np.linalg.pinv(JB).dot((dpBd + (PTv - Pv).dot(self.B)))
        JA = np.array([(Pv[0] - self.xo) / dvo, (Pv[1] - self.yo) / dvo], 'float64').transpose()
        VA = np.linalg.pinv(JA).dot((self.d - dvo)).dot(self.A)
        a = 0
        Vd = VB + (np.eye(2) - np.linalg.pinv(JB).dot(JB)).dot(VA)
        self.sa.append(a)
        Pv = Pv + Vd.dot(self.ts)
        self.sPv.append((Pv[0], Pv[1]))
        return Vd, Pv

    def avoidObstacle(self, Pv, t):
        dvo1 = sqrt(pow((Pv[0] - self.Po1[0]), 2) + pow((Pv[1] - self.Po1[1]), 2))
        dvo2 = sqrt(pow((Pv[0] - self.Po2[0]), 2) + pow((Pv[1] - self.Po2[1]), 2))
        self.sdvo1.append(dvo1)
        self.sdvo2.append(dvo2)

        if dvo1 <= dvo2:
            dvo = dvo1
            self.xo = self.Po1[0]
            self.yo = self.Po1[1]
        elif dvo2 < dvo1:
            dvo = dvo2
            self.xo = self.Po2[0]
            self.yo = self.Po2[1]
        self.sdvo.append(dvo)
        xTv = 1 + 0.9 * t
        yTv = 1 + 0.9 * t
        PTv = np.array([[xTv], [yTv]])
        dvTv = sqrt(pow((Pv[0] - PTv[0]), 2) + pow((Pv[1] - PTv[1]), 2))
        self.sdvTv.append(dvTv)
        JB = np.eye(2)
        dpBd = np.array([[0.9], [0.9]])
        VB = np.linalg.pinv(JB).dot((dpBd + (PTv - Pv).dot(self.B)))
        JA = np.array([(Pv[0] - self.xo) / dvo, (Pv[1] - self.yo) / dvo], 'float64').transpose()

        VA = np.linalg.pinv(JA).dot((self.d - dvo)).dot(self.A)
        a = 1
        # Vd = VB + (np.eye(2) - np.linalg.pinv(JB).dot(JB)).dot(VA)
        Vd = VA + (np.eye(2) - np.linalg.pinv(JA).dot(JA)).dot(VB)
        self.sa.append(a)
        Pv = Pv + Vd.dot(self.ts)
        self.sPv.append((Pv[0], Pv[1]))
        return Vd, Pv

    def step(self, observation, action, t):
        # global n_observation
        cur_x = observation[0]
        cur_y = observation[1]
        done = False
        p_vector = np.array([[cur_x], [cur_y]])
        if action == 0:  # B move to target
            v, n_observation = Env.moveToTarget(self, p_vector, t)
        if action == 1:  # A avoid obstacle
            v, n_observation = Env.avoidObstacle(self, p_vector, t)
        n_x = n_observation[0][0]
        n_y = n_observation[1][0]
        s_ = np.array([n_x, n_y])
        # reward function
        # dt = symbols('dt')

        # err, acc = quad(lambda dt: (((1 + 0.9 * t + 0.9 * dt - cur_x - v[0][0] * dt) ** 2 +
        #                              (1 + 0.9 * t + 0.9 * dt - cur_y - v[1][0] * dt) ** 2) ** (
        #                                     1 / 2)), t, t + self.ts)
        err = ((cur_x + v[0][0] * self.ts - 1 - 0.9 * (t + self.ts)) ** 2 + \
               (cur_y + v[1][0] * self.ts - 1 - 0.9 * (t + self.ts)) ** 2)
        r1 = -tanh(pow(err, 2) - 0.4) * 15
        d_safe = sqrt(pow(n_x - self.xo, 2) + pow(n_y - self.yo, 2))
        r3 = 0
        r2 = 0
        if d_safe < self.d:
            r2 = -12
        if d_safe >= self.d:
            r2 = 0
        # if n_x > cur_x and n_y > cur_y:
        #     r3 = 5
        if n_x < cur_x and n_y < cur_y:
            r3 = -3
        if n_x == self.xvg or n_y == self.yvg:
            r3 = 100
        reward = r1 + r2 + r3
        # print('err: ', err, '      r1:', r1, '  r2:', r2, '  r3:', r3, '   reward:', reward)
        #         print('done:',done,'   t: ',t)
        if r3 == 100 or t == self.J * self.ts:
            print('plot')
            done = true
            Env.plot(self, self.id)
            self.id = self.id + 1

        return s_, reward, done
