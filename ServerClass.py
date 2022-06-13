import math
import random
from typing import List


class Server:

    """
    构造函数
    sv_num: 服务器编号
    ds_connect: 连接的数据源编号集
    f_max: 最大可分配CPU频率
    """

    def __init__(self, sv_num, ds_connect: List[int], f_max=100):
        # vars of property
        self.num = sv_num
        self.connect = ds_connect
        self.f_max = f_max
        self.f_lower = 0
        self.f_upper = 0
        self.random_f_bound()
        # needed vars in algorithm
        self.lent_all = []
        self.theta = 0
        self.f_lent = 0
        self.p0 = 0
        self.p2 = 0
        self.q0 = 1
        self.q2 = 1
        self.algorithm = ''
        # vars used to record task message
        self.task_num = -1
        self.cpu_cycles = 0
        self.ds_used = []
        # extra vars used to count time
        self.time = 0

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def random_f_bound(self):
        self.f_lower = self.f_max / 100.0
        self.f_upper = self.f_max

    def fix_price(self):
        self.p0 = 3
        self.p2 = 3

    def fix_frequency(self):
        f_tmp = self.theta * math.sqrt(self.cpu_cycles)
        if f_tmp > self.f_upper:
            return self.f_upper
        elif f_tmp < self.f_lower:
            return self.f_lower
        else:
            return f_tmp

    def fix_theta(self):
        if self.algorithm == 'DOTA':
            dividend = sum(math.sqrt(elems[1]) * elems[2] for elems in self.lent_all)
            divisor = sum(elems[1] for elems in self.lent_all)
            self.theta = dividend / divisor
        elif self.algorithm == 'MYOPIC' and self.theta == 0:
            lent = self.lent_all[0]
            self.theta = lent[2] / math.sqrt((lent[1]))

    def fix_f_lent(self):
        # check whether chosen
        if self.cpu_cycles == 0:
            return
        # check algorithm
        f_lent_tmp = 0
        if self.algorithm == 'DOTA' or self.algorithm == 'MYOPIC':
            f_lent_tmp = self.fix_frequency()
            if self.theta == 0:
                self.f_lent = (self.f_lower + self.f_upper) / 2
                return
        elif self.algorithm == 'GREEDY':
            f_lent_tmp = self.f_upper
        elif self.algorithm == 'RANDOM':
            f_lent_tmp = random.uniform(self.f_lower, self.f_upper)
        elif self.algorithm == 'CONSERVATIVE':
            f_lent_tmp = self.f_lower
        # make final decision by results of algorithm
        if self.cpu_cycles != 0:
            total_f_lent = sum(elem[2] for elem in self.lent_all)
            if total_f_lent >= self.f_max:
                self.f_lent = 0
            elif f_lent_tmp + total_f_lent > self.f_max:
                self.f_lent = self.f_max - total_f_lent
            else:
                self.f_lent = f_lent_tmp

    def insert_task_msg(self):
        end_time = self.cpu_cycles / self.f_lent + self.time
        task_msg = [self.task_num, self.cpu_cycles, self.f_lent, end_time]
        if len(self.lent_all) == 0:
            self.lent_all.append(task_msg)
        elif len(self.lent_all) != 0 and self.lent_all[-1][3] < end_time:
            self.lent_all.append(task_msg)
        else:
            for elems in self.lent_all:
                if elems[3] > end_time:
                    self.lent_all.insert(self.lent_all.index(elems), task_msg)
                    break

    # first step: publish the connectivity
    def publish_connect(self):
        return [self.num, self.connect]

    # second step: fix price and frequency
    def fix_price_frequency(self, task_num, cpu_cycles, time):
        self.task_num = task_num
        self.cpu_cycles = cpu_cycles
        self.time = time
        self.remove_task(time)
        self.fix_price()
        self.fix_f_lent()
        if self.f_lent == 0:
            return {'SV_NUM': self.num,
                    'SV_FAIL': 1}
        else:
            return {'SV_NUM': self.num,
                    'SV_FAIL': 0,
                    'P0': self.p0,
                    'P2': self.p2,
                    'F_LENT': self.f_lent}

    # third step: update parameters
    def update_params(self, borrow_msg):
        if borrow_msg['T_FAIL'] == 0:
            self.insert_task_msg()
            self.fix_theta()

    # extra step: remove finished task
    def remove_task(self, time):
        if len(self.lent_all) == 0:
            return
        elif self.lent_all[-1][3] < time:
            self.lent_all.clear()
        else:
            for elems in self.lent_all:
                if elems[3] >= time:
                    for i in range(self.lent_all.index(elems)):
                        del (self.lent_all[0])
                    break
