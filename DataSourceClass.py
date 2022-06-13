import math
import random


class DataSource:

    """
    构造函数
    ds_num: 数据源编号
    t_max: 最大可分配传输速率
    """

    def __init__(self, ds_num, t_max=100):
        # vars of property
        self.num = ds_num
        self.t_max = t_max
        self.t_lower = 0
        self.t_upper = 0
        self.random_t_bound()
        # needed vars in algorithm
        self.lent_all = []
        self.theta = 0
        self.t_lent = 0
        self.p1 = 0
        self.p3 = 0
        self.q1 = 1
        self.q3 = 1
        self.algorithm = ''
        # vars used to record task message
        self.task_num = -1
        self.data_size = 0
        self.sv_used = 0
        # extra vars used to count time
        self.time = 0

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def random_t_bound(self):
        self.t_upper = self.t_max
        self.t_lower = self.t_max / 100.0

    def fix_price(self):
        self.p1 = 3
        self.p3 = 3

    def fix_trans_speed(self):
        t_tmp = self.theta * math.sqrt(self.data_size)
        if t_tmp > self.t_upper:
            return self.t_upper
        elif t_tmp < self.t_lower:
            return self.t_lower
        else:
            return t_tmp

    def fix_theta(self):
        if self.algorithm == 'DOTA':
            dividend = sum(math.sqrt(elems[1]) * elems[2] for elems in self.lent_all)
            divisor = sum(elems[1] for elems in self.lent_all)
            self.theta = dividend / divisor
        elif self.algorithm == 'MYOPIC' and self.theta == 0:
            lent = self.lent_all[0]
            self.theta = lent[2] / math.sqrt((lent[1]))

    def fix_t_lent(self):
        # check algorithm
        t_lent_tmp = 0
        if self.algorithm == 'DOTA' or self.algorithm == 'MYOPIC':
            t_lent_tmp = self.fix_trans_speed()
            if self.theta == 0:
                self.t_lent = (self.t_lower + self.t_upper) / 2
                return
        elif self.algorithm == 'GREEDY':
            t_lent_tmp = self.t_upper
        elif self.algorithm == 'RANDOM':
            t_lent_tmp = random.uniform(self.t_lower, self.t_upper)
        elif self.algorithm == 'CONSERVATIVE':
            t_lent_tmp = self.t_lower
        # make final decision by results of algorithm
        if self.data_size != 0:
            total_t_lent = sum(elem[2] for elem in self.lent_all)
            if total_t_lent >= self.t_max:
                self.t_lent = 0
            elif t_lent_tmp + total_t_lent > self.t_max:
                self.t_lent = self.t_max - total_t_lent
            else:
                self.t_lent = t_lent_tmp

    def insert_task_msg(self):
        end_time = self.data_size / self.t_lent + self.time
        task_msg = [self.task_num, self.data_size, self.t_lent, end_time]
        if len(self.lent_all) == 0:
            self.lent_all.append(task_msg)
        elif len(self.lent_all) != 0 and self.lent_all[-1][3] < end_time:
            self.lent_all.append(task_msg)
        else:
            for elems in self.lent_all:
                if elems[3] > end_time:
                    self.lent_all.insert(self.lent_all.index(elems), task_msg)
                    break

    # first step: fix price and transmission speed
    def fix_price_trans_speed(self, task_num, data_size, time):
        self.task_num = task_num
        self.data_size = data_size
        self.time = time
        self.remove_task(time)
        self.fix_price()
        self.fix_t_lent()
        if self.t_lent == 0:
            return {'DS_NUM': self.num,
                    'DS_FAIL': 1}
        else:
            return {'DS_NUM': self.num,
                    'DS_FAIL': 0,
                    'P1': self.p1,
                    'P3': self.p3,
                    'T_LENT': self.t_lent}

    # second step: update parameters
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
