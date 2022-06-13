import random
from typing import List


def list_and(my_list):
    result = True
    for elem in my_list:
        result = result and elem
    return result


class Task:
    """
    构造函数
    t_num: 任务编号
    c: 所需CPU周期数
    n_vec: 所需数据源相关，其中List类型元素格式为[数据源编号, 所需数据量]
    v: 任务期望收益
    time: 任务到达时间
    """

    def __init__(self, t_num: int, c: float,
                 n_vec: List[List],
                 v: int, time: float):
        # vars of property
        self.num = t_num
        # needed vars in algorithm
        self.c = c
        self.n_vec = n_vec
        self.v = v
        self.time = time
        self.sv_chosen = []
        self.sv_decided = 0
        self.f_borrow = 0
        self.sv_pay = 0
        self.ds_pays = []
        self.t_borrow = []
        self.utility = 0
        self.succ = 0
        self.algorithm = ''

    """
    设置使用的算法
    """

    def set_algorithm(self, algorithm: str):
        self.algorithm = algorithm

    """
    第一步：选择满足数据源需求的服务器
    """

    def choose_server(self, sv_connects):
        ds_need = [elems[0] for elems in self.n_vec]
        for elems in sv_connects:
            if not [False for elem in ds_need if elem not in elems[1]]:
                self.sv_chosen.append(elems[0])
        return self.sv_chosen

    """
    第二步：决定是否租借服务器与租借哪个服务器
    """

    def decide_rent(self, sv_msgs, ds_msgs):
        sv_succ_vec = []
        ds_succ_vec = []
        ds_fail_count = 0
        for sv_msg in sv_msgs:
            if sv_msg['SV_FAIL'] == 0:
                sv_num = sv_msg['SV_NUM']
                p0 = sv_msg['P0']
                p2 = sv_msg['P2']
                f_lent = sv_msg['F_LENT']
                sv_succ_vec.append([sv_num, p0, p2, f_lent])
        for ds_msg in ds_msgs:
            if ds_msg['DS_FAIL'] == 0:
                ds_num = ds_msg['DS_NUM']
                p1 = ds_msg['P1']
                p2 = ds_msg['P3']
                t_lent = ds_msg['T_LENT']
                ds_succ_vec.append([ds_num, p1, p2, t_lent])
            else:
                ds_fail_count += 1
        if len(sv_succ_vec) == 0 or ds_fail_count != 0:
            return {'TASK_NUM': self.num,
                    'T_FAIL': 1}
        # choose server
        if self.algorithm == 'RANDOM':  # random choose server
            rand_int = random.randint(0, len(sv_succ_vec) - 1)
            elem = sv_succ_vec[rand_int]
            self.sv_decided = elem[0]
            basic_pay = elem[1] * self.c
            extra_pay = elem[2] * (self.c / elem[3])
            self.sv_pay = basic_pay + extra_pay
            self.f_borrow = elem[3]
        else:  # find the best server
            self.sv_decided = 0
            self.sv_pay = float('inf')
            for elem in sv_succ_vec:
                basic_pay = elem[1] * self.c
                extra_pay = elem[2] * (self.c / elem[3])
                pay = basic_pay + extra_pay
                if pay < self.sv_pay:
                    self.sv_decided = elem[0]
                    self.sv_pay = pay
                    self.f_borrow = elem[3]
        # calculate total pay to data sources
        ds_pays = 0
        for elem in ds_succ_vec:
            n = [n_msg[1] for n_msg in self.n_vec if n_msg[0] == elem[0]][0]
            basic_pay = elem[1] * n
            extra_pay = elem[2] * (n / elem[3])
            ds_pay = basic_pay + extra_pay
            self.ds_pays.append([elem[0], ds_pay])
            self.t_borrow.append([elem[0], elem[3]])
            ds_pays += ds_pay
        # check utility
        self.utility = self.v - self.sv_pay - ds_pays
        if self.utility < 0:
            return {'TASK_NUM': self.num,
                    'T_FAIL': 1}
        else:
            self.succ = 1
            return {'TASK_NUM': self.num,
                    'T_FAIL': 0,
                    'SV_CHOSEN': self.sv_decided,
                    'SV_PAY': self.sv_pay,
                    'DS_PAYS': self.ds_pays
                    }
