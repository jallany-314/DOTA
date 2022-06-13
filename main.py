import copy
import csv
import json
import random
import time
from typing import Dict, List

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.pyplot import MultipleLocator
from matplotlib.pyplot import LinearLocator
from sko.tools import set_run_mode
from myRCGA import myRCGA
import DataSourceClass
import ServerClass
import TaskClass

'''
define params
'''
# 数据集路径相关配置
google_file_head = r'D:\学习\毕业论文\数据集\google cluster\instance_events-change'
google_file_tail = '.json'

ali_machine_head = r'D:\学习\毕业论文\数据集\ali cluster\machine_meta_'
ali_task_head = r'D:\学习\毕业论文\数据集\ali cluster\batch_task_'
ali_tail = '.csv'

# 以下为用于初始化的配置参数
task_count = 0  # 任务数
sv_count = 0  # 服务器数
ds_count = 0  # 数据源数
connect_count = 0  # 服务器连接数据源数（度），仅在rand_init_params函数的priority参数为1时表示标记的服务器编号

# 以下为初始化后的测试样例相关数据
f_max: List[float] = []  # 所有服务器最大可分配CPU频率
t_max: List[float] = []  # 所有数据源最大可分配传输速率

tasks: List[type(TaskClass)] = []  # 所有任务
svs: List[type(ServerClass)] = []  # 所有服务器
dss: List[type(DataSourceClass)] = []  # 所有数据源
connects: List[List] = []  # [[服务器编号,[数据源编号]]] 所有连接

tasks_copy: List[type(TaskClass)] = []  # 所有任务初始化后的备份
svs_copy: List[type(ServerClass)] = []  # 所有服务器初始化后的备份
dss_copy: List[type(DataSourceClass)] = []  # 所有数据源初始化后的备份
f_max_copy = []  # 所有最大可分配CPU频率初始化后的备份

# 运行时间测试时用到的相关参数
test_time: int = 0  # 是否在进行时间测试
run_time: int = 0  # 运行时间

# 运行利用率测试时用到的相关参数
test_util = 0  # 是否在进行利用率测试
f_util: Dict = {}  # 各算法CPU频率利用率, key为算法名称
t_util: Dict = {}  # 各算法传输速率利用率, key为算法名称
f_total = 0  # 所有服务器总CPU频率
t_total = 0  # 所有数据源总CPU频率

"""
按设定参数初始化（测试用）
"""


def init_params():
    task = TaskClass.Task(1, 20, [[1, 20], [2, 15]], 100, 0)
    sv = ServerClass.Server(1, [1, 2, 3])
    ds = DataSourceClass.DataSource(1)
    tasks.append(task)
    svs.append(sv)
    dss.append(ds)


"""
初始化配置参数以及清空测试样例
"""


def set_counts_clear(task_count_tmp, sv_count_tmp, ds_count_tmp, connect_count_tmp):
    global task_count
    global sv_count
    global ds_count
    global connect_count
    global f_max
    global t_max
    global tasks
    global svs
    global dss
    global connects
    global tasks_copy
    global svs_copy
    global dss_copy

    task_count = task_count_tmp
    sv_count = sv_count_tmp
    ds_count = ds_count_tmp
    connect_count = connect_count_tmp

    f_max = []
    t_max = []
    tasks = []
    svs = []
    dss = []
    connects = []
    tasks_copy = []
    svs_copy = []
    dss_copy = []


"""
根据配置参数随机生成测试样例
data_set: 所选数据集名，None为不选数据集
task_set_size: 选择任务数据集大小，对应执行完'xxx_cluster_prepare.py'后的结果
machine_set_size: 选择机器数据集大小，对应执行完'xxx_cluster_prepare.py'后的结果
priority: 为0时初始化顺序为'数据源-连接-服务器-任务'；为1时初始化顺序为'数据源-任务-连接-服务器'
"""


def rand_init_params(data_set=None, task_set_size=10000, machine_set_size=10000, priority=0):
    global tasks_copy
    global svs_copy
    global dss_copy
    global f_max_copy
    global f_total
    global t_total

    # 定义任务、服务器、数据源需设置的信息
    task_infos: List[List[float]] = []  # [[start time, cpu_needed, data_needed]]
    sv_infos: List[float] = [100] * sv_count  # [f_max] 设置所有服务器默认f_max为100
    ds_infos: List[float] = [100] * ds_count  # [t_max] 设置所有数据源默认f_max为100

    # 从数据集获取相关数据
    if data_set == 'google_cluster':  # google数据集相关
        # 读取信息
        google_file_path = google_file_head + str(task_set_size) + google_file_tail
        google_file = open(google_file_path, 'r')
        google_contents = google_file.readlines()
        contents_choose = random.sample(google_contents, task_count)
        # 使用信息
        for content in contents_choose:
            content_dict = json.loads(content)
            time = int(content_dict['time'])
            resource_request = content_dict['resource_request']
            cpu_needed = float(resource_request['cpus'])
            data_needed = float(resource_request['memory'])
            task_infos.append([time, cpu_needed, data_needed])  # 任务获取时间、CPU周期需求和数据量需求
        google_file.close()
    elif data_set == 'ali_cluster':  # 阿里数据集相关
        # 读取信息
        ali_task_path = ali_task_head + str(task_set_size) + ali_tail
        ali_machine_path = ali_machine_head + str(machine_set_size) + ali_tail
        ali_task = open(ali_task_path, 'r')
        ali_machine = open(ali_machine_path, 'r')
        ali_task_csv = list(csv.reader(ali_task))
        ali_machine_csv = list(csv.reader(ali_machine))
        ali_sv_csv = [elem[0] for elem in ali_machine_csv]
        ali_ds_csv = [elem[1] for elem in ali_machine_csv]
        # 使用信息
        task_infos_tmp = random.sample(ali_task_csv, task_count)  # 任务获取到达时间、CPU周期需求和数据量需求
        sv_infos_tmp = random.sample(ali_sv_csv, sv_count)  # 服务器获取总CPU频率
        ds_infos_tmp = random.sample(ali_ds_csv, ds_count)  # 数据源获取总传输速率
        # 转换为正确类型格式
        task_infos = [[int(elem[0]), float(elem[1]), float(elem[2])] for elem in task_infos_tmp]
        sv_infos = [float(elem) for elem in sv_infos_tmp]
        ds_infos = [float(elem) for elem in ds_infos_tmp]
    else:  # 不使用数据集
        time = 0  # 初始任务到达时间设为0
        for i in range(task_count):
            # 随机设置CPU周期需求1-100和数据量需求1-100
            task_infos.append([time, random.randint(1, 100), random.randint(1, 100)])
            time += random.uniform(0, 0.5)  # 下一个任务到达时间与前一个相距0-0.5

    # 初始化数据源
    for i in range(1, ds_count + 1):
        # 新建数据源
        ds = DataSourceClass.DataSource(i, ds_infos[i - 1])  # 参数：数据源编号、总传输速率
        dss.append(ds)
        # 更新t_max
        t_max.append(ds.t_max)

    # 初始化顺序为'数据源-连接-服务器-任务'
    if priority == 0:
        # 初始化服务器与数据源的连接情况
        ds_list = range(1, ds_count + 1)
        unselected_list = ds_list
        # 遍历所有服务器
        for i in range(1, sv_count + 1):
            # 如果有未建立连接的数据源，优先为其建立连接
            if len(unselected_list) != 0:
                # 如果未建立连接的数据源数量大于度，随机选择建立连接
                if len(unselected_list) >= connect_count:
                    connect = random.sample(unselected_list, connect_count)
                    unselected_list = [i for i in unselected_list if i not in connect]
                # 如果未建立连接的数据源数量小于度，先分配完这些，再随机建立连接
                else:
                    tmp_list = [i for i in ds_list if i not in unselected_list]
                    connect = unselected_list + random.sample(tmp_list, connect_count - len(unselected_list))
                    unselected_list.clear()
            # 如果没有未建立连接的数据源，随机建立连接
            else:
                connect = random.sample(ds_list, connect_count)
            connect.sort()
            connects.append([i, connect])

        # 初始化服务器
        for i in range(1, sv_count + 1):
            sv = ServerClass.Server(i, connects[i - 1][1], sv_infos[i - 1])  # 参数：服务器编号、连接情况、总CPU频率
            svs.append(sv)
            # 更新f_max
            f_max.append(sv.f_max)

        # 服务器的连接情况两两做交集，得到同时由至少两个服务器连接的数据源集合的集合sv_intersects
        sv_intersects = []
        for i in range(1, sv_count):
            for j in range(i + 1, sv_count + 1):
                sv_intersect = [k for k in connects[i - 1][1] if k in connects[j - 1][1]]
                if len(sv_intersect) != 0:
                    sv_intersects.append([i, j, sv_intersect])

        # 初始化任务
        sv_intersects_len = len(sv_intersects)
        # 遍历所有任务
        for i in range(1, task_count + 1):
            n_vec = []
            # 如果连接交集不为空，则选一个数据源集合，从中随机挑选一或多个数据源作为目标数据源
            if sv_intersects_len != 0:
                rand_int = random.randint(0, sv_intersects_len - 1)
                sv_intersect = sv_intersects[rand_int][2]
                sv_intersect_len = len(sv_intersect)
                dss_selected = random.sample(sv_intersect, random.randint(1, sv_intersect_len))
                dss_selected.sort()
                for j in dss_selected:
                    # 更新n_vec，存入数据源编号和随机1-100的数据量需求
                    n_vec.append([j, random.randint(1, 100) if data_set is None else task_infos[i - 1][2]])
            # 如果连接交集为空，则随机挑选一个数据源作为目标数据源
            else:
                t_needed = random.randint(1, 100) if data_set is None else task_infos[i - 1][2]
                # 更新n_vec，存入数据源编号和随机1-100的数据量需求
                n_vec.append([random.randint(1, ds_count), t_needed])
            n_vec_len = len(n_vec)
            # 设置任务期望收益为随机 [200*所需数据源数, 210*所需数据源数]
            # 可修改为随机 [系数1*资源总量, 系数2*资源总量]
            # 其中，资源总量=CPU周期数+总数据量需求
            v = random.randint(200 * n_vec_len, 210 * n_vec_len)
            # 生成任务
            task = TaskClass.Task(i, task_infos[i - 1][1], n_vec, v, task_infos[i - 1][0])
            tasks.append(task)
    # 初始化顺序为'数据源-任务-连接-服务器'，仅在测试服务器数量时被设置
    elif priority == 1:
        # 初始化任务
        xil = []
        for i in range(1, task_count + 1):
            n_vec = []
            # 随机决定选择几个数据源
            ds_choose_count = random.randint(1, ds_count)
            # 如果不是最后一个任务，随机选择一个或多个数据源提出需求
            if i != task_count:
                xil.append(random.sample(list(range(1, ds_count + 1)), ds_choose_count))
            # 如果是最后一个任务，选择被选择过的数据源并随机选择其他数据源提出需求
            else:
                ds_choose1 = random.sample(list(range(1, ds_count + 1)), ds_choose_count)
                ds_used = []
                for j in range(task_count - 1):
                    ds_used += xil[j]
                ds_choose2 = list(set(range(1, ds_count + 1)) - set(ds_used))
                ds_choose = list(set(ds_choose1 + ds_choose2))
                ds_choose.sort()
                xil.append(ds_choose)
            # 对于选择的数据源，初始化n_vec
            for k in xil[i - 1]:
                n_vec.append([k, random.randint(1, 100) if data_set is None else task_infos[i - 1][2]])
            n_vec_len = len(n_vec)
            # 设置任务期望收益为随机 [200*所需数据源数, 210*所需数据源数]
            v = random.randint(200 * n_vec_len, 210 * n_vec_len)
            task = TaskClass.Task(i, task_infos[i - 1][1], n_vec, v, task_infos[i - 1][0])
            tasks.append(task)

        # 初始化服务器
        task_list = list(range(1, task_count + 1))
        # 遍历所有服务器
        for j in range(1, sv_count + 1):
            # 随机选择可以执行的任务
            task_choose_count = random.randint(1, task_count)
            task_used = random.sample(task_list, task_choose_count)
            # 如果是标记的服务器，需要加上未被选择的任务
            if j == connect_count:
                tmp = []
                for i in range(connect_count - 1):
                    tmp += connects[i][1]
                task_unused = list(set(task_list) - set(tmp))
                task_used += task_unused
            ds_connect = []

            # 根据选择的任务需要的数据源建立对应的连接
            for i in task_used:
                ds_connect += xil[i - 1]
            ds_connect = list(set(ds_connect))
            ds_connect.sort()
            connects.append([j, ds_connect])
            # 生成任务
            sv = ServerClass.Server(j, ds_connect, sv_infos[j - 1])
            svs.append(sv)
            # 更新f_max
            f_max.append(sv.f_max)

    # 拷贝以备份初始化的数据
    tasks_copy = copy.deepcopy(tasks)
    svs_copy = copy.deepcopy(svs)
    dss_copy = copy.deepcopy(dss)
    f_max_copy = copy.deepcopy(f_max)

    # 计算总CPU频率和传输速率
    f_total = sum(f_max)
    t_total = sum(t_max)


"""
将测试后被修改过的测试样例还原为初始化值
"""


def reset_params():
    global tasks
    global svs
    global dss
    global f_max

    tasks = copy.deepcopy(tasks_copy)
    svs = copy.deepcopy(svs_copy)
    dss = copy.deepcopy(dss_copy)
    f_max = copy.deepcopy(f_max_copy)


"""
给所有任务、服务器、数据源的algorithm属性配置当前测试算法
"""


def set_algorithm(algorithm):
    for task in tasks:
        task.set_algorithm(algorithm)
    for sv in svs:
        sv.set_algorithm(algorithm)
    for ds in dss:
        ds.set_algorithm(algorithm)


"""
运行在线算法的总流程
"""


def start_algorithm():
    global run_time

    start_time = 0
    # 如果在测试运行时间，则开始计时
    if test_time == 1:
        start_time = time.perf_counter()

    alg = tasks[0].algorithm

    # 算法开始
    for task in tasks:
        # 步骤1: 所有服务器发布连接信息
        sv_connects = []
        for sv in svs:
            sv_connects.append(sv.publish_connect())
        # 步骤2: 任务选择能满足数据源需求的服务器
        sv_chosen = task.choose_server(sv_connects)
        # 步骤3: 服务器决定定价和租借CPU频率
        sv_msgs = []
        for i in sv_chosen:
            sv_msg = svs[i - 1].fix_price_frequency(task.num, task.c, task.time)
            sv_msgs.append(sv_msg)
        # 步骤4: 数据源决定定价和租借传输速率
        ds_chosen = [elem[0] for elem in task.n_vec]
        ds_msgs = []
        for i in ds_chosen:
            n = [elem[1] for elem in task.n_vec if elem[0] == i][0]
            ds_msg = dss[i - 1].fix_price_trans_speed(task.num, n, task.time)
            ds_msgs.append(ds_msg)
        # 步骤5: 任务决定是否租借与租借哪个服务器
        task_msg = task.decide_rent(sv_msgs, ds_msgs)
        # step 6: servers and data sources update parameters
        if task_msg['T_FAIL'] == 0:
            sv_decided = task_msg['SV_CHOSEN']
            for i in sv_chosen:
                if i == sv_decided:
                    svs[i - 1].update_params(task_msg)
                else:
                    svs[i - 1].update_params({'T_FAIL': 1})
            for i in ds_chosen:
                dss[i - 1].update_params(task_msg)
        # 如果在进行利用率测试，则计算
        if test_util == 1:
            total_f_lent = 0
            total_t_lent = 0
            for sv in svs:
                total_f_lent += sum(elem[2] for elem in sv.lent_all)
            for ds in dss:
                total_t_lent += sum(elem[2] for elem in ds.lent_all)
            f_util[alg].append(total_f_lent / f_total)
            t_util[alg].append(total_t_lent / t_total)

    # 如果在运行时间测试，则停止计时
    if test_time == 1:
        end_time = time.perf_counter()
        run_time = end_time - start_time


"""
计算成功数与社会总效用
"""


def count_result():
    succ = 0
    revenue = 0
    for task in tasks:
        if task.succ == 1:
            # count success
            succ += 1
            # calculate total revenue
            sv = svs[task.sv_decided - 1]
            sv_cost = task.c * sv.q0 + (task.c / task.f_borrow) * sv.q2
            dss_cost = 0
            for i in [elem[0] for elem in task.ds_pays]:
                ds = dss[i - 1]
                n = sum([elem[1] for elem in task.n_vec if elem[0] == i])
                t = sum([elem[1] for elem in task.t_borrow if elem[0] == i])
                dss_cost += n * ds.q1 + (n / t) * ds.q3
            revenue += task.v - sv_cost - dss_cost
    return succ, revenue


'''
定义DOTA算法
'''


def dota():
    # reset params message
    reset_params()
    # init algorithm message
    set_algorithm('DOTA')
    # start algorithm
    start_algorithm()
    # count result
    return count_result()


'''
定义贪婪算法
'''


def greedy():
    # reset params message
    reset_params()
    # init algorithm message
    set_algorithm('GREEDY')
    # start algorithm
    start_algorithm()
    # count_result
    return count_result()


'''
定义随机算法
'''


def rand():
    # reset params message
    reset_params()
    # init algorithm message
    set_algorithm('RANDOM')
    # start algorithm
    start_algorithm()
    # count_result
    return count_result()


'''
定义目光短浅算法
'''


def myopic():
    # reset params message
    reset_params()
    # init algorithm message
    set_algorithm('MYOPIC')
    # start algorithm
    start_algorithm()
    # count_result
    succ, revenue = count_result()
    print('myopic:', revenue)
    return succ, revenue


'''
定义保守算法
'''


def conservative():
    # reset params message
    reset_params()
    # init algorithm message
    set_algorithm('CONSERVATIVE')
    # start algorithm
    start_algorithm()
    # count_result
    return count_result()


'''
定义csv算法（该项目无法使用，会报错）
'''


def cvx():
    # define variables and parameters
    xij = cp.Variable(shape=(task_count, sv_count), boolean=True)
    fij = cp.Variable(shape=(task_count, sv_count), nonneg=True)
    til = cp.Variable(shape=(task_count, ds_count), nonneg=True)
    vi = cp.Parameter(shape=(1, task_count), nonneg=True)
    ci = cp.Parameter(shape=(1, task_count), nonneg=True)
    xil = cp.Parameter(shape=(task_count, ds_count), boolean=True)
    nil = cp.Parameter(shape=(task_count, ds_count), nonneg=True)
    q0 = cp.Parameter(shape=(sv_count, 1), nonneg=True)
    q1 = cp.Parameter(shape=(ds_count, 1), nonneg=True)
    q2 = cp.Parameter(shape=(sv_count, 1), nonneg=True)
    q3 = cp.Parameter(shape=(ds_count, 1), nonneg=True)
    # construct objective
    tasks_revenue = cp.sum(vi @ cp.sum(xij, axis=1))
    svs_basic_cost = cp.sum(ci @ xij @ q0)
    svs_extra_cost = cp.sum(ci @ cp.multiply(xij, cp.inv_pos(fij)) @ q2)
    dss_basic_cost = cp.sum(cp.sum(xij, axis=1).T @ cp.multiply(nil, xil) @ q1)
    dss_extra_cost = cp.sum(cp.sum(xij, axis=1).T @ cp.multiply(cp.multiply(nil, cp.inv_pos(til)), xil) @ q3)
    obj = cp.Maximize(tasks_revenue - svs_basic_cost - svs_extra_cost - dss_basic_cost - dss_extra_cost)
    # construct constraints
    constraints = [cp.sum(xij, axis=1) <= 1, cp.sum(fij) <= 1, cp.sum(til) <= 1]
    # construct problem
    prob = cp.Problem(obj, constraints)
    # initialize parameters
    vi_tmp = []
    ci_tmp = []
    xil_tmp = []
    nil_tmp = []
    q0_tmp = []
    q1_tmp = []
    q2_tmp = []
    q3_tmp = []
    for task in tasks:
        vi_tmp.append([task.v])
        ci_tmp.append([task.c])
        xil_i_tmp = [0] * ds_count
        nil_i_tmp = [0] * ds_count
        for elem in task.n_vec:
            xil_i_tmp[elem[0] - 1] = 1
            nil_i_tmp[elem[0] - 1] = elem[1]
        xil_tmp.append(xil_i_tmp)
        nil_tmp.append(nil_i_tmp)
    xil_tmp = list(zip(*xil_tmp))
    nil_tmp = list(zip(*nil_tmp))
    for sv in svs:
        q0_tmp.append(sv.q0)
        q2_tmp.append(sv.q2)
    for ds in dss:
        q1_tmp.append(ds.q1)
        q3_tmp.append(ds.q3)
    vi.value = vi_tmp
    ci.value = ci_tmp
    xil.value = xil_tmp
    nil.value = nil_tmp
    q0.value = [q0_tmp]
    q1.value = [q1_tmp]
    q2.value = [q2_tmp]
    q3.value = [q3_tmp]
    # solve problem
    if prob.is_dcp():
        result = prob.solve()
        print(result)
    else:
        print('Problem is not dcp')


"""
定义遗传算法
"""


def genetic():
    # 定义社会总效用计算公式（适应度函数）
    def cal_total_revenue(task_count, sv_count, ds_count):
        def cal(p):
            xij = np.array(p[0: task_count * sv_count]).reshape(task_count, sv_count)
            fij = np.array(p[task_count * sv_count: 2 * task_count * sv_count]).reshape(task_count, sv_count)
            til = np.array(p[2 * task_count * sv_count: len(p)]).reshape(task_count, ds_count)
            tasks_revenue = np.sum(vi @ np.sum(xij, axis=1))
            svs_basic_cost = np.sum(ci @ xij @ q0)
            xij_dvd_fij = np.zeros((task_count, sv_count))
            for i in range(task_count):
                for j in range(sv_count):
                    if fij[i][j] != 0:
                        xij_dvd_fij[i][j] = xij[i][j] / fij[i][j]
            svs_extra_cost = np.sum(ci @ xij_dvd_fij @ q2)
            dss_basic_cost = np.sum(np.sum(xij, axis=1).T @ (nil * xil) @ q1)
            nil_xil_dvd_til = np.zeros((task_count, ds_count))
            for i in range(task_count):
                for k in range(ds_count):
                    if til[i][k] != 0:
                        nil_xil_dvd_til[i][k] = nil[i][k] * xil[i][k] / til[i][k]
            dss_extra_cost = np.sum(np.sum(xij, axis=1).T @ nil_xil_dvd_til @ q3)
            return -1 * (tasks_revenue - svs_basic_cost - svs_extra_cost - dss_basic_cost - dss_extra_cost)

        return cal

    global run_time

    set_run_mode(cal_total_revenue, 'vectorization')
    # reset params message
    reset_params()
    # init algorithm message
    set_algorithm('GENETIC')
    # initialize parameters
    vi_tmp = []
    ci_tmp = []
    xil_tmp = []
    nil_tmp = []
    q0_tmp = []
    q1_tmp = []
    q2_tmp = []
    q3_tmp = []
    for task in tasks:
        vi_tmp.append(task.v)
        ci_tmp.append(task.c)
        xil_i_tmp = [0] * ds_count
        nil_i_tmp = [0] * ds_count
        for elem in task.n_vec:
            xil_i_tmp[elem[0] - 1] = 1
            nil_i_tmp[elem[0] - 1] = elem[1]
        xil_tmp.append(xil_i_tmp)
        nil_tmp.append(nil_i_tmp)
    for sv in svs:
        q0_tmp.append([sv.q0])
        q2_tmp.append([sv.q2])
    for ds in dss:
        q1_tmp.append([ds.q1])
        q3_tmp.append([ds.q3])
    vi = np.array(vi_tmp)
    ci = np.array(ci_tmp)
    xil = np.array(xil_tmp)
    nil = np.array(nil_tmp)
    q0 = np.array(q0_tmp)
    q1 = np.array(q1_tmp)
    q2 = np.array(q2_tmp)
    q3 = np.array(q3_tmp)
    # prepare params for algorithm
    n_dim = 2 * task_count * sv_count + task_count * ds_count
    lb = []
    ub = []
    # prepare lb and ub
    sv_f_lower = []
    sv_f_upper = []
    ds_t_lower = []
    ds_t_upper = []
    for sv in svs:
        sv_f_lower.append(sv.f_lower)
        sv_f_upper.append(sv.f_upper)
    for ds in dss:
        ds_t_lower.append(ds.t_lower)
        ds_t_upper.append(ds.t_upper)
    lb += [0] * task_count * sv_count + sv_f_lower * task_count + ds_t_lower * task_count
    ub += [1] * task_count * sv_count + sv_f_upper * task_count + ds_t_upper * task_count
    # start algorithm
    ga = myRCGA(func=cal_total_revenue(task_count, sv_count, ds_count),
                n_dim=n_dim,
                size_pop=50,
                max_iter=100,
                lb=lb,
                ub=ub,
                task_count=task_count,
                sv_count=sv_count,
                ds_count=ds_count,
                f_max=f_max,
                t_max=t_max,
                xil=xil
                )
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # count result
    start_time = 0
    # 如果在运行时间测试，则开始计时
    if test_time == 1:
        start_time = time.perf_counter()
    vars, revenue = ga.run()
    # 如果在运行时间测试，则停止计时
    if test_time == 1:
        end_time = time.perf_counter()
        run_time = end_time - start_time
    # 计算成功数和社会总效用
    succ = sum(vars[0: task_count * sv_count])
    revenue = revenue[0] * -1
    return succ, revenue


'''
test start
'''

algorithms = [
    'DOTA',
    'MYOPIC',
    'GREEDY',
    'RANDOM',
    'GENETIC',
    'CONSERVATIVE'
]

alg_dict = {
    'DOTA': dota,
    'GREEDY': greedy,
    'RANDOM': rand,
    'MYOPIC': myopic,
    'CONSERVATIVE': conservative,
    'GENETIC': genetic
}

succ_count: Dict = {}
total_revenue: Dict = {}
total_time: Dict = {}

save_path_head_normal = r'D:\学习\毕业论文\实验结果图\\'
save_path_head_google = r'D:\学习\毕业论文\实验结果图\google\\'
save_path_head_ali = r'D:\学习\毕业论文\实验结果图\ali\\'

save_path_tail_png = '.png'
save_path_tail_csv = '.csv'

"""
清空存放的测试结果
"""


def result_clear():
    global succ_count
    global total_revenue
    global total_time
    global f_util
    global t_util

    succ_count = {
        'DOTA': [],
        'GREEDY': [],
        'RANDOM': [],
        'MYOPIC': [],
        'CONSERVATIVE': [],
        'GENETIC': []
    }
    total_revenue = {
        'DOTA': [],
        'GREEDY': [],
        'RANDOM': [],
        'MYOPIC': [],
        'CONSERVATIVE': [],
        'GENETIC': []
    }
    total_time = {
        'DOTA': [],
        'GREEDY': [],
        'RANDOM': [],
        'MYOPIC': [],
        'CONSERVATIVE': [],
        'GENETIC': []
    }
    f_util = {
        'DOTA': [],
        'GREEDY': [],
        'RANDOM': [],
        'MYOPIC': [],
        'CONSERVATIVE': [],
        'GENETIC': []
    }
    t_util = {
        'DOTA': [],
        'GREEDY': [],
        'RANDOM': [],
        'MYOPIC': [],
        'CONSERVATIVE': [],
        'GENETIC': []
    }


"""
定义测试结果画图函数
"""


def plt_plot(title, x_data, measure_unit=None):
    # check save path
    if data_set == 'google_cluster':
        save_path_head = save_path_head_google
    elif data_set == 'ali_cluster':
        save_path_head = save_path_head_ali
    else:
        save_path_head = save_path_head_normal

    plt.grid(linestyle=":")
    if measure_unit is not None:
        plt.xlabel(title + '(' + measure_unit + ')')
    else:
        plt.xlabel(title)
    plt.ylabel('success count')
    # x_major_locator = plt.LinearLocator(numticks=len(x_data))
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.xlim(x_data[0], x_data[-1])
    plt.xticks(x_data, x_data)
    # plot about succ count
    for alg in algorithms:
        if alg == 'DOTA':
            plt.plot(x_data, succ_count[alg], 'o-', label=alg, zorder=len(alg), markerfacecolor='white', markersize=9)
        elif alg == 'MYOPIC':
            plt.plot(x_data, succ_count[alg], '>-', label=alg, markerfacecolor='white', markersize=9)
        elif alg == 'GREEDY':
            plt.plot(x_data, succ_count[alg], 'D-', label=alg, markerfacecolor='white', markersize=9)
        elif alg == 'GENETIC':
            plt.plot(x_data, succ_count[alg], 'v-', label=alg, markerfacecolor='white', markersize=9)
        elif alg == 'CONSERVATIVE':
            plt.plot(x_data, succ_count[alg], 's-', label=alg, markerfacecolor='white', markersize=9)
        elif alg == 'RANDOM':
            plt.plot(x_data, succ_count[alg], 'x-', label=alg, markerfacecolor='white', markersize=9)
    plt.legend(loc=1, labelspacing=2, handlelength=2, fontsize=6, shadow=False)
    plt.savefig(save_path_head + title + '相关测试成功数' + save_path_tail_png)
    # plot about total revenue
    plt.clf()
    plt.grid(linestyle=":")
    if measure_unit is not None:
        plt.xlabel(title + '(' + measure_unit + ')')
    else:
        plt.xlabel(title)
    plt.ylabel('total revenue')
    # x_major_locator = plt.LinearLocator(numticks=len(x_data))
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.xlim(x_data[0], x_data[-1])
    plt.xticks(x_data, x_data)
    for alg in algorithms:
        print(alg, ':', total_revenue[alg])
        if alg == 'DOTA':
            plt.plot(x_data, total_revenue[alg], 'o-', label=alg, zorder=len(alg), markerfacecolor='white',
                     markersize=9)
        elif alg == 'MYOPIC':
            plt.plot(x_data, total_revenue[alg], '>-', label=alg, markerfacecolor='white', markersize=9)
        elif alg == 'GREEDY':
            plt.plot(x_data, total_revenue[alg], 'd-', label=alg, markerfacecolor='white', markersize=9)
        elif alg == 'GENETIC':
            plt.plot(x_data, total_revenue[alg], 'v-', label=alg, markerfacecolor='white', markersize=9)
        elif alg == 'CONSERVATIVE':
            plt.plot(x_data, total_revenue[alg], 's-', label=alg, markerfacecolor='white', markersize=9)
        elif alg == 'RANDOM':
            plt.plot(x_data, total_revenue[alg], 'x-', label=alg, markerfacecolor='white', markersize=9)
    plt.legend(loc=1, labelspacing=2, handlelength=2, fontsize=6, shadow=False)
    plt.savefig(save_path_head + title + '相关测试总收益' + save_path_tail_png)
    print(title + '相关测试完成')


"""
定义运行时间测试结果画图函数
"""


def plt_plot_time(title, x_data):
    # check save path
    if data_set == 'google_cluster':
        save_path_head = save_path_head_google
    elif data_set == 'ali_cluster':
        save_path_head = save_path_head_ali
    else:
        save_path_head = save_path_head_normal

    plt.grid(linestyle=":")
    plt.xlabel(title)
    plt.ylabel('run time')
    # x_major_locator = plt.LinearLocator(numticks=len(x_data))
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.xlim(x_data[0], x_data[-1])
    plt.xticks(x_data, x_data)
    # plot about succ count
    for alg in algorithms:
        if alg == 'GENETIC':
            continue
        elif alg == 'DOTA':
            plt.plot(x_data, total_time[alg], 'o-', label=alg, zorder=len(alg), markerfacecolor='white', markersize=9)
        elif alg == 'MYOPIC':
            plt.plot(x_data, total_time[alg], '>-', label=alg, markerfacecolor='white', markersize=9)
        elif alg == 'GREEDY':
            plt.plot(x_data, total_time[alg], 'd-', label=alg, markerfacecolor='white', markersize=9)
        elif alg == 'GENETIC':
            plt.plot(x_data, total_time[alg], 'v-', label=alg, markerfacecolor='white', markersize=9)
        elif alg == 'CONSERVATIVE':
            plt.plot(x_data, total_time[alg], 's-', label=alg, markerfacecolor='white', markersize=9)
        elif alg == 'RANDOM':
            plt.plot(x_data, total_time[alg], 'x-', label=alg, markerfacecolor='white', markersize=9)
    plt.legend(loc=1, labelspacing=2, handlelength=2, fontsize=6, shadow=False)
    plt.savefig(save_path_head + title + '相关测试运行时间' + save_path_tail_png)
    print(title + '相关测试完成')


"""
定义利用率测试结果画图函数
"""


def plt_plot_util(x_data):
    # check save path
    if data_set == 'google_cluster':
        save_path_head = save_path_head_google
    elif data_set == 'ali_cluster':
        save_path_head = save_path_head_ali
    else:
        save_path_head = save_path_head_normal

    plt.grid(linestyle=":")
    plt.xlabel('task num')
    plt.ylabel('f utilization rate')
    x_major_locator = plt.LinearLocator(numticks=len(x_data))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(x_data[0], x_data[-1])
    # plot about succ count
    for alg in algorithms:
        if alg == 'DOTA':
            plt.plot(x_data, f_util[alg], label=alg, linewidth=3, zorder=len(alg))
        elif alg != 'GENETIC':
            plt.plot(x_data, f_util[alg], label=alg, linestyle=':')
    plt.legend(loc=1, labelspacing=2, handlelength=2, fontsize=6, shadow=False)
    plt.savefig(save_path_head + 'f utilization rate相关测试' + save_path_tail_png)
    # plot about total revenue
    plt.clf()
    plt.grid(linestyle=":")
    plt.xlabel('task num')
    plt.ylabel('t utilization rate')
    x_major_locator = plt.LinearLocator(numticks=len(x_data))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(x_data[0], x_data[-1])
    # plot about succ count
    for alg in algorithms:
        if alg == 'DOTA':
            plt.plot(x_data, t_util[alg], label=alg, linewidth=3, zorder=len(alg))
        elif alg != 'GENETIC':
            plt.plot(x_data, t_util[alg], label=alg, linestyle=':')
    plt.legend(loc=1, labelspacing=2, handlelength=2, fontsize=6, shadow=False)
    plt.savefig(save_path_head + 't utilization rate相关测试' + save_path_tail_png)
    print('utilization rate相关测试完成')


"""
定义测试结果存csv函数
"""


def csv_save(title, header):
    # check save path
    if data_set == 'google_cluster':
        save_path_head = save_path_head_google
    elif data_set == 'ali_cluster':
        save_path_head = save_path_head_ali
    else:
        save_path_head = save_path_head_normal

    header = [''] + header

    file = open(save_path_head + title + '相关测试成功数' + save_path_tail_csv, 'w', newline='')
    rows = []
    for alg in algorithms:
        rows.append([alg] + succ_count[alg])
    f_csv = csv.writer(file)
    f_csv.writerow(header)
    f_csv.writerows(rows)
    file.close()

    file = open(save_path_head + title + '相关测试总收益' + save_path_tail_csv, 'w', newline='')
    rows = []
    for alg in algorithms:
        print(alg, ':', total_revenue[alg])
        rows.append([alg] + total_revenue[alg])
    f_csv = csv.writer(file)
    f_csv.writerow(header)
    f_csv.writerows(rows)
    file.close()


"""
定义运行时间测试结果存csv函数
"""


def csv_save_time(title, header):
    # check save path
    if data_set == 'google_cluster':
        save_path_head = save_path_head_google
    elif data_set == 'ali_cluster':
        save_path_head = save_path_head_ali
    else:
        save_path_head = save_path_head_normal

    header = [''] + header

    file = open(save_path_head + title + '相关测试运行时间' + save_path_tail_csv, 'w', newline='')
    rows = []
    for alg in algorithms:
        rows.append([alg] + total_time[alg])
    f_csv = csv.writer(file)
    f_csv.writerow(header)
    f_csv.writerows(rows)
    file.close()


# set_counts(2, 3, 3, 2)
# set_counts(20, 5, 10, 3)
# set_counts(100, 50, 100, 30)


"""
任务密度相关测试参数和函数
"""

test_task_density_ratio_normal = [0.04, 0.2, 1, 5, 25, 125]
test_task_density_ratio_google = [0.0000000001, 0.000000001, 0.0000001, 0.0000001, 0.000001]
test_task_density_ratio_ali = [0.000000000001, 0.00000000001, 0.000000001, 0.000000001, 0.00000001]


def task_about_task_density(data_set=None):
    global tasks_copy

    # set test_task_density
    if data_set == 'ali_cluster':
        test_task_density_ratio = test_task_density_ratio_ali
    elif data_set == 'google_cluster':
        test_task_density_ratio = test_task_density_ratio_google
    else:
        test_task_density_ratio = test_task_density_ratio_normal

    result_clear()
    # initialize
    set_counts_clear(50, 20, 20, 10)
    rand_init_params(data_set)
    tasks_copy1 = copy.deepcopy(tasks_copy)

    for ratio in test_task_density_ratio:
        # set density
        tasks_copy = copy.deepcopy(tasks_copy1)
        for task in tasks_copy:
            task.time = task.time / ratio
        # start test
        print('density:', ratio)
        for alg in algorithms:
            if alg == 'GENETIC':
                continue
            print(alg, 'start...')
            succ, revenue = alg_dict[alg]()
            succ_count[alg].append(succ)
            total_revenue[alg].append(revenue)
            print(alg, 'finish!!!')
        print()

    if 'GENETIC' in algorithms:
        print('GENETIC start...')
        succ, revenue = genetic()
        for density in test_task_density_ratio:
            succ_count['GENETIC'].append(succ)
            total_revenue['GENETIC'].append(revenue)
        print('GENETIC finish!!!')

    plt_plot('task density', [round(np.log(elem) / np.log(5), 2) for elem in test_task_density_ratio],
             '5^x relative density')


"""
任务分布相关测试参数和函数
"""

test_task_distribution = ['uniform', 'normal']


def task_about_task_distribution(data_set=None):
    global tasks_copy

    result_clear()
    # initialize
    tasks_count = 50
    set_counts_clear(tasks_count, 10, 7, 5)
    rand_init_params(data_set)
    tasks_copy1 = copy.deepcopy(tasks_copy)
    max_time = 100

    for distribution in test_task_distribution:
        # set distribution
        tasks_copy = copy.deepcopy(tasks_copy1)
        times = None
        if distribution == 'uniform':  # uniform distribution
            times = stats.uniform.rvs(loc=0, scale=max_time, size=tasks_count)
        elif distribution == 'normal':  # normal distribution
            times = stats.truncnorm.rvs(-1, 1, size=tasks_count)
            times = (times + 1) * max_time / 2
        time_list = times.tolist()
        time_list.sort()
        for i in range(50):
            tasks[i].time = time_list[i]
        # start test
        print('distribution:', distribution)
        for alg in algorithms:
            if alg == 'GENETIC':
                continue
            print(alg, 'start...')
            succ, revenue = alg_dict[alg]()
            succ_count[alg].append(succ)
            total_revenue[alg].append(revenue)
            print(alg, 'finish!!!')
        print()

    '''
    print('GENETIC start...')
    succ, revenue = genetic()
    for distribution in test_task_distribution:
        succ_count['GENETIC'].append(succ)
        total_revenue['GENETIC'].append(revenue)
    print('GENETIC finish!!!')
    '''
    plt_plot('task distribution', test_task_distribution)


"""
最大CPU频率和最大传输速率测试参数和函数
"""

test_f_t_ratio = [0.125, 0.25, 0.5, 1, 2, 4]


def test_about_f_t(data_set=None):
    global svs_copy
    global dss_copy

    result_clear()
    # initialize
    set_counts_clear(50, 10, 7, 5)
    rand_init_params(data_set)
    svs_copy1 = copy.deepcopy(svs_copy)
    dss_copy1 = copy.deepcopy(dss_copy)

    for ratio in test_f_t_ratio:
        # set density
        svs_copy = copy.deepcopy(svs_copy1)
        dss_copy = copy.deepcopy(dss_copy1)
        for sv in svs_copy:
            sv.f_max = sv.f_max * ratio
            sv.f_upper = sv.f_max
        for ds in dss_copy:
            ds.t_max = ds.t_max * ratio
            ds.t_upper = ds.t_max
        # start test
        print('max f and t:', ratio)
        for alg in algorithms:
            print(alg, 'start...')
            succ, revenue = alg_dict[alg]()
            succ_count[alg].append(succ)
            total_revenue[alg].append(revenue)
            print(alg, 'finish!!!')
        print()

    plt_plot('max f and t', [np.log2(elem) for elem in test_f_t_ratio], '2^x relative f and t')


"""
最大CPU频率测试参数和函数
"""

test_f_ratio = [0.25, 0.5, 1, 2, 4]


def test_about_f(data_set=None):
    global svs_copy

    result_clear()
    # initialize
    set_counts_clear(50, 10, 7, 5)
    rand_init_params(data_set)
    for ds in dss_copy:
        ds.t_max = np.inf
    svs_copy1 = copy.deepcopy(svs_copy)

    for ratio in test_f_ratio:
        # set density
        svs_copy = copy.deepcopy(svs_copy1)
        for sv in svs_copy:
            sv.f_max = sv.f_max * ratio
            sv.f_upper = sv.f_max
        # start test
        print('max f:', ratio)
        for alg in algorithms:
            print(alg, 'start...')
            succ, revenue = alg_dict[alg]()
            succ_count[alg].append(succ)
            total_revenue[alg].append(revenue)
            print(alg, 'finish!!!')
        print()

    plt_plot('max f', [np.log2(elem) for elem in test_f_t_ratio], '2^x relative f')


"""
服务器数量测试参数和函数
"""

test_sv_counts = [1, 2, 3, 4, 5, 6]


def test_about_sv_counts(data_set=None):
    global svs_copy
    global sv_count
    global f_max_copy

    result_clear()
    # initialize
    set_counts_clear(50, test_sv_counts[-1], 100, 10)
    rand_init_params(data_set=data_set, priority=1)
    for ds in dss_copy:
        ds.t_max = np.inf
    svs_copy1 = copy.deepcopy(svs_copy)
    f_max_copy1 = copy.deepcopy(f_max_copy)

    for test_sv_count in test_sv_counts:
        # set server count
        sv_count = test_sv_count
        svs_copy = copy.deepcopy(svs_copy1[0:test_sv_count])
        f_max_copy = copy.deepcopy(f_max_copy1[0:test_sv_count])
        # start test
        print('server count:', test_sv_count)
        for alg in algorithms:
            print(alg, 'start...')
            succ, revenue = alg_dict[alg]()
            succ_count[alg].append(succ)
            print(alg, ':', revenue)
            total_revenue[alg].append(revenue)
            print(alg, 'finish!!!')
        print()
    plt_plot('server count', test_sv_counts)


"""
度测试参数和函数
"""

test_degree = [5, 8, 11, 14, 17, 20]


def test_about_degree(data_set=None):
    result_clear()
    set_counts_clear(50, 20, 20, test_degree[0])
    rand_init_params(data_set)

    ds_list = range(1, ds_count + 1)
    for i in range(len(test_degree)):
        # update degrees of sv
        if i != 0:
            for j in range(len(svs_copy)):
                connect = svs_copy[j].connect
                disconnect = [elem for elem in ds_list if elem not in connect]
                connect += random.sample(disconnect, test_degree[i] - test_degree[i - 1])
                connect.sort()
                svs_copy[j].connect = connect
                connects[j] = [j + 1, connect]
        # start test
        print('degree:', test_degree[i])
        for alg in algorithms:
            succ, revenue = alg_dict[alg]()
            succ_count[alg].append(succ)
            total_revenue[alg].append(revenue)
            print(alg, 'finish!!!')
            if alg != 'GENETIC' and total_revenue['DOTA'][i] < total_revenue[alg][i]:
                print('FAIL')
                break
        print()

    csv_save('degree', test_degree)


"""
运行时间测试参数和函数
"""

test_task_counts = [10, 28, 46, 64, 82, 100]


def test_about_run_time(data_set=None):
    global test_time

    test_time = 1
    result_clear()
    for task_num in test_task_counts:
        set_counts_clear(task_num, 20, 15, 10)
        rand_init_params(data_set)
        print('task count:', task_num)
        for alg in algorithms:
            alg_dict[alg]()
            total_time[alg].append(run_time)
            print(alg, 'finish!!!')
        print()
    test_time = 0
    csv_save_time('task count', test_task_counts)
    plt_plot_time('task count', test_task_counts)


"""
利用率测试参数和函数
"""

test_util_task_counts = 100


def test_about_util_rate(data_set=None):
    global test_util

    test_util = 1
    task_counts_list = list(range(1, test_util_task_counts + 1))
    result_clear()
    set_counts_clear(test_util_task_counts, 20, 15, 10)
    rand_init_params(data_set)
    for alg in algorithms:
        if alg != 'GENETIC':
            alg_dict[alg]()
    plt_plot_util(task_counts_list)
    test_util = 0


# data_set = 'ali_cluster'
# data_set = 'google_cluster'
data_set = None

# test_about_task_counts()
# task_about_task_density(data_set)     # 任务密度测试
# task_about_task_distribution(data_set)
# test_about_f_t(data_set)  # 最大总频率和传输速率测试
# test_about_f(data_set)
# test_about_sv_counts(data_set)    # 服务器数量测试
test_about_degree(data_set)  # 服务器的度（连接数）测试
# test_about_run_time(data_set)   # 运行时间测试
# # test_about_util_rate(dat_set)   # 资源利用率测试
