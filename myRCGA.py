from sko.GA import RCGA
import numpy as np
import random
from sko.operators import ranking, selection

"""
定义实数变异方式
"""


def mutation_base(y):
    y_low = 0
    y_up = 1
    delta1 = 1.0 * (y - y_low) / (y_up - y_low)
    delta2 = 1.0 * (y_up - y) / (y_up - y_low)
    r = np.random.random()
    mut_pow = 1.0 / (1 + 1.0)
    if r <= 0.5:
        xy = 1.0 - delta1
        val = 2.0 * r + (1.0 - 2.0 * r) * (xy ** (1 + 1.0))
        delta_q = val ** mut_pow - 1.0
    else:
        xy = 1.0 - delta2
        val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy ** (1 + 1.0))
        delta_q = 1.0 - val ** mut_pow
    y = y + delta_q * (y_up - y_low)
    y = min(y_up, max(y, y_low))
    return y


class myRCGA(RCGA):
    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001,
                 prob_cros=0.9,
                 lb=-1, ub=1,
                 task_count=0,
                 sv_count=0,
                 ds_count=0,
                 xil=None,
                 f_max=0,
                 t_max=0,
                 ):
        self.count = 0

        self.task_count = task_count
        self.sv_count = sv_count
        self.ds_count = ds_count
        self.xil = np.array(xil)
        self.f_max = np.array([f_max])
        self.t_max = np.array([t_max])

        self.x_count = task_count * sv_count
        self.f_count = task_count * sv_count
        self.t_count = task_count * ds_count
        self.x_start = 0
        self.x_end = self.f_start = self.x_count
        self.f_end = self.t_start = self.f_start + self.f_count
        self.t_end = self.t_start + self.t_count

        self.f_lower = np.array(lb[self.f_start: self.f_end: self.task_count])
        self.f_upper = np.array(ub[self.f_start: self.f_end: self.task_count])
        self.t_lower = np.array(lb[self.t_start: self.t_end: self.task_count])
        self.t_upper = np.array(ub[self.t_start: self.t_end: self.task_count])
        self.fj_count_max = (self.f_max / self.f_lower).flatten().astype(int)
        self.tl_count_max = (self.t_max / self.t_lower).flatten().astype(int)

        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, prob_cros, lb, ub)

    def crtbp(self):
        # create test samples
        self.Chrom = np.zeros([self.size_pop, self.n_dim])
        for size in range(self.size_pop):
            xij = np.zeros([self.task_count, self.sv_count])
            fij = np.zeros([self.task_count, self.sv_count])
            til = np.zeros([self.task_count, self.ds_count])
            # constraints xij: xi = 0
            for i in range(self.task_count):
                rand_int = np.random.randint(self.sv_count)
                xij[i, rand_int] = 1
                fij[i, rand_int] = random.uniform(self.f_lower[rand_int], self.f_upper[rand_int])
                xl_list = [k for k in range(self.ds_count) if self.xil[i].tolist()[k] == 1]
                for k in xl_list:
                    til[i, k] = random.uniform(self.t_lower[k], self.t_upper[k])
            # create population from samples
            chrom1 = xij.reshape(self.task_count * self.sv_count)
            chrom2 = (fij - self.f_lower) / (self.f_upper - self.f_lower)
            chrom2 = np.where(chrom2 < 0, 0, chrom2)
            chrom2 = chrom2.reshape(self.task_count * self.sv_count)
            chrom3 = (til - self.t_lower) / (self.t_upper - self.t_lower)
            chrom3 = np.where(chrom3 < 0, 0, chrom3)
            chrom3 = chrom3.reshape(self.task_count * self.ds_count)
            self.Chrom[size] = np.concatenate([chrom1, chrom2, chrom3])
        # adjust Chrom by constraints
        self.adjust_by_constraints()
        return self.Chrom

    def crossover(self):
        self.count += 1
        print('crossover:', self.count)
        for a in range(0, self.size_pop, 2):
            chrom1 = self.Chrom[a]
            chrom2 = self.Chrom[a + 1]
            xij1 = chrom1[self.x_start:self.x_end].reshape([self.task_count, self.sv_count])
            fij1 = chrom1[self.f_start:self.f_end].reshape([self.task_count, self.sv_count])
            til1 = chrom1[self.t_start:self.t_end].reshape([self.task_count, self.ds_count])
            xij2 = chrom2[self.x_start:self.x_end].reshape([self.task_count, self.sv_count])
            fij2 = chrom2[self.f_start:self.f_end].reshape([self.task_count, self.sv_count])
            til2 = chrom2[self.t_start:self.t_end].reshape([self.task_count, self.ds_count])
            # for x and f, do 1 point crossover;
            # for t, do simulated binary crossover
            n = np.random.randint(0, self.task_count)
            x_seg1, x_seg2 = xij1[n:].copy(), xij2[n:].copy()
            xij1[n:], xij2[n:] = x_seg2, x_seg1
            f_seg1, f_seg2 = fij1[n:].copy(), fij2[n:].copy()
            fij1[n:], fij2[n:] = f_seg2, f_seg1
            for i in range(n + 1, self.task_count):
                for k in range(n + 1, self.ds_count):
                    if np.sum(xij1[i]) == 1 and np.sum(xij2[i]) == 1:
                        if np.random.random() > self.prob_cros:
                            continue
                        y_low = 0
                        y_up = 1
                        y1 = til1[i][k]
                        y2 = til2[i][k]
                        r = np.random.random()
                        if r <= 0.5:
                            beta_q = (2 * r) ** (1.0 / (1 + 1.0))
                        else:
                            beta_q = (0.5 / (1.0 - r)) ** (1.0 / (1 + 1.0))
                        child1 = 0.5 * ((1 + beta_q) * y1 + (1 - beta_q) * y2)
                        child2 = 0.5 * ((1 - beta_q) * y1 + (1 + beta_q) * y2)
                        child1 = min(max(child1, y_low), y_up)
                        child2 = min(max(child2, y_low), y_up)
                        til1[i][k] = child1
                        til2[i][k] = child2
                    else:
                        tmp = til1[i, k]
                        til1[i, k] = til2[i, k]
                        til2[i, k] = tmp
            chrom11 = xij1.reshape(self.task_count * self.sv_count)
            chrom12 = fij1.reshape(self.task_count * self.sv_count)
            chrom13 = til1.reshape(self.task_count * self.ds_count)
            chrom21 = xij2.reshape(self.task_count * self.sv_count)
            chrom22 = fij2.reshape(self.task_count * self.sv_count)
            chrom23 = til2.reshape(self.task_count * self.ds_count)
            self.Chrom[a] = np.concatenate([chrom11, chrom12, chrom13])
            self.Chrom[a + 1] = np.concatenate([chrom21, chrom22, chrom23])

    def mutation(self):
        print('mutation:', self.count)
        for a in range(self.size_pop):
            xij = self.Chrom[a, self.x_start:self.x_end].reshape([self.task_count, self.sv_count])
            fij = self.Chrom[a, self.f_start:self.f_end].reshape([self.task_count, self.sv_count])
            til = self.Chrom[a, self.t_start:self.t_end].reshape([self.task_count, self.ds_count])
            # x mutation
            for i in range(self.task_count):
                if np.random.random() > self.prob_mut:
                    continue
                xi_sum = np.sum(xij[i])
                if xi_sum == 1:
                    n = np.random.randint(0, self.sv_count + 1)
                    while n != self.sv_count and xij[i, n] == 1:
                        n = np.random.randint(0, self.sv_count + 1)
                    xij[i] = np.zeros(self.sv_count)
                    fij[i] = np.zeros(self.sv_count)
                    if n != self.sv_count:
                        xij[i, n] = 1
                        fij[i, n] = np.random.random()
                    else:
                        til[i] = np.zeros(self.ds_count)
                else:
                    n = np.random.randint(0, self.sv_count)
                    xij[i, n] = 1
                    fij[i, n] = np.random.random()
                    for k in range(self.ds_count):
                        if self.xil[i, k] == 1:
                            til[i, k] = np.random.random()
            # f and t mutation
            for i in range(self.task_count):
                for j in range(self.sv_count):
                    r = np.random.random()
                    if fij[i, j] == 0 or r > self.prob_mut:
                        continue
                    fij[i, j] = mutation_base(fij[i, j])
                for k in range(self.ds_count):
                    r = np.random.random()
                    if til[i, k] == 0 or r > self.prob_mut:
                        continue
                    til[i, k] = mutation_base(til[i, k])
            chrom1 = xij.reshape(self.task_count * self.sv_count)
            chrom2 = fij.reshape(self.task_count * self.sv_count)
            chrom3 = til.reshape(self.task_count * self.ds_count)
            self.Chrom[a] = np.concatenate([chrom1, chrom2, chrom3])
        # adjust Chrom by constraints
        self.adjust_by_constraints()

    def adjust_by_constraints(self):
        for a in range(self.size_pop):
            xij = self.Chrom[a, self.x_start:self.x_end].reshape([self.task_count, self.sv_count])
            fij = self.Chrom[a, self.f_start:self.f_end].reshape([self.task_count, self.sv_count])
            til = self.Chrom[a, self.t_start:self.t_end].reshape([self.task_count, self.ds_count])
            for j in range(self.sv_count):
                xj_list = xij[:, j].flatten().tolist()
                xj_list = [i for i in range(len(xj_list)) if xj_list[i] == 1]
                fj_count = len(xj_list)
                # delete tasks if tasks are too many for sv
                fj_delete_count = fj_count - self.fj_count_max[j]
                if fj_delete_count > 0:
                    delete_task = random.sample(xj_list, fj_delete_count)
                    for i in delete_task:
                        xij[i, j] = 0
                        fij[i, j] = 0
                        til[i] = np.zeros(self.ds_count)
                        xj_list.remove(i)
                # re-decide fij
                fj_count = len(xj_list)
                total_provide_f = self.f_max[0, j] - self.f_lower[j] * fj_count
                total_lent_f = np.sum(fij[:, j])
                if total_lent_f == 0:
                    continue
                for i in xj_list:
                    fij[i, j] = total_provide_f * (fij[i, j] / total_lent_f) / (self.f_upper[j] - self.f_lower[j])
                    fij[i, j] = min(fij[i, j], 1)
            for k in range(self.ds_count):
                xl_list = self.xil[:, k].flatten().tolist()
                xl_list = [i for i in range(len(xl_list)) if xl_list[i] == 1]
                tl_count = len(xl_list)
                # delete tasks if tasks are too many for ds
                tl_delete_count = tl_count - self.tl_count_max[k]
                if tl_delete_count > 0:
                    delete_task = random.sample(xl_list, tl_delete_count)
                    for i in delete_task:
                        xij[i] = np.zeros(self.sv_count)
                        fij[i] = np.zeros(self.sv_count)
                        til[i] = np.zeros(self.ds_count)
                        xl_list.remove(i)
                # re-decide til
                tl_count = len(xl_list)
                total_provide_t = self.t_max[0, k] - self.t_lower[k] * tl_count
                total_lent_t = np.sum(til[:, k])
                if total_lent_t == 0:
                    continue
                for i in xl_list:
                    til[i, k] = total_provide_t * (til[i, k] / total_lent_t) / (self.t_upper[k] - self.t_lower[k])
                    til[i, k] = min(til[i, k], 1)
            chrom1 = xij.reshape(self.task_count * self.sv_count)
            chrom2 = fij.reshape(self.task_count * self.sv_count)
            chrom3 = til.reshape(self.task_count * self.ds_count)
            self.Chrom[a] = np.concatenate([chrom1, chrom2, chrom3])

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover
    mutation = mutation
