import numpy as np
import geatpy as ea
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
import time

# from ConvPE_Modeling import ConvPE_Modeling
# from BCMPU_modeling import BCMPU_modeling
# from Seach_Space import C2D_SEARCH, BCM_SEARCH
from analytical_model.ConvPE_Modeling import ConvPE_Modeling
from analytical_model.BCMPU_modeling import BCMPU_modeling
from analytical_model.Seach_Space import C2D_SEARCH, BCM_SEARCH


class GAProblem_C2D(ea.Problem):
    def __init__(self, DSP, BRAM, BW, layers):
        name = 'Net Mapping'
        M = 1                       # the target aim dimmension, here signle aim
        maxormins = [1]             # 1:minimize the target, -1 ,maximize
        Dim = 7                     # [tr, tm, tn, in_bw_id, wt_bw_id, out_bw_id, bn_bw_id] tc = tr
        varTypes = [1] * Dim        # 0:continuous,1:Discrete
        lb = [0] * Dim
        ub = [3, 4, 9, 6, 6, 5, 5]   # index (num of selection)
        lbin = [1] * Dim            # 0 not include the bound,1 include the bound
        ubin = [0] * Dim            # index range:0~ub-1
        self.DSP = DSP
        self.BRAM = BRAM
        self.BW = BW
        self.layers = layers
        cpu_cores = int(mp.cpu_count() - 4)     # cores of the cpu
        self.pool = ProcessPool(cpu_cores)      # pool number

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen
        N = Vars.shape[0]
        args = list(
            zip(Vars, N * [self.layers])
        )
        # only one positive parameter
        result = self.pool.map_async(evaluate_C2D, args)
        result.wait()
        re = np.array(result.get())
        pop.ObjV = re[:, [3]] + re[:, [2]]      # aim function: total latency + bw
        pop.CV = np.hstack([                    # constraint
            re[:, [0]] - self.DSP,
            re[:, [1]] - self.BRAM,
            re[:, [2]] - self.BW]
        )


def evaluate_C2D(args):
    hw_params = args[0]
    layers = args[1]
    tr_id, tm_id, tn_id, in_bw_id, wt_bw_id, out_bw_id, bn_bw_id = hw_params[0:7]
    tr, tm, tn = C2D_SEARCH['tr'][tr_id], C2D_SEARCH['tm'][tm_id], C2D_SEARCH['tn'][tn_id]
    in_bw, wt_bw = C2D_SEARCH['in_bw'][in_bw_id], C2D_SEARCH['wt_bw'][wt_bw_id]
    out_bw, bn_bw = C2D_SEARCH['out_bw'][out_bw_id], C2D_SEARCH['bn_bw'][bn_bw_id]
    bram, dsp, bw, layers_lat = ConvPE_Modeling([tr, tr, tm, tn], layers, [in_bw, wt_bw, out_bw, bn_bw])
    lat = np.sum(layers_lat)
    total_bw = in_bw + wt_bw + out_bw * 2 + bn_bw
    return dsp, bram, total_bw, lat


def GA_Engine_C2D(layers, DSP, BRAM, BW, NIND=100, MAX_GEN=20):
    problem = GAProblem_C2D(DSP, BRAM, BW, layers)

    Encoding = 'RI'
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)   # inst the population

    myAlgorithm = ea.soea_SEGA_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_GEN            # Maximum Generation
    myAlgorithm.mutOper.F = 0.5             # Parameter F in differential evolution
    myAlgorithm.recOper.XOVR = 0.6          # Reorganization probability
    myAlgorithm.mutOper.Pm = 0.7            # Variation probability
    myAlgorithm.drawing = 0
    myAlgorithm.logTras = 0                 # 20 # 0: no log info, >0: interval gerneration one log
    myAlgorithm.verbose = True              # True # # print log
    [BestIndi, population] = myAlgorithm.run()

    problem.pool.close()
    problem.pool.join()
    # print('Comsuming %s s' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        # print('latency %s' % BestIndi.ObjV[0][0])
        length = BestIndi.Phen.shape[1]
        block_param = BestIndi.Phen[0, 0:length]
        # print(block_param)
        return block_param, BestIndi.ObjV[0][0]
    else:
        raise Exception('NO feasiable Result')


class GAProblem_BCM(ea.Problem):
    def __init__(self, DSP, BRAM, BW, layers):
        name = 'Net Mapping'
        M = 1                       # the target aim dimmension, here signle aim
        maxormins = [1]             # 1:minimize the target, -1 ,maximize
        Dim = 6                     # [tr, tm, tn, in_bw_id, wt_bw_id, out_bw_id]  tc = tr
        varTypes = [1] * Dim        # 0:continuous,1:Discrete
        lb = [0] * Dim
        ub = [3, 4, 8, 5, 7, 5]     # index (num of selection)
        lbin = [1] * Dim            # 0 not include the bound,1 include the bound
        ubin = [0] * Dim            # index range:0~ub-1
        self.DSP = DSP
        self.BRAM = BRAM
        self.BW = BW
        self.layers = layers
        cpu_cores = int(mp.cpu_count() - 4)     # cores of the cpu
        self.pool = ProcessPool(cpu_cores)      # pool number

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen
        N = Vars.shape[0]
        args = list(
            zip(Vars, N * [self.layers])
        )
        # only one positive parameter
        result = self.pool.map_async(evaluate_BCM, args)
        result.wait()
        re = np.array(result.get())
        # result = map(evaluate_BCM, args)
        # re = np.array(list(result))
        pop.ObjV = re[:, [3]] + re[:, [2]]      # aim function: total latency + bw
        pop.CV = np.hstack([                    # constraint
            re[:, [0]] - self.DSP,
            re[:, [1]] - self.BRAM,
            re[:, [2]] - self.BW]
        )


def evaluate_BCM(args):
    hw_params = args[0]
    layers = args[1]
    tr_id, tm_id, tn_id, in_bw_id, wt_bw_id, out_bw_id= hw_params[0:6]
    tr, tm, tn = BCM_SEARCH['tr'][tr_id], BCM_SEARCH['tm'][tm_id], BCM_SEARCH['tn'][tn_id]
    in_bw, wt_bw, out_bw = BCM_SEARCH['in_bw'][in_bw_id], BCM_SEARCH['wt_bw'][wt_bw_id], BCM_SEARCH['out_bw'][out_bw_id]
    (dsp, bram), layers_lat = BCMPU_modeling(layers, [tr, tr, tm, tn, in_bw, wt_bw, out_bw])
    lat = sum(layers_lat)
    total_bw = in_bw + wt_bw + out_bw
    return dsp, bram, total_bw, lat


def GA_Engine_BCM(layers, DSP, BRAM, BW, NIND=100, MAX_GEN=20):
    problem = GAProblem_BCM(DSP, BRAM, BW, layers)

    Encoding = 'RI'
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)  # inst the population

    myAlgorithm = ea.soea_SEGA_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_GEN        # Maximum Generation
    myAlgorithm.mutOper.F = 0.5         # Parameter F in differential evolution
    myAlgorithm.recOper.XOVR = 0.6      # Reorganization probability
    myAlgorithm.mutOper.Pm = 0.7        # Variation probability
    myAlgorithm.drawing = 0
    myAlgorithm.logTras = 5             # 20 # 0: no log info, >0: interval gerneration one log
    myAlgorithm.verbose = True          # True # # print log
    [BestIndi, population] = myAlgorithm.run()

    problem.pool.close()
    problem.pool.join()
    # print('Comsuming %s s' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        # print('latency %s' % BestIndi.ObjV[0][0])
        length = BestIndi.Phen.shape[1]
        block_param = BestIndi.Phen[0, 0:length]
        # print(block_param)
        return block_param, BestIndi.ObjV[0][0]
    else:
        raise Exception('NO feasiable Result')


class GAProblem_All(ea.Problem):
    def __init__(self, DSP, BRAM, BW, C2D_layers, BCM_layers):
        name = 'Net Mapping'
        M = 1                       # the target aim dimmension, here signle aim
        maxormins = [1]             # 1:minimize the target, -1 ,maximize
        Dim = 13                    # ConvPE + BCMPU params (7 + 6)  tr = tc
        varTypes = [1] * Dim        # 0:continuous,1:Discrete
        lb = [0] * Dim
        ub = [3, 8, 9, 6, 6, 6, 5, 3, 5, 6, 7, 6, 6]    # index (num of selection)
        lbin = [1] * Dim            # 0 not include the bound,1 include the bound
        ubin = [0] * Dim            # index range:0~ub-1
        self.DSP = DSP
        self.BRAM = BRAM
        self.BW = BW
        self.C2D_layers = C2D_layers
        self.BCM_layers = BCM_layers
        cpu_cores = int(mp.cpu_count() - 4)     # cores of the cpu
        self.pool = ProcessPool(cpu_cores)      # pool number

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen
        N = Vars.shape[0]
        args = list(
            zip(Vars, N * [self.C2D_layers], N * [self.BCM_layers])
        )
        # # only one positive parameter
        result = self.pool.map_async(evaluate_All, args)
        result.wait()
        re = np.array(result.get())
        pop.ObjV = re[:, [3]] + 100*re[:, [2]]      # aim function: total latency + bw
        pop.CV = np.hstack([                    # constraint
            re[:, [0]] - self.DSP,
            re[:, [1]] - self.BRAM,
            re[:, [2]] - self.BW]
        )


def evaluate_All(args):
    hw_params = args[0]
    C2D_layers = args[1]
    BCM_layers = args[2]

    C2D_bram, C2D_dsp, C2D_bw, C2D_layers_lat = 0, 0, 0, 0
    in_bw, wt_bw, out_bw, bn_bw = 0, 0, 0, 0
    BCM_dsp, BCM_bram, BCM_layers_lat = 0, 0, 0
    in_bw_b, wt_bw_b, out_bw_b = 0, 0, 0

    if len(C2D_layers):
        tr_id, tm_id, tn_id, in_bw_id, wt_bw_id, out_bw_id, bn_bw_id = hw_params[0:7]
        tr, tm, tn = C2D_SEARCH['tr'][tr_id], C2D_SEARCH['tm'][tm_id], C2D_SEARCH['tn'][tn_id]
        in_bw, wt_bw = C2D_SEARCH['in_bw'][in_bw_id], C2D_SEARCH['wt_bw'][wt_bw_id]
        out_bw, bn_bw = C2D_SEARCH['out_bw'][out_bw_id], C2D_SEARCH['bn_bw'][bn_bw_id]
        C2D_bram, C2D_dsp, C2D_bw, C2D_layers_lat = ConvPE_Modeling([tr, tr, tm, tn], np.array(C2D_layers), [in_bw, wt_bw, out_bw, bn_bw])

    if len(BCM_layers):
        tr_b_id, tm_b_id, tn_b_id, in_bw_b_id, wt_bw_b_id, out_bw_b_id = hw_params[7:13]
        tr_b, tm_b, tn_b = BCM_SEARCH['tr'][tr_b_id], BCM_SEARCH['tm'][tm_b_id], BCM_SEARCH['tn'][tn_b_id]
        in_bw_b, wt_bw_b, out_bw_b = BCM_SEARCH['in_bw'][in_bw_b_id], BCM_SEARCH['wt_bw'][wt_bw_b_id], BCM_SEARCH['out_bw'][out_bw_b_id]
        (BCM_dsp, BCM_bram), BCM_layers_lat = BCMPU_modeling(np.array(BCM_layers), [tr_b, tr_b, tm_b, tn_b, in_bw_b, wt_bw_b, out_bw_b])

    dsp = C2D_dsp + BCM_dsp
    bram = C2D_bram + BCM_bram
    total_bw = in_bw + wt_bw + max(out_bw*2 , bn_bw) + in_bw_b + wt_bw_b + out_bw_b #TODO change branch-add load method
    lat = max(np.sum(C2D_layers_lat),  np.sum(BCM_layers_lat) )
    return dsp, bram, total_bw, lat


def GA_Engine_All(C2D_layers, BCM_layers, DSP, BRAM, BW, NIND=200, MAX_GEN=100):
    problem = GAProblem_All(DSP, BRAM, BW, C2D_layers, BCM_layers)

    Encoding = 'RI'
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)  # inst the population

    myAlgorithm = ea.soea_SEGA_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_GEN        # Maximum Generation
    myAlgorithm.mutOper.F = 0.5         # Parameter F in differential evolution
    myAlgorithm.recOper.XOVR = 0.6      # Reorganization probability
    myAlgorithm.mutOper.Pm = 0.7        # Variation probability
    myAlgorithm.drawing = 0
    myAlgorithm.logTras = 0             # 20 # 0: no log info, >0: interval gerneration one log
    myAlgorithm.verbose = True          # True # # print log
    [BestIndi, population] = myAlgorithm.run()

    problem.pool.close()
    problem.pool.join()
    # print('Comsuming %s s' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        # print('latency %s' % BestIndi.ObjV[0][0])
        length = BestIndi.Phen.shape[1]
        block_param = BestIndi.Phen[0, 0:length]
        # print(block_param)
        return block_param, BestIndi.ObjV[0][0]
    else:
        raise Exception('NO feasiable Result')



if __name__ == "__main__":
    alloc_layers = np.array([
        # conv2_x * 3
        [64, 64, 56, 56, 3, 1, 1, 1],
        [64, 64, 56, 56, 3, 1, 1, 1],
        [64, 64, 56, 56, 3, 1, 1, 3],
        [64, 64, 56, 56, 3, 1, 1, 3],
        [64, 64, 56, 56, 3, 1, 1, 1],
        [64, 64, 56, 56, 3, 1, 1, 0],
        # conv3_x * 4
        [64, 128, 56, 56, 3, 2, 1, 3],
        [128, 128, 28, 28, 3, 1, 1, 3],
        [128, 128, 28, 28, 3, 1, 1, 3],
        [128, 128, 28, 28, 3, 1, 1, 2],
        [128, 128, 28, 28, 3, 1, 1, 2],
        [128, 128, 28, 28, 3, 1, 1, 1],
        [128, 128, 28, 28, 3, 1, 1, 1],
        [128, 128, 28, 28, 3, 1, 1, 0],
        # conv4_x * 6
        [128, 256, 28, 28, 3, 2, 1, 3],
        [256, 256, 14, 14, 3, 1, 1, 3],
        [256, 256, 14, 14, 3, 1, 1, 3],
        [256, 256, 14, 14, 3, 1, 1, 3],
        [256, 256, 14, 14, 3, 1, 1, 2],
        [256, 256, 14, 14, 3, 1, 1, 2],
        [256, 256, 14, 14, 3, 1, 1, 2],
        [256, 256, 14, 14, 3, 1, 1, 1],
        [256, 256, 14, 14, 3, 1, 1, 1],
        [256, 256, 14, 14, 3, 1, 1, 1],
        [256, 256, 14, 14, 3, 1, 1, 0],
        [256, 256, 14, 14, 3, 1, 1, 0],
        # conv5_x * 3
        [256, 512, 14, 14, 3, 2, 1, 3],
        [512, 512, 7, 7, 3, 1, 1, 2],
        [512, 512, 7, 7, 3, 1, 1, 1],
        [512, 512, 7, 7, 3, 1, 1, 1],
        [512, 512, 7, 7, 3, 1, 1, 1],
        [512, 512, 7, 7, 3, 1, 1, 0],
    ])


    t1 = time.time()
    tile_param, lat = GA_Engine_C2D(alloc_layers, 500, 200, 64, NIND=200, MAX_GEN=50)
    t2 = time.time()
    print("GA time: %s ms" % ((t2 - t1) * 1000))

    tr_id, tm_id, tn_id, in_bw_id, wt_bw_id, out_bw_id, bn_bw_id = tile_param[0:7]
    tr, tm, tn = C2D_SEARCH['tr'][tr_id], C2D_SEARCH['tm'][tm_id], C2D_SEARCH['tn'][tn_id]
    in_bw, wt_bw = C2D_SEARCH['in_bw'][in_bw_id], C2D_SEARCH['wt_bw'][wt_bw_id]
    out_bw, bn_bw = C2D_SEARCH['out_bw'][out_bw_id], C2D_SEARCH['bn_bw'][bn_bw_id]
    print("tr: {}, tc: {}, tm: {}, tn: {}".format(tr, tr, tm, tn))
    print("in_bw: {}, wt_bw: {}, out_bw: {}, bn_bw: {}".format(in_bw, wt_bw, out_bw, bn_bw))
    print(lat)
