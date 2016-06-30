""" this module is used for all other cache replacement algorithms excluding LRU(LRU also works, but slow compared to
    using pardaProfiler),
"""
# -*- coding: utf-8 -*-


import math
import os
import time

import traceback

import numpy as np
from multiprocessing import Process, Pipe, Array
from os import path

from mimircache.cacheReader.vscsiReader import vscsiCacheReader

from mimircache.cache.adaptiveSLRU import AdaptiveSLRU
from mimircache.cache.clock import clock
from mimircache.cache.FIFO import FIFO
from mimircache.cache.LFU_LRU__NEED_OPTIMIZATION import LFU_LRU
from mimircache.cache.LFU_MRU import LFU_MRU
from mimircache.cache.LFU_RR import LFU_RR
from mimircache.cache.LRU import LRU
from mimircache.cache.MRU import MRU
from mimircache.cache.Random import Random
from mimircache.cache.SLRU import SLRU
from mimircache.cache.S4LRU import S4LRU
from mimircache.cache.Optimal import optimal


from mimircache.cacheReader.plainReader import plainCacheReader

import mimircache.c_generalProfiler as c_generalProfiler

import matplotlib.pyplot as plt

from mimircache.profiler.abstract.abstractProfiler import profilerAbstract

from mimircache.const import *


class generalProfiler(profilerAbstract):
    def __init__(self, reader, cache_class, cache_size, bin_size=-1, cache_params=None,
                 num_of_process=DEFAULT_NUM_OF_PROCESS):
        if isinstance(cache_class, str):
            cache_class = cache_name_to_class(cache_class)
        super(generalProfiler, self).__init__(cache_class, cache_size, reader)
        self.cache_params = cache_params
        self.num_of_process = num_of_process

        if bin_size == -1:
            self.bin_size = int(self.cache_size / DEFAULT_BIN_NUM_PROFILER)
        else:
            self.bin_size = bin_size
        self.num_of_process = num_of_process

        if self.cache_size != -1:

            self.num_of_cache = self.num_of_blocks = math.ceil(self.cache_size / bin_size)

            self.HRC = np.zeros((self.num_of_blocks + 1,), dtype=np.double)
            self.MRC = np.zeros((self.num_of_blocks + 1,), dtype=np.double)

        else:
            raise RuntimeError("you input -1 as cache size")
        self.cache = None
        self.cache_list = None

        self.process_list = []
        self.cache_distribution = [[] for _ in range(self.num_of_process)]

        # shared memory for storing MRC count
        self.MRC_shared_array = Array('i', range(self.num_of_blocks))
        for i in range(len(self.MRC_shared_array)):
            self.MRC_shared_array[i] = 0

        # dispatch different cache size to different processes, does not include size = 0
        for i in range(self.num_of_blocks):
            self.cache_distribution[i % self.num_of_process].append((i + 1) * bin_size)

            # build pipes for communication between main process and children process
            # the pipe mainly sends element from main process to children
        self.pipe_list = []
        for i in range(self.num_of_process):
            self.pipe_list.append(Pipe())
            p = Process(target=self._addOneTraceElementSingleProcess,
                        args=(self.num_of_process, i, self.cache_class, self.cache_distribution[i],
                              self.cache_params, self.pipe_list[i][1], self.MRC_shared_array))
            self.process_list.append(p)
            p.start()

        self.calculated = False

    all = ["get_hit_count", "get_hit_rate", "get_miss_rate", "plotMRC", "plotHRC"]


    def addOneTraceElement(self, element):
        super().addOneTraceElement(element)

        for i in range(len(self.pipe_list)):
            # print("send out: " + element)
            self.pipe_list[i][0].send(element)

        return

    # noinspection PyMethodMayBeStatic
    def _addOneTraceElementSingleProcess(self, num_of_process, process_num, cache_class, cache_size_list,
                                         cache_args, pipe, MRC_shared_array):
        """

        :param num_of_process:
        :param process_num:
        :param cache_class:
        :param cache_size_list: a list of different cache size dispached to this process
        :param cache_args: the extra argument (besides size) for instantiate a cache class
        :param pipe:            for receiving cache record from main process
        :param MRC_shared_array:       storing MRC count for all cache sizes
        :return:
        """
        cache_list = []
        for i in range(len(cache_size_list)):
            if cache_args:
                cache_list.append(cache_class(cache_size_list[i], **cache_args))
            else:
                cache_list.append(cache_class(cache_size_list[i]))

        elements = pipe.recv()

        # TODO this part should be changed
        while elements[-1] != 'END_1a1a11a_ENDMARKER':
            for i in range(len(cache_list)):
                # print("i = %d"%i)
                # cache_list[i].printCacheLine()
                # print('')
                for element in elements:
                    if not cache_list[i].addElement(element):
                        MRC_shared_array[i * num_of_process + process_num] += 1
            elements = pipe.recv()
            # print(element)
            # print(cache_list)


    def run(self, buffer_size=10000):
        super().run()
        self.reader.reset()
        l = []
        for i in self.reader:
            l.append(i)
            if len(l) == buffer_size:
                self.add_elements(l)
                l.clear()
                # self.addOneTraceElement(i)
        # p.printMRC()
        if len(l) > 0:
            self.add_elements(l)
        self.calculate()
        # self.outputHRC()
        # self.plotHRC()

    def add_elements(self, elements):
        for element in elements:
            super().addOneTraceElement(element)

        for i in range(len(self.pipe_list)):
            # print("send out: " + element)
            self.pipe_list[i][0].send(elements)

        return

    def calculate(self):
        self.calculated = True
        for i in range(len(self.pipe_list)):
            self.pipe_list[i][0].send(["END_1a1a11a_ENDMARKER"])
            self.pipe_list[i][0].close()
        for i in range(len(self.process_list)):
            self.process_list[i].join()

        self.MRC[0] = 1
        for i in range(1, len(self.MRC), 1):
            # print(self.MRC_shared_array[i])
            self.MRC[i] = self.MRC_shared_array[i - 1] / self.num_of_trace_elements
        for i in range(1, len(self.HRC), 1):
            self.HRC[i] = 1 - self.MRC[i]

    def get_hit_count(self):
        if not self.calculated:
            self.run()

        HC = np.zeros((self.num_of_blocks + 1,), dtype=np.longlong)
        for i in range(1, len(HC), 1):
            HC[i] = self.num_of_trace_elements - self.MRC_shared_array[i - 1]

        return HC

    def get_hit_rate(self):
        if not self.calculated:
            self.run()
        return self.HRC

    def get_miss_rate(self):
        if not self.calculated:
            self.run()
        return self.MRC

    def plotMRC(self, figname="MRC.png", **kwargs):
        if not self.calculated:
            self.calculate()
        try:
            num_of_blocks = self.num_of_blocks + 1
            plt.plot(range(0, self.bin_size * num_of_blocks, self.bin_size), self.MRC[:num_of_blocks])
            plt.xlabel("cache Size")
            plt.ylabel("Miss Rate")
            plt.title('Miss Rate Curve', fontsize=18, color='black')
            plt.savefig(figname)
            plt.show()
            plt.clf()
        except Exception as e:
            plt.savefig(figname)
            print("the plotting function is not wrong, is this a headless server?")
            print(e)
            traceback.print_exc()

    def plotHRC(self, figname="HRC.png", **kwargs):
        if not self.calculated:
            self.calculate()
        try:
            num_of_blocks = self.num_of_blocks + 1

            plt.plot(range(0, self.bin_size * num_of_blocks, self.bin_size), self.HRC[:num_of_blocks])
            plt.xlabel("cache Size")
            plt.ylabel("Hit Rate")
            plt.title('Hit Rate Curve', fontsize=18, color='black')
            plt.savefig(figname)
            plt.show()
            plt.clf()
        except Exception as e:
            plt.savefig(figname)
            print("the plotting function is not wrong, is this a headless server?")
            print(e)




if __name__ == "__main__":
    import time

    t1 = time.time()
    # r = plainCacheReader('../data/test.dat')
    # r = plainCacheReader('../data/parda.trace')
    r = vscsiCacheReader('../data/trace.vscsi')

    arc_dict = {'p': 0.5, 'ghostlist_size': -1}
    # p = generalProfiler(r, ARC, 1000, 100, arc_dict, 8)

    # p = generalProfiler(r, "Random", 3000, 200, num_of_process=8)
    p = generalProfiler(r, "SLRU", 3000, 200, cache_params={"ratio": 1}, num_of_process=8)
    print(p.get_hit_rate())
    print(p.get_hit_count())
    t2 = time.time()
    print("TIME: %f" % (t2 - t1))

    t1 = time.time()
    p.plotMRC()
    # hr = c_generalProfiler.get_hit_rate(r.cReader, 2000, "Optimal", bin_size=200, num_of_threads=8)
    # print(hr)

    t2 = time.time()
    print("TIME: %f" % (t2 - t1))

    # for i in r:
    #     # print(i)
    #     p.addOneTraceElement(i)
    # # p.printMRC()
    # p.plotHRC()