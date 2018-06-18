"""
Hania Kranas, 2017
This is a simple script that can be used for calculating insulation score for given Hi-C map(s) and square size.
"""
import csv
import logging
import numpy as np
import matplotlib
from operator import itemgetter

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

logger = logging.getLogger('IS-calling')
logging.basicConfig(level=logging.INFO)


def save_mat_as_np(chrom, folder, filename):
    filename = chrom.join(filename)
    try:
        hmap = np.loadtxt(folder + filename, delimiter='\t')
    except:
        hmap = np.loadtxt(folder + filename, delimiter='\t', skiprows=1)  # heat files
    np.save(folder + 'mtx-' + chrom + '-' + chrom + '.npy', hmap)


class HiCmap:
    """
    This class is meant to hold one HiCmap and run the Insulation Score analysis on it
    """

    def __init__(self, folder, chromosomes, bin_resolution):
        self.folder = folder
        self.chromosomes = {chrom: HiCChromosome(chrom, self) for chrom in chromosomes}
        self.bin_res = bin_resolution

    def calculate_insulation_all_chr(self, square_size):
        self.calculate_insulation_chrs(self.chromosomes.keys(), square_size)

    def calculate_insulation_chrs(self, chromosomes, square_size):
        chromosomes = set(self.chromosomes.keys()).intersection(set(chromosomes))
        for chrom in chromosomes:
            self.chromosomes[chrom].calculate_IS(square_size)
            self.chromosomes[chrom].normalize_IS(square_size)

    def call_tads_by_IS_all_chr(self, square_size, delta_window_size):
        self.call_tads_by_IS_chrs(self.chromosomes.keys(), square_size, delta_window_size)

    def call_tads_by_IS_chrs(self, chromosomes, square_size, delta_window_size, border_strength=0.1):
        chromosomes = set(self.chromosomes.keys()).intersection(set(chromosomes))
        for chrom in chromosomes:
            self.chromosomes[chrom].get_deltas(square_size, delta_window_size)
            self.chromosomes[chrom].get_IS_minimums(square_size, delta_window_size)
            self.chromosomes[chrom].call_TADs(square_size, border_strength)

    def get_name(self):
        """
        Returns the name of the highest folder that map is kept in.
        :return:
        """
        return "_".join(os.path.basename(self.folder[:-1]).split())


class HiCChromosome:
    """This class holds one chromosome of the given Hi-C map and performs Insulation Score analysis on it"""

    def __init__(self, chrom, parent_map):
        self.chrom = chrom
        self.parent_map = parent_map
        self.loaded = False
        self.map = None
        self.size = None
        self.ISs = []
        self.normalized_ISs = []
        self.deltas = []
        self.border_info = []
        self._load()

    def _load(self):
        # load from self.parent_map.folder and self.chrom
        filename = 'mtx-' + self.chrom + '-' + self.chrom + '.npy'
        if not self.loaded:
            self.map = np.load(self.parent_map.folder + filename)
            if not np.allclose(self.map.transpose(), self.map):
                logger.warning('This map is not too symmetric')
            self.size = self.map.shape[0]
            self.loaded = True

    def _calculate_IS_for_bin(self, i, square_size):
        is_i = np.nan
        if i >= square_size and i < self.size - square_size:  # skip first s_s and last s_s bins
            square = self.map[i - square_size:i, i + 1:i + 1 + square_size]
            is_i = np.nanmean(square)

        return (is_i)

    def calculate_IS(self, square_size):
        """

        :param square_size: either a list of integers corresponding to bp sizes of squares or one integer.
            Square sizes should be dividable by bin resolution of maps provided (i.e map resolution is 10000bp and
            square sizes are 40000, 50000, 100000 bp. Bad square size here would be 15000bp or 14021 bp).
        :return:
        """
        self.ISs = []
        for b in range(self.size):
            self.ISs.append(self._calculate_IS_for_bin(b, square_size))

    def normalize_IS(self, square_size):
        """
        Calculates the normalized Insulation Scores
        :param square_size: square size(s) that IS has been calculated to be normalized for
        :return:
        """
        is_avg = n(self.ISs)
        self.normalized_ISs = list(map(np.log2, [i / is_avg for i in self.ISs]))

    def get_deltas(self, square_size, delta_window_size):
        """
        Calculates deltas for each bin from normalized IS
        :param square_size:
        :param delta_window_size: Size of the delta window to calculate the left_mean and right_mean
        :return:delta
        """

        self.deltas = []
        for b in range(self.size):
            left_square = self.normalized_ISs[b - delta_window_size:b]
            right_square = self.normalized_ISs[b + 1:b + 1 + delta_window_size]
            left_mean = np.nanmean(left_square)
            right_mean = np.nanmean(right_square)
            delta = left_mean - right_mean
            if np.isinf(delta) or np.isneginf(delta):
                delta = np.nan
            self.deltas.append(delta)

        self.plot_deltas(square_size)

    def plot_deltas(self, square_size):
        plt.plot(self.deltas, alpha=0.6)
        plt.plot(self.normalized_ISs, alpha=0.6)
        plt.savefig('./figs/deltas_chr' + self.chrom + '_' + str(square_size) + '.png', dpi=300)
        plt.clf()

    def get_IS_minimums(self, square_size, delta_window_size, border_strength=0.1):
        border_info = set()
        # get the zero crossings
        zero_crossings = list(np.where(np.diff(np.sign(np.nan_to_num(self.deltas))))[0]) + list(
            np.where(np.array(self.deltas) == 0.0)[0])
        border_info.update([(z, 0.0, 0) for z in zero_crossings])
        gradient = np.gradient(self.deltas)
        extremes = list(np.where(np.diff(np.sign(np.nan_to_num(gradient))))[0]) + list(np.where(gradient == 0.0)[0])
        border_info.update([(e, abs(np.nan_to_num(self.deltas[e])), np.sign(self.deltas[e])) for e in extremes])
        self.border_info = sorted(list(border_info), key=itemgetter(0, 1))

    def check_Si_threshold(self, max, min, border_strength):
        """Checks if the border is strong enough according to Border Strength (Si)"""
        Si = max - min
        if Si <= border_strength:
            return False
        return True

    def call_TADs(self, square_size, border_strength):
        """
        Calls TADs by calculated IS and deltas
        :param square_size: square size that IS has been calculated and normalized to call TADs for.
        :return:
        """
        borders = [0]  # we always start the first TAD at 0
        last_min = None
        last_max = None
        diff_max = 0
        diff_min = 0
        for i, b in enumerate(self.border_info):
            if b[2] == 0:
                if last_max == b:
                    continue
                next_max = self.find_next_max(i, square_size)
                next_min = self.find_next_min(i, square_size)
                dist_to_next_max = next_max[0] - b[0]
                dist_to_last_max = b[0] - last_max[0] if last_max is not None else 0
                diff_max = dist_to_last_max - dist_to_next_max 
                dist_to_next_min = next_min[0] - b[0]
                dist_to_last_min = b[0] - last_min[0] if last_min is not None else 0
                diff_min = dist_to_next_min - dist_to_last_min

                if diff_max <= 0 and diff_min <= 0 and self.check_Si_threshold(
                        last_max[1] if last_max is not None else b[1],
                        last_min[2] * last_min[1] if last_min is not None else b[2] * b[1],
                        border_strength):
                    borders.append(b[0])

            elif b[2] < 0:
                last_min = b
            else:
                last_max = b

        borders.append(self.size)
        self.save_borders_to_bed(borders, square_size)

    def save_borders_to_bed(self, borders, square_size):
        res = self.parent_map.bin_res / 1000  # in Kb
        borders_bed = open('./BEDs/borders-bin_' + self.parent_map.folder.split('/')[-2] + '_' + self.chrom + "_" + str(
            int(square_size * res)) + ".bed", 'w')
        borders_bed.write('header\n')
        bed_writer = csv.writer(borders_bed, delimiter="\t")
        for i, border in enumerate(borders[:-1]):
            bed_writer.writerow([i, self.chrom, border, borders[i + 1]])
        borders_bed.close()
        borders_bed_2 = open(
            './BEDs/borders-bp_' + self.parent_map.folder.split('/')[-2] + '_' + self.chrom + "_" + str(
                int(square_size * res)) + ".bed", 'w')
        bed_writer_2 = csv.writer(borders_bed_2, delimiter="\t")
        for i, border in enumerate(borders[:-1]):
            bed_writer_2.writerow(
                [self.chrom, border * self.parent_map.bin_res, borders[i + 1] * self.parent_map.bin_res])
        tad_ends = open(
            './tad_ends/tadends-' + self.parent_map.folder.split('/')[-2] + '-' + self.chrom + '-' + str(
                int(square_size * res)) + '.txt',
            'w')
        for border in borders[1:]:
            tad_ends.write(str(border) + '\n')

    def find_next_max(self, index, square_size):
        for b in self.border_info[index + 1:-square_size]:
            if b[2] > 0:
                return (b)
        return self.border_info[index]

    def find_next_min(self, index, square_size):
        for b in self.border_info[index + 1:-square_size]:
            if b[2] < 0:
                return (b)
        return self.border_info[index]


if __name__ == "__main__":
    # Testing
    folders_list = []  # put your folders with maps in format mtx-CHR-CHR.npy here
    Is = [5, 10, 15]  # set your Insulation Square Sizes (in bins) here for which you want to get TADS
    chrs = [str(x) for x in range(1, 23)] + ['X']  # human chrs
    resolution = 40000  # set your resolution here (in bp)
    delta = 3  # set your Delta Square Size (in bins) with which you want to get TADS

    for folder in folders_list:
        for I in Is:  # 200,400,600kb
            print(folder, I)
            h = HiCmap(folder, chrs, resolution)
            h.calculate_insulation_all_chr(I)
            h.call_tads_by_IS_all_chr(I, delta)
