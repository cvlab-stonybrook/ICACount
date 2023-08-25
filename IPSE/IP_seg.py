'''
Integral Penalty Density Map Segmentation
By Yifeng Huang(yifehuang@cs.stonybrook.edu)
Created 2021.10.5
Last Modified 2022.2.14
'''
import datetime
import os
import cv2
import tqdm
import json
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

#This four hyper-parameters is not used
#Just Ignore them
REGION_BOTTOM_BOUND = 250
REGION_UPPER_BOUND = 1250
TOO_SMALL_PENALTY = 1
TOO_LARGE_PENALTY = 1
'''
#####################################################Region Viusalization##############################################
'''
class int_region():

    def __init__(self):
        #Region Attribute
        self.pixel_list = []
        self.label_index = 0
        self.pix_num = 0
        self.peakx = 0
        self.peaky = 0
        self.sum = 0
        self.panelty = 0
        self.boundary = []

        #Inter Search
        self.best_pixel_list = None
        self.best_panelty = None
        self.best_sum = None

    def peak_sarch(self, smooth_density):
        smooth_density_map_flatten = np.ndarray.flatten(smooth_density)
        sort_index = np.flip(np.argsort(smooth_density_map_flatten))
        max_index = sort_index[0]
        y = np.floor(max_index / smooth_density.shape[1]).astype(np.int32)
        x = max_index - y * smooth_density.shape[1]
        self.peaky = y
        self.peakx = x

    def check_pixel(self, y, x, density_map, label_map):
        if y < 0 or y == density_map.shape[0]:
            return False
        if x < 0 or x == density_map.shape[1]:
            return False
        if label_map[y][x] != 0:
            return False
        return True

    def expand(self, density, smooth_density, label, region_label, Limit=4):
        w, h = density.shape
        total = w * h
        prediction = np.rint(np.sum(density))
        CONUT_LIMIT = Limit
        #T_{l} in our paper
        REGION_BOTTOM_BOUND = min(np.rint(total / prediction) * CONUT_LIMIT, 150)
        #Stop size T in our paper
        REGION_UPPER_BOUND = min(REGION_BOTTOM_BOUND * 5, 750)

        py = self.peaky
        px = self.peakx
        new_pix = [[py, px]]
        label[py][px] = region_label
        #For Search Next Peak
        smooth_density[py][px] = -1
        #Cal Panelty
        sum = density[py][px]
        #Size Constrain
        pix_num = 1
        self.label_index = region_label
        #Expanding
        for_num = 1
        bak_num = 0
        self.best_pixel_list = new_pix
        self.best_sum = copy.deepcopy(sum)
        self.best_penalty = np.abs(sum - np.rint(sum)) / max(np.rint(sum), 1)
        if pix_num < REGION_BOTTOM_BOUND:
            self.best_penalty += ((REGION_BOTTOM_BOUND - pix_num) / REGION_BOTTOM_BOUND)
        if sum > CONUT_LIMIT:
            self.best_penalty += np.ceil(sum - CONUT_LIMIT)

        while len(new_pix) != 0:
            sy, sx = new_pix.pop(0)
            self.pixel_list.append([sy, sx])
            for new_y in [sy - 1, sy, sy + 1]:
                for new_x in [sx - 1, sx, sx + 1]:
                    if self.check_pixel(new_y, new_x, smooth_density, label):
                        #Add Foreground Pixel
                        if smooth_density[new_y][new_x] > 0:
                            new_pix.append([new_y, new_x])
                            label[new_y][new_x] = region_label
                            smooth_density[new_y][new_x] = -1
                            sum += density[new_y][new_x]
                            pix_num += 1
                            for_num += 1
                            if self.best_pixel_list is None:
                                self.best_pixel_list = self.pixel_list + new_pix
                                self.best_sum = copy.deepcopy(sum)
                                self.best_penalty = np.abs(sum - np.rint(sum)) / max(np.rint(sum), 1)
                                if pix_num < REGION_BOTTOM_BOUND:
                                    self.best_penalty += ((REGION_BOTTOM_BOUND - pix_num) / REGION_BOTTOM_BOUND)
                                if sum > CONUT_LIMIT:
                                    self.best_penalty += np.ceil(sum - CONUT_LIMIT)
                            else:
                                new_penalty = np.abs(sum - np.rint(sum)) / max(np.rint(sum), 1)
                                if pix_num < REGION_BOTTOM_BOUND:
                                    new_penalty += ((REGION_BOTTOM_BOUND - pix_num) / REGION_BOTTOM_BOUND)
                                if sum > CONUT_LIMIT:
                                    new_penalty += np.ceil(sum - CONUT_LIMIT)
                                if new_penalty <= self.best_penalty:
                                    self.best_penalty = new_penalty
                                    self.best_pixel_list = self.pixel_list + new_pix
                                    self.best_sum = copy.deepcopy(sum)
                        elif for_num > bak_num:
                            if self.check_pixel(new_y, new_x, smooth_density, label) and density[new_y][new_x] == 0:
                                new_pix.append([new_y, new_x])
                                self.pixel_list.append([new_y, new_x])
                                label[new_y][new_x] = region_label
                                smooth_density[new_y][new_x] = -1
                                pix_num += 1
                                bak_num += 1
            if pix_num > REGION_UPPER_BOUND:
                break

        if self.best_pixel_list is not None and len(self.best_pixel_list) != 0:
            self.pixel_list = self.best_pixel_list
            self.sum = self.best_sum
            self.panelty = self.best_penalty
        self.pix_num = len(self.pixel_list)
        return self.pixel_list, self.sum

class back_region():

    def __init__(self):
        #Region Attribute
        self.pixel_list = []
        self.label_index = 0
        self.pix_num = 0
        self.peakx = 0
        self.peaky = 0
        self.sum = 0
        self.boundary = []

    def peak_sarch(self, smooth_density):
        smooth_density_map_flatten = np.ndarray.flatten(smooth_density)
        sort_index = np.flip(np.argsort(smooth_density_map_flatten))
        max_index = sort_index[0]
        y = np.floor(max_index / smooth_density.shape[1]).astype(np.int32)
        x = max_index - y * smooth_density.shape[1]
        self.peaky = y
        self.peakx = x

    def check_pixel(self, y, x, density_map, label_map):
        if y < 0 or y == density_map.shape[0]:
            return False
        if x < 0 or x == density_map.shape[1]:
            return False
        if label_map[y][x] != 0:
            return False
        return True

    def expand(self, density, smooth_density, label, region_label):

        '''
        This function is to generate background regions
        T_{l} is not used in this function, since I don't constrain the region size of background regions
        T is the same as foreground regions' generation
        '''

        w, h = density.shape
        total = w * h
        prediction = np.rint(np.sum(density))
        REGION_BOTTOM_BOUND = min(np.rint(total / prediction) * 10, 150)
        REGION_UPPER_BOUND = min(REGION_BOTTOM_BOUND * 5, 750)
        py = self.peaky
        px = self.peakx
        new_pix = [[py, px]]
        label[py][px] = region_label
        smooth_density[py][px] = -1
        sum = density[py][px]
        pix_num = 0
        self.label_index = region_label


        while len(new_pix) != 0:
            sy, sx = new_pix.pop(0)
            self.pixel_list.append([sy, sx])
            is_boundary = True
            for new_y in [sy - 1, sy, sy + 1]:
                for new_x in [sx - 1, sx, sx + 1]:
                    if self.check_pixel(new_y, new_x, smooth_density, label):
                        new_pix.append([new_y, new_x])
                        label[new_y][new_x] = region_label
                        smooth_density[new_y][new_x] = -1
                        sum += density[new_y][new_x]
                        pix_num += 1
                        is_boundary = False
            if pix_num > REGION_UPPER_BOUND:
                self.sum = sum
                break
        self.pix_num = len(self.pixel_list)
        return self.pixel_list, self.sum, self.boundary

class VIS():

    def __init__(self, density, downsample = 4):
        density = density.squeeze()
        self.Ldensity = copy.deepcopy(density)
        self.Llabel = np.zeros(self.Ldensity.shape).astype(np.int32)
        self.prediction = np.sum(density)
        kernel_1 = np.ones((9, 9), np.float32) / 81
        kernel_2 = np.ones((7, 7), np.float32) / 49
        ratio = 255 / np.max(density)
        smooth_density = density * ratio
        smooth_density = cv2.filter2D(smooth_density, -1, kernel_1)
        smooth_density = cv2.filter2D(smooth_density, -1, kernel_2)
        self.Lsmooth_density = copy.deepcopy(smooth_density)

        t_smooth_density = torch.from_numpy(smooth_density).unsqueeze(0).unsqueeze(0)
        t_density = torch.from_numpy(density).unsqueeze(0).unsqueeze(0)
        s_t_smooth_density = F.interpolate(t_smooth_density, size=(
        int(t_smooth_density.shape[-2] / downsample), int(t_smooth_density.shape[-1] / downsample)), mode='bilinear')
        s_t_density = F.interpolate(t_density, size=(int(t_density.shape[-2] / downsample), int(t_density.shape[-1] / downsample)),
                                    mode='bilinear')
        Ssmooth_density = s_t_smooth_density.numpy().squeeze()
        Ssmooth_density *= np.sum(smooth_density) / np.sum(Ssmooth_density)
        Sdensity = s_t_density.numpy().squeeze()
        Sdensity *= np.sum(density) / np.sum(Sdensity)

        # Final segmentation result
        Slabel = np.zeros(Sdensity.shape).astype(np.int32)
        self.Sdensity = copy.deepcopy(Sdensity)
        self.Ssmooth_density = copy.deepcopy(Ssmooth_density)
        self.Slabel = Slabel
        self.boundary_mask = np.zeros(Sdensity.shape).astype(np.int32)

    def merge(self, region, region_list):
        label = copy.deepcopy(self.Slabel)
        height, width = label.shape
        new_label = None
        if new_label == None:
            for spix in region.pixel_list:
                spixy, spixx = spix
                if spixy + 1 < height and label[spixy + 1][spixx] != region.label_index:
                    new_label = label[spixy + 1][spixx]
                    break

                if spixx + 1 < width and label[spixy][spixx + 1] != region.label_index:
                    new_label = label[spixy][spixx + 1]
                    break

                if spixy - 1 >= 0 and label[spixy - 1][spixx] != region.label_index:
                    new_label = label[spixy - 1][spixx]
                    break

                if spixx - 1 >= 0 and label[spixy][spixx - 1] != region.label_index:
                    new_label = label[spixy][spixx - 1]
                    break

        for reg_index in range(len(region_list)):
            if region_list[reg_index].label_index == new_label:
                new_label_index = reg_index
        region_list[new_label_index].pixel_list += region.pixel_list
        region_list[new_label_index].sum += region.sum
        region_list[new_label_index].pix_num += region.pix_num
        for pix in region.pixel_list:
            label[pix[0]][pix[1]] = new_label
        self.Slabel = label
        return region_list

    def solve(self):
        #Pos Region Gen
        pixel_sum = self.prediction
        self.region_list = []
        label_index = 1
        total_list = []
        while pixel_sum > 1:
            temp_region = int_region()
            density = copy.deepcopy(self.Sdensity)
            smooth_density = copy.deepcopy(self.Ssmooth_density)
            label = copy.deepcopy(self.Slabel)
            temp_region.peak_sarch(smooth_density)
            pix_list, _ = temp_region.expand(density, smooth_density, label, label_index)
            sum = 0
            for reg_pix in pix_list:
                self.Ssmooth_density[reg_pix[0]][reg_pix[1]] = -1
                self.Slabel[reg_pix[0]][reg_pix[1]] = label_index
                sum += self.Sdensity[reg_pix[0]][reg_pix[1]]
            label_index += 1
            total_list += pix_list
            pixel_sum -= sum
            self.region_list.append(temp_region)

        temp_region = int_region()
        density = copy.deepcopy(self.Sdensity)
        smooth_density = copy.deepcopy(self.Ssmooth_density)
        label = copy.deepcopy(self.Slabel)
        temp_region.peak_sarch(smooth_density)
        pix_list, sum = temp_region.expand(density, smooth_density, label, label_index)
        for reg_pix in pix_list:
            self.Ssmooth_density[reg_pix[0]][reg_pix[1]] = -1
            self.Slabel[reg_pix[0]][reg_pix[1]] = label_index
        pixel_sum -= sum
        self.region_list.append(temp_region)
        label_index += 1

        #Background Regions
        while np.max(self.Ssmooth_density) != -1:
            temp_region = back_region()
            density = copy.deepcopy(self.Sdensity)
            smooth_density = copy.deepcopy(self.Ssmooth_density)
            label = copy.deepcopy(self.Slabel)
            temp_region.peak_sarch(smooth_density)
            pix_list, sum, boundary = temp_region.expand(density, smooth_density, label, label_index)
            for reg_pix in pix_list:
                self.Ssmooth_density[reg_pix[0]][reg_pix[1]] = -1
                self.Slabel[reg_pix[0]][reg_pix[1]] = label_index
            self.region_list.append(temp_region)
            label_index += 1

        #After Merge Small Region

        while True:
            no_small = True
            for i in range(len(self.region_list)):
                reg = self.region_list[i]
                if reg.pix_num < 250 and reg.sum < 0.5:
                    # region_list.remove(reg)
                    new_region_list = self.merge(reg, self.region_list)
                    new_region_list.remove(reg)
                    self.region_list = new_region_list
                    no_small = False
                    break
            if no_small:
                break

        #Reindex Each Region
        index = 1
        for region in self.region_list:
            old_label = region.label_index
            new_label = index
            region.label_index = new_label
            self.Slabel[self.Slabel == old_label] = new_label
            index += 1

        self.Slabel -= 1
        small_label = torch.from_numpy(self.Slabel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        #Must use the nearest to do the upsampling!!!!!!!!!!!!!
        resize_label = F.interpolate(small_label, size=(self.Llabel.shape[0], self.Llabel.shape[1]), mode='nearest')
        resize_label = resize_label.numpy().squeeze()
        self.Llabel = np.rint(resize_label).astype(np.int32)

    def get_boundary(self):
        #Only for visualization, you can ignore this function
        #Draw the boundary of each region
        b_num = 0
        self.Sboundary_mask = np.ones(self.Slabel.shape)
        boundary_set = []
        h, w = self.Slabel.shape
        index = 1
        for reg in self.region_list:
            new_boundary_set = []
            exist_boundary_set = []
            for pix in boundary_set:
                y, x = pix[0], pix[1]
                for new_pix in [[y + 1, x], [y - 1, x], [y, x + 1], [y, x - 1]]:
                    if new_pix in reg.pixel_list:
                        exist_boundary_set.append(pix)
            check_list = reg.pixel_list + exist_boundary_set
            for pix in check_list:
                y,x = pix[0], pix[1]
                is_bound = False
                for new_pix in [[y+1, x], [y-1, x], [y, x+1], [y, x-1]]:
                    if new_pix not in check_list:
                        is_bound = True
                if is_bound:
                    b_num += 1
                    new_boundary_set.append([y, x])
                    self.Sboundary_mask[y][x] = 0
            index += 1
            boundary_set += new_boundary_set
        self.Sboundary = np.rint(self.Sboundary_mask).astype(np.int32)
        small_boundary = torch.from_numpy(self.Sboundary_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        resize_boundary = F.interpolate(small_boundary, size=(self.Llabel.shape[0], self.Llabel.shape[1]), mode='bilinear')
        resize_boundary = resize_boundary.numpy().squeeze()
        self.Lboundary = np.rint(resize_boundary).astype(np.int32)

    def get_boundary_poly(self):
        # Only for visualization, you can ignore this function
        # Draw the boundary of each region using poly
        b_num = 0
        self.Sboundary_mask = np.zeros(self.Slabel.shape)
        boundary_set = []
        h, w = self.Slabel.shape
        index = 1
        for reg in self.region_list:
            new_boundary_set = []
            for pix in reg.pixel_list:
                y,x = pix[0], pix[1]
                is_bound = False
                for new_pix in [[y + 1, x], [y - 1, x], [y, x + 1], [y, x - 1]]:
                    if new_pix not in reg.pixel_list:
                        is_bound = True
                if is_bound:
                    b_num += 1
                    new_boundary_set.append([y, x])
                    self.Sboundary_mask[y][x] = index
            index += 1
            boundary_set += new_boundary_set
            break
        small_boundary = torch.from_numpy(self.Sboundary_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        resize_boundary = F.interpolate(small_boundary, size=(self.Llabel.shape[0], self.Llabel.shape[1]), mode='bilinear')
        resize_boundary = resize_boundary.numpy().squeeze()
        self.Lboundary = np.rint(resize_boundary).astype(np.int32)

    def get_color(self):
        # Only for visualization, you can ignore this function
        # Create a color mask of each different regions
        classes = np.max(self.Llabel) + 1
        palette = np.random.randint(0, 255, size=(classes, 3))
        palette = np.array(palette)
        color_seg = np.zeros((self.Llabel.shape[0], self.Llabel.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[self.Llabel == label, :] = color
        color_seg = color_seg[..., ::-1]
        return color_seg