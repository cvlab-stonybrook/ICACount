import json
import os
import csv
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import matplotlib
import tkinter.font as font
from tkinter import filedialog, messagebox
import torch
from FamNet.Model import Resnet50FPN, F_CountRegressor_CS
import numpy as np
from FamNet.utils import extract_features, TransformTrain
import copy
import datetime
import matplotlib.pyplot as plt
from IPSE.IP_seg import VIS
from omegaconf import OmegaConf
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageTk

def interactive_loss_app(density, estimate_lb, estimate_ub, mask, count_limit = 4):
    density = density * mask
    if estimate_ub == 5:
        loss = max(0, count_limit - density.sum())
    elif estimate_lb == -1:
        loss = (density ** 2).sum()
    else:
        loss = max(0, estimate_lb - density.sum()) + max(0, density.sum() - estimate_ub)
    return loss


def get_uncertain_state_app(density, estimate_lb, estimate_ub, mask, count_limit = 4):
    density = density * mask
    if estimate_ub == 5:
        uncertain_state = -1
    elif estimate_lb == -1 and density.sum() != 0:
        uncertain_state = 1
    elif density.sum() > estimate_ub:
        uncertain_state = 1
    elif density.sum() < estimate_lb:
        uncertain_state = -1
    else:
        uncertain_state = 0
    return uncertain_state


def Region_Dot(visual, density, rects):
    peakset = []
    first_len_wid = True
    for rec in rects:
        y1, x1, y2, x2 = rec
        if first_len_wid:
            min_y = y2 - y1
            min_x = x2 - x1
            first_len_wid = False
        else:
            min_y = min(min_y, y2 - y1)
            min_x = min(min_x, x2 - x1)
    min_y = int(min_y / 4)
    min_x = int(min_x / 4)
    density = density.squeeze()
    prediction = np.sum(density)
    ratio = 255 / np.max(density)
    smooth_density = density * ratio
    kernel_1 = np.ones((9, 9), np.float32) / 81
    kernel_2 = np.ones((7, 7), np.float32) / 49
    smooth_density = cv2.filter2D(smooth_density, -1, kernel_1)
    smooth_density = cv2.filter2D(smooth_density, -1, kernel_2)
    t_smooth_density = torch.from_numpy(smooth_density).unsqueeze(0).unsqueeze(0)
    t_density = torch.from_numpy(density).unsqueeze(0).unsqueeze(0)
    s_t_smooth_density = F.interpolate(t_smooth_density, size=(
        int(t_smooth_density.shape[-2] / 4), int(t_smooth_density.shape[-1] / 4)), mode='bilinear')
    s_t_density = F.interpolate(t_density, size=(int(t_density.shape[-2] / 4), int(t_density.shape[-1] / 4)),
                                mode='bilinear')
    Ssmooth_density = s_t_smooth_density.numpy().squeeze()
    Ssmooth_density *= np.sum(smooth_density) / np.sum(Ssmooth_density)
    Sdensity = s_t_density.numpy().squeeze()
    Sdensity *= np.sum(density) / np.sum(Sdensity)
    smooth_density = Sdensity
    peak_set = []
    for reg in visual.region_list:
        index = reg.label_index
        temp_label = copy.deepcopy(visual.Slabel)
        temp_label -= (index - 1)
        temp_mask = np.zeros((temp_label.shape[0], temp_label.shape[1]), dtype=np.uint8)
        temp_mask[temp_label == 0] = 1
        temp_smooth_density = smooth_density * temp_mask
        #temp_smooth_density *= visual.Sboundary
        dot_num = np.rint(reg.sum).astype(np.int32)
        if dot_num >0:
            y = reg.peaky
            x = reg.peakx
            peak_set.append([y * 4, x * 4])
            temp_smooth_density[max(y - (min_y) // 2, 0): min(y + (min_y) // 2 + 1, temp_smooth_density.shape[0]),
            max(x - (min_x) // 2, 0): min(x + (min_x) // 2 + 1, temp_smooth_density.shape[1])] = -1
            dot_num -= 1
            for i in range(dot_num):
                smooth_density_map_flatten = np.ndarray.flatten(temp_smooth_density)
                sort_index = np.flip(np.argsort(smooth_density_map_flatten))
                max_index = sort_index[i]
                y = np.floor(max_index / temp_smooth_density.shape[1]).astype(np.int32)
                x = max_index - y * temp_smooth_density.shape[1]
                peak_set.append([y * 4, x * 4])
                temp_smooth_density[max(y - (min_y) // 2, 0): min(y + (min_y) // 2 + 1, temp_smooth_density.shape[0]),
                max(x - (min_x) // 2, 0): min(x + (min_x) // 2 + 1, temp_smooth_density.shape[1])] = -1
    return peak_set

class ICACountInterface(tk.Tk):
    def __init__(self, path):
        super().__init__()
        self.title("Interactive Class-Agnostic Counting Interface")
        self.geometry("1460x950")
        self.resizable(False, False)
        self.Main_cv = tk.Canvas(self, width = 1460, height= 950)
        self.Display_width = 640
        self.Display_height = 360
        # Init Variable
        self.COUNT_RESULT = 0
        self.DRAWING_ENABLED = False
        self.EXEMPLAR_START_X = None
        self.EXEMPLAR_START_Y = None
        self.EXEMPLAR_END_X = None
        self.EXEMPLAR_END_Y = None
        self.Image_path = None
        # Init Layout
        self.init_image_density_result_area()
        self.init_interactive_counting_area()
        self.init_exemplar_area()
        self.init_menu_bar()
        self.init_visual_counter()
        self.init_popup_menu()
        self.bind('<Button-3>', self.popup)

    def init_exemplar_area(self):
        RECTANGLE_BASE_X = 740
        RECTANGLE_BASE_Y = 500
        BASE_X = 758
        BASE_Y = 500

        self.Main_cv.create_rectangle(RECTANGLE_BASE_X, RECTANGLE_BASE_Y, RECTANGLE_BASE_X + 700,
                                      RECTANGLE_BASE_Y + 180)
        self.Main_cv.pack()
        Input_Image_Text = tk.Label(self, text="Exempalr Providing", fg='blue')
        Input_Image_Text.pack()
        Input_Image_Text['font'] = font.Font(size=15)
        Input_Image_Text.place(x=BASE_X + 25, y=BASE_Y - 12)

        self.ExempalrB = tk.Button(self, text="Exempalr", command=self.activate_drawing)
        self.ExempalrB['font'] = font.Font(size=15)
        self.ExempalrB.place(x=BASE_X + 50, y=BASE_Y + 70)
        self.ExempalrB["width"] = 15
        self.ExempalrB["height"] = 1
        self.ExempalrB["relief"] = tk.GROOVE

        self.ExempalrUnDoB = tk.Button(self, text="Undo", command=self.exemplar_undo)
        self.ExempalrUnDoB['font'] = font.Font(size=15)
        self.ExempalrUnDoB.place(x=BASE_X + 250, y=BASE_Y + 70)
        self.ExempalrUnDoB["width"] = 15
        self.ExempalrUnDoB["height"] = 1
        self.ExempalrUnDoB["relief"] = tk.GROOVE

        self.ExempalrResetB = tk.Button(self, text="Reset", command=self.exemplar_reset)
        self.ExempalrResetB['font'] = font.Font(size=15)
        self.ExempalrResetB.place(x=BASE_X + 450, y=BASE_Y + 70)
        self.ExempalrResetB["width"] = 15
        self.ExempalrResetB["height"] = 1
        self.ExempalrResetB["relief"] = tk.GROOVE

        self.EXEMPLAR_LIST = []


    def init_interactive_counting_area(self):
        RECTANGLE_BASE_X = 740
        RECTANGLE_BASE_Y = 740
        BASE_X = 758
        BASE_Y = 735

        # Init Layout
        self.Main_cv.create_rectangle(RECTANGLE_BASE_X, RECTANGLE_BASE_Y, RECTANGLE_BASE_X + 700, RECTANGLE_BASE_Y + 180)
        self.Main_cv.pack()
        Input_Image_Text = tk.Label(self, text="Interactive Counting", fg='blue')
        Input_Image_Text.pack()
        Input_Image_Text['font'] = font.Font(size=15)
        Input_Image_Text.place(x= BASE_X + 25, y = BASE_Y - 12)

        # Count Label
        count_label = tk.Label(self, text="Counting Result: ", fg='red')
        count_label.pack()
        count_label['font'] = font.Font(size=18)
        count_label.place(x=BASE_X + 50, y=BASE_Y + 20)
        self.count_res_string_var = tk.StringVar()
        self.count_res_string_var.set("0")
        self.count_res_label = tk.Label(self, textvariable=self.count_res_string_var, fg='red')
        self.count_res_label['font'] = font.Font(size=18)
        self.count_res_label.pack()
        self.count_res_label.place(x=BASE_X + 250, y=BASE_Y + 21)

        self.count_button = tk.Button(self, text="Initial Count", command=self.initial_count)
        self.count_button['font'] = font.Font(size=15)
        self.count_button.place(x=BASE_X + 50, y=BASE_Y + 70)
        self.count_button["width"] = 15
        self.count_button["height"] = 1
        self.count_button["relief"] = tk.GROOVE

        self.UnDoB = tk.Button(self, text="Undo", command=self.interactive_undo)
        self.UnDoB['font'] = font.Font(size=15)
        self.UnDoB.place(x=BASE_X + 250, y=BASE_Y + 70)
        self.UnDoB["width"] = 15
        self.UnDoB["height"] = 1
        self.UnDoB["relief"] = tk.GROOVE

        self.ResetB = tk.Button(self, text="Reset", command=self.interactive_reset)
        self.ResetB['font'] = font.Font(size=15)
        self.ResetB.place(x=BASE_X + 450, y=BASE_Y + 70)
        self.ResetB["width"] = 15
        self.ResetB["height"] = 1
        self.ResetB["relief"] = tk.GROOVE

        gs_str_var = tk.StringVar()
        gs_str_var.set('10')
        gs_label = tk.Label(self, text="Gradient steps: ")
        gs_label.pack()
        gs_label['font'] = font.Font(size=12)
        gs_label.place(x=BASE_X + 100, y=BASE_Y + 130)

        self.GS_entry = tk.Entry(self, bd=3, textvariable=gs_str_var)
        self.GS_entry['font'] = font.Font(size=12)
        self.GS_entry.place(x=BASE_X + 230, y=BASE_Y + 130)
        self.GS_entry["width"] = 5

        lr_str_var = tk.StringVar()
        lr_str_var.set('0.02')
        lr_label = tk.Label(self, text="Learning rate: ")
        lr_label.pack()
        lr_label['font'] = font.Font(size=12)
        lr_label.place(x=BASE_X + 330, y=BASE_Y + 130)

        self.lr_entry = tk.Entry(self, bd=3, textvariable=lr_str_var)
        self.lr_entry['font'] = font.Font(size=12)
        self.lr_entry.place(x=BASE_X + 460, y=BASE_Y + 130)
        self.lr_entry["width"] = 5


    def init_image_density_result_area(self):
        # Init Image Area
        IMAGE_RECTANGLE_BASE_X = 20
        IMAGE_RECTANGLE_BASE_Y = 30
        self.Main_cv.create_rectangle(IMAGE_RECTANGLE_BASE_X, IMAGE_RECTANGLE_BASE_Y, IMAGE_RECTANGLE_BASE_X + 700, IMAGE_RECTANGLE_BASE_Y + 420)
        self.Main_cv.pack()
        Input_Image_Text = tk.Label(self, text="Input Image", fg='blue')
        Input_Image_Text.pack()
        Input_Image_Text['font'] = font.Font(size=15)
        Input_Image_Text.place(x=IMAGE_RECTANGLE_BASE_X + 25, y=IMAGE_RECTANGLE_BASE_Y - 12)
        self.Input_Image_Label = tk.Label(self, text="")
        self.Input_Image_Label.pack()
        self.Input_Image_Label.place(x=IMAGE_RECTANGLE_BASE_X + 25, y=IMAGE_RECTANGLE_BASE_Y + 28)
        self.EXEMPLAR_LIST = []

        default_img = Image.open('interface_default.jpg')
        default_img = default_img.resize((self.Display_width, self.Display_height))
        default_show = ImageTk.PhotoImage(default_img)
        self.Input_Image = default_img
        self.Input_Image_backup = self.Input_Image.copy()
        self.Input_Image_with_exemplar = self.Input_Image.copy()

        self.Input_Image_Label.configure(image=default_show)
        self.Input_Image_Label.image = default_show
        self.Input_Image_Label.bind("<Button-1>", self.start_draw_rectangle)
        self.Input_Image_Label.bind("<B1-Motion>", self.update_rectangle)
        self.Input_Image_Label.bind("<ButtonRelease-1>", self.end_draw_rectangle)

        # Init Density Area
        DENSITY_RECTANGLE_BASE_X = 740
        DENSITY_RECTANGLE_BASE_Y = 30
        self.Main_cv.create_rectangle(DENSITY_RECTANGLE_BASE_X, DENSITY_RECTANGLE_BASE_Y, DENSITY_RECTANGLE_BASE_X + 700, DENSITY_RECTANGLE_BASE_Y + 420)
        self.Main_cv.pack()
        self.Density_Label = tk.Label(self, text="")
        self.Density_Label.pack()
        self.Density_Label.place(x=DENSITY_RECTANGLE_BASE_X + 25, y=DENSITY_RECTANGLE_BASE_Y + 28)
        default_img = Image.open('interface_default.jpg')
        default_img = default_img.resize((self.Display_width, self.Display_height))
        default_show = ImageTk.PhotoImage(default_img)
        self.Density_Label.configure(image=default_show)
        self.Density_Label.image = default_show
        Density_Text = tk.Label(self, text="Estimated Density", fg='blue')
        Density_Text.pack()
        Density_Text['font'] = font.Font(size=15)
        Density_Text.place(x=DENSITY_RECTANGLE_BASE_X + 25, y=DENSITY_RECTANGLE_BASE_Y - 12)

        # Init Result Area
        RESULT_RECTANGLE_BASE_X = 20
        RESULT_RECTANGLE_BASE_Y = 500
        self.Main_cv.create_rectangle(RESULT_RECTANGLE_BASE_X, RESULT_RECTANGLE_BASE_Y, RESULT_RECTANGLE_BASE_X + 700, RESULT_RECTANGLE_BASE_Y + 420)
        self.Main_cv.pack()
        Input_Image_Text = tk.Label(self, text="Result Visualization", fg='blue')
        Input_Image_Text.pack()
        Input_Image_Text['font'] = font.Font(size=15)
        Input_Image_Text.place(x=RESULT_RECTANGLE_BASE_X + 25, y=RESULT_RECTANGLE_BASE_Y - 12)
        self.Visual_Label = tk.Label(self, text="")
        self.Visual_Label.pack()
        self.Visual_Label.place(x=RESULT_RECTANGLE_BASE_X + 25, y=RESULT_RECTANGLE_BASE_Y + 28)
        default_img = Image.open('interface_default.jpg')
        default_img = default_img.resize((self.Display_width, self.Display_height))
        default_show = ImageTk.PhotoImage(default_img)
        self.Visual_Label.configure(image=default_show)
        self.Visual_Label.image = default_show


    def init_menu_bar(self):
        # Init menu bar
        self.MB = tk.Menu(self)
        self.config(menu=self.MB)
        self.MB.add_command(label="Load Image", command=self.Menu_Bar_Load_Image)


    def Menu_Bar_Load_Image(self):
        # Init Variable
        self.COUNT_RESULT = 0
        self.DRAWING_ENABLED = False
        self.EXEMPLAR_START_X = None
        self.EXEMPLAR_START_Y = None
        self.EXEMPLAR_END_X = None
        self.EXEMPLAR_END_Y = None
        self.Image_path = None
        # Init Layout
        self.init_image_density_result_area()
        self.init_interactive_counting_area()
        self.init_exemplar_area()
        self.init_menu_bar()
        self.init_visual_counter()
        self.init_popup_menu()
        # The only important para
        self.Image_path = filedialog.askopenfilename(initialdir="./", title="Select image.")
        img_open = Image.open(self.Image_path)
        self.max_hw = 1504
        W, H = img_open.size
        img_open = img_open.resize((self.Display_width, self.Display_height))
        img_show = ImageTk.PhotoImage(img_open)
        self.Input_Image_Label.configure(image=img_show)
        self.Input_Image_Label.image = img_show
        self.Input_Image = img_open.copy()
        self.Input_Image_with_exemplar = img_open.copy()
        self.Input_Image_backup = img_open.copy()
        self.Image_Ori_W = W
        self.Image_Ori_H = H

    def activate_drawing(self):
        self.DRAWING_ENABLED = True

    def start_draw_rectangle(self, event):
        if not self.DRAWING_ENABLED:
            return
        self.EXEMPLAR_START_X = event.x
        self.EXEMPLAR_START_Y = event.y
        self.EXEMPLAR_BBX = (self.EXEMPLAR_START_X, self.EXEMPLAR_START_Y, self.EXEMPLAR_START_X, self.EXEMPLAR_START_Y)


    def update_rectangle(self, event):
        if not self.DRAWING_ENABLED or self.EXEMPLAR_BBX is None:
            return
        self.EXEMPLAR_BBX = (self.EXEMPLAR_START_X, self.EXEMPLAR_START_Y, event.x, event.y)
        self.Input_Image_with_exemplar = self.Input_Image.copy()
        draw = ImageDraw.Draw(self.Input_Image_with_exemplar)
        draw.rectangle(self.EXEMPLAR_BBX, outline='red', width=2)
        photo = ImageTk.PhotoImage(self.Input_Image_with_exemplar)
        self.Input_Image_Label.configure(image=photo)
        self.Input_Image_Label.image = photo


    def end_draw_rectangle(self, event):
        if not self.DRAWING_ENABLED:
            return
        self.Input_Image = self.Input_Image_with_exemplar.copy()
        self.EXEMPLAR_END_X = event.x
        self.EXEMPLAR_END_Y = event.y
        x_min = min(self.EXEMPLAR_START_X, self.EXEMPLAR_END_X)
        x_max = max(self.EXEMPLAR_START_X, self.EXEMPLAR_END_X)
        y_min = min(self.EXEMPLAR_START_Y, self.EXEMPLAR_END_Y)
        y_max = max(self.EXEMPLAR_START_Y, self.EXEMPLAR_END_Y)
        self.EXEMPLAR_LIST.append((x_min, y_min, x_max, y_max))
        self.EXEMPLAR_BBX = None
        self.DRAWING_ENABLED = False


    def exemplar_reset(self):
        self.EXEMPLAR_LIST = []
        self.Input_Image = self.Input_Image_backup.copy()
        self.Input_Image_with_exemplar = self.Input_Image_backup.copy()
        photo = ImageTk.PhotoImage(self.Input_Image)
        self.Input_Image_Label.configure(image=photo)
        self.Input_Image_Label.image = photo

        default_img = Image.open('./interface_default.jpg')
        default_img = default_img.resize((self.Display_width, self.Display_height))
        default_show = ImageTk.PhotoImage(default_img)
        self.Density_Label.configure(image=default_show)
        self.Density_Label.image = default_show

        default_img = Image.open('./interface_default.jpg')
        default_img = default_img.resize((self.Display_width, self.Display_height))
        default_show = ImageTk.PhotoImage(default_img)
        self.Visual_Label.configure(image=default_show)
        self.Visual_Label.image = default_show

    def exemplar_undo(self):
        if len(self.EXEMPLAR_LIST) == 0:
            return
        self.EXEMPLAR_LIST.pop(-1)
        self.Input_Image_with_exemplar = self.Input_Image_backup.copy()
        for exemplar in self.EXEMPLAR_LIST:
            draw = ImageDraw.Draw(self.Input_Image_with_exemplar)
            draw.rectangle(exemplar, outline='red', width=2)
        self.Input_Image = self.Input_Image_with_exemplar.copy()
        photo = ImageTk.PhotoImage(self.Input_Image_with_exemplar)
        self.Input_Image_Label.configure(image=photo)
        self.Input_Image_Label.image = photo

    def init_visual_counter(self):
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = 'cpu'
        self.resnet50_conv = Resnet50FPN()
        self.visual_counter = F_CountRegressor_CS(6, pool='mean')
        self.visual_counter.load_state_dict(torch.load('../Checkpoints/FSC_147/FamNet.pth'))
        self.visual_counter.to(self.device)
        self.resnet50_conv.to(self.device)
        self.resnet50_conv.eval()
        self.MAPS = ['map3', 'map4']
        self.Scales = [0.9, 1.1]
        self.INGS = 10
        self.INLR = 2e-2
        self.label = None
        self.estimate_lb = None
        self.estimate_ub = None
        self.inter_mask_list = []
        self.visual_couter_list = []
        self.visual_list = []
        self.density_list = []
        self.INIT_COUNT_FLAG = False

    def initial_count(self):
        if self.Image_path is None:
            messagebox.showinfo("Image Invalid", "Please select an image.")
            return

        if len(self.EXEMPLAR_LIST) < 3:
            messagebox.showinfo("Exemplar Invalid", "Please select at least 3 exemplars.")
            return

        if len(self.EXEMPLAR_LIST) > 3:
            messagebox.showinfo("Exemplar Invalid", "For FamNet only the first 3 exemplars are used.")
            self.COUNTING_EXEMPLAR_LIST = copy.deepcopy(self.EXEMPLAR_LIST[:3])
        else:
            self.COUNTING_EXEMPLAR_LIST = copy.deepcopy(self.EXEMPLAR_LIST)

        rects = list()
        for exemplar in self.COUNTING_EXEMPLAR_LIST:
            x_min, y_min, x_max, y_max = exemplar
            image_x_min = int(x_min * self.Image_Ori_W / self.Display_width)
            image_x_max = int(x_max * self.Image_Ori_W / self.Display_width)
            image_y_min = int(y_min * self.Image_Ori_H / self.Display_height)
            image_y_max = int(y_max * self.Image_Ori_H / self.Display_height)
            rects.append([image_y_min, image_x_min, image_y_max, image_x_max])
        self.rects = copy.deepcopy(rects)
        image = Image.open(self.Image_path)
        image.load()
        W, H = image.size
        density = np.zeros((W, H))
        boxes = np.array(rects)
        sample = {'image': image, 'lines_boxes': boxes, 'gt_density': density}
        sample = TransformTrain(sample)
        # Preparing Data
        image, boxes, gt_density = sample['image'].to(self.device), sample['boxes'].to(self.device), sample['gt_density'].to(self.device)

        # Initial Counting
        with torch.no_grad():
            self.features = extract_features(self.resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), self.MAPS, self.Scales)
            output, self.simifeat = self.visual_counter(self.features)
        self.visual_counter.reset_refinement_module(self.features.shape[-2], self.features.shape[-1])
        self.visual_counter.to(self.device)
        count_result = output.sum().item()
        count_result = np.round(count_result, decimals=2)
        self.count_res_string_var.set(str(count_result))

        # Save the density map as jpg, and show it
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axis_off()
        ax.imshow(output.squeeze().detach().cpu().numpy())
        fig.savefig('Density_map.jpg', bbox_inches="tight", dpi=600, pad_inches=0.0)
        density_img = Image.open('./Density_map.jpg')
        density_img = density_img.resize((self.Display_width, self.Display_height))
        density_show = ImageTk.PhotoImage(density_img)
        self.Density_Label.configure(image=density_show)
        self.Density_Label.image = density_show

        # Segmentation then show the result
        density = output.squeeze().detach().cpu().numpy()
        self.Real_Height = density.shape[0]
        self.Real_Width = density.shape[1]
        self.vis_start_time = datetime.datetime.now()
        visual = VIS(density)
        visual.solve()

        label = visual.Llabel
        self.label = label
        self.vis_end_time = datetime.datetime.now()
        max_label = np.max(self.label)
        seg_img = cv2.imread(self.Image_path)
        seg_img = cv2.resize(seg_img, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_AREA)
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
        fig = plt.figure()
        axe = fig.add_subplot(1, 1, 1)
        axe.imshow(seg_img)
        axe.set_axis_off()
        peak_set = Region_Dot(visual, density, self.rects)
        axe.zorder = 0
        axe.contour(label, colors='#FFFF00', linewidths=2.5, corner_mask=True, levels=max_label, zorder=0)
        axe.contour(label, colors='k', linewidths=0.8, linestyles='-', alpha=0.8, corner_mask=True, levels=max_label,
                    zorder=0)
        self.old_peak_set = peak_set
        for peak in peak_set:
            peak_y, peak_x = peak
            circle = plt.Circle((peak_x, peak_y), 2, color='r')
            circle.set_zorder(1)
            axe.add_patch(circle)
        fig.savefig('Visualize.jpg', bbox_inches="tight", dpi=600, pad_inches=0.0)
        visual_img = Image.open('./Visualize.jpg')
        visual_img = visual_img.resize((self.Display_width, self.Display_height))
        visual_show = ImageTk.PhotoImage(visual_img)
        self.Visual_Label.configure(image=visual_show)
        self.Visual_Label.image = visual_show

        self.visual_list.append(copy.deepcopy(visual))
        self.density_list.append(copy.deepcopy(density))
        self.visual_couter_list.append(copy.deepcopy(self.visual_counter))
        self.INIT_COUNT_FLAG = True
        plt.cla()
        plt.clf()
        plt.close()
        plt.close('all')

    def init_popup_menu(self):
        # For range speification
        self.popup_menu = tk.Menu(self, tearoff=0)
        self.popup_menu.add_command(label='(-inf,0]', command=self.popup_menu_0)
        self.popup_menu.add_command(label='(0,1]', command=self.popup_menu_1)
        self.popup_menu.add_command(label='(1,2]', command=self.popup_menu_2)
        self.popup_menu.add_command(label='(2,3]', command=self.popup_menu_3)
        self.popup_menu.add_command(label='(3,4]', command=self.popup_menu_4)
        self.popup_menu.add_command(label='(4,inf]', command=self.popup_menu_5)

    def popup_menu_0(self):
        self.estimate_lb = -1
        self.estimate_ub = 0
        self.interactive_adaptation()

    def popup_menu_1(self):
        self.estimate_lb = 0
        self.estimate_ub = 1
        self.interactive_adaptation()

    def popup_menu_2(self):
        self.estimate_lb = 1
        self.estimate_ub = 2
        self.interactive_adaptation()

    def popup_menu_3(self):
        self.estimate_lb = 2
        self.estimate_ub = 3
        self.interactive_adaptation()

    def popup_menu_4(self):
        self.estimate_lb = 3
        self.estimate_ub = 4
        self.interactive_adaptation()

    def popup_menu_5(self):
        self.estimate_lb = 4
        self.estimate_ub = 5
        self.interactive_adaptation()

    def popup(self, event):
        global click_index
        if self.label is not None:
            # Convert to Image coordinate
            trans_y = int((self.Real_Height / self.Display_height) * event.y)
            trans_x = int((self.Real_Width / self.Display_width) * event.x)
            self.selected_region = self.label[trans_y, trans_x]
        click_index = 0
        self.popup_menu.post(event.x_root, event.y_root)

    def interactive_adaptation(self):
        if not self.INIT_COUNT_FLAG:
            messagebox.showinfo("Initial Counting First", "Please do initial counting first.")
            return
        self.INLR = float(self.lr_entry.get())
        self.INGS = float(self.GS_entry.get())
        output = self.visual_counter.inter_inference(self.simifeat)
        sample_label = self.selected_region
        inter_mask = np.zeros((self.label.shape[0], self.label.shape[1]), dtype=np.uint8)
        inter_mask[self.label == sample_label] = 1
        inter_mask = torch.from_numpy(inter_mask).cuda()
        self.inter_mask_list.append([inter_mask, self.estimate_lb, self.estimate_ub])
        over_counting_num = 0
        under_counting_num = 0
        for mask_range in self.inter_mask_list:
            inmask, estimate_lb, estimate_ub = mask_range
            uncertain_state = get_uncertain_state_app(output, estimate_lb, estimate_ub, inmask)
            if uncertain_state == 1:
                over_counting_num += 1
            elif uncertain_state == -1:
                under_counting_num += 1
        scale_1 = min(1, np.exp(((len(self.inter_mask_list) + 1) - 3) / 2))
        if over_counting_num == 0 or under_counting_num == 0:
            scale_2 = 1
        else:
            over_p = over_counting_num / (over_counting_num + under_counting_num)
            under_p = under_counting_num / (over_counting_num + under_counting_num)
            uncertain = (over_p * np.log(over_p)) + (under_p * np.log(under_p))
            scale_2 = 1 + uncertain
        scale = (scale_1 + scale_2) / 2
        scale_INLR = self.INLR * scale
        scale_INGS = np.rint(self.INGS / scale).astype(np.int32)
        optimizer_inter = torch.optim.Adam(
            [self.visual_counter.ch_scale, self.visual_counter.ch_bias, self.visual_counter.sp_scale,
             self.visual_counter.sp_bias], lr=scale_INLR)
        self.features.required_grad = True

        for step in range(0, scale_INGS):
            optimizer_inter.zero_grad()
            output = self.visual_counter.inter_inference(self.simifeat)
            inter_loss = 0
            # Local Adaptation loss
            local_region_loss = 0.
            for mask_range in self.inter_mask_list:
                inmask, estimate_lb, estimate_ub = mask_range
                inter_loss = interactive_loss_app(output, estimate_lb, estimate_ub, inmask)
                local_region_loss += inter_loss

            # Global Adaptation Loss
            all_inter_mask = np.zeros((self.label.shape[0], self.label.shape[1]), dtype=np.uint8)
            all_lb = 0
            all_ub = 0
            global_region_num = 0
            for mask_range in self.inter_mask_list:
                inmask, estimate_lb, estimate_ub = mask_range
                if estimate_ub == 5:
                    continue
                if estimate_lb == -1:
                    estimate_lb = 0
                all_lb += estimate_lb
                all_ub += estimate_ub
                global_region_num += 1
                all_inter_mask += inmask.cpu().numpy()
            all_inter_mask = torch.from_numpy(all_inter_mask).cuda()
            new_count_limit = 4 * global_region_num
            global_region_loss = interactive_loss_app(output, all_lb, all_ub, all_inter_mask, new_count_limit)

            inertial_loss = ((self.visual_counter.ch_scale - 1) ** 2).sum() + (
                    self.visual_counter.ch_bias ** 2).sum() + (
                                    (self.visual_counter.sp_scale - 1) ** 2).sum() + (
                                    self.visual_counter.sp_bias ** 2).sum()

            inter_loss = 0.5 * local_region_loss + 0.5 * global_region_loss + 1e-3 * inertial_loss
            if torch.is_tensor(inter_loss):
                inter_loss.backward()
                optimizer_inter.step()

        self.visual_counter.eval()
        with torch.no_grad():
            output = self.visual_counter.inter_inference(self.simifeat)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axis_off()
        ax.imshow(output.squeeze().detach().cpu().numpy())
        fig.savefig('Density_map.jpg', bbox_inches="tight", dpi=600, pad_inches=0.0)
        density_img = Image.open('./Density_map.jpg')
        density_img = density_img.resize((self.Display_width, self.Display_height))
        density_show = ImageTk.PhotoImage(density_img)
        self.Density_Label.configure(image=density_show)
        self.Density_Label.image = density_show
        density = output.squeeze().detach().cpu().numpy()
        visual = VIS(density)
        visual.solve()
        label = visual.Llabel
        self.label = label
        min_label = np.min(label)
        max_label = np.max(label)
        self.Real_Height = density.shape[0]
        self.Real_Width = density.shape[1]
        seg_img = cv2.imread(self.Image_path)
        seg_img = cv2.resize(seg_img, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_AREA)
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

        # Contour
        fig = plt.figure()
        axe = fig.add_subplot(1, 1, 1)
        axe.imshow(seg_img)

        axe.set_axis_off()
        peak_set = Region_Dot(visual, density, self.rects)
        axe.zorder = 0
        axe.contour(label, colors='#FFFF00', linewidths=2.5, corner_mask=True, levels=max_label, zorder=0)
        axe.contour(label, colors='k', linewidths=0.8, linestyles='-', alpha=0.8, corner_mask=True, levels=max_label,
                    zorder=0)
        for peak in peak_set:
            peak_y, peak_x = peak
            circle = plt.Circle((peak_x, peak_y), 2, color='r')
            circle.set_zorder(1)
            axe.add_patch(circle)
        fig.savefig('Visualize.jpg', bbox_inches="tight", dpi=600, pad_inches=0.0)
        visual_img = Image.open('./Visualize.jpg')
        visual_img = visual_img.resize((self.Display_width, self.Display_height))
        visual_show = ImageTk.PhotoImage(visual_img)
        self.Visual_Label.configure(image=visual_show)
        self.Visual_Label.image = visual_show

        self.visual_list.append(copy.deepcopy(visual))
        self.density_list.append(copy.deepcopy(density))
        self.visual_couter_list.append(copy.deepcopy(self.visual_counter))
        plt.cla()
        plt.clf()
        plt.close()
        plt.close('all')

    def interactive_undo(self):
        if not self.INIT_COUNT_FLAG:
            messagebox.showinfo("Initial Counting First", "Please do initial counting first.")
            return
        if len(self.inter_mask_list) == 0:
            self.visual_counter = self.visual_couter_list[0]
            temp_density = self.density_list[0]
            temp_visual = self.visual_list[0]
            assert len(self.visual_couter_list) == len(self.density_list) == len(self.visual_list) == len(self.inter_mask_list) + 1
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.imshow(temp_density)
            fig.savefig('Density_map.jpg', bbox_inches="tight", dpi=600, pad_inches=0.0)
            density_img = Image.open('./Density_map.jpg')
            density_img = density_img.resize((self.Display_width, self.Display_height))
            density_show = ImageTk.PhotoImage(density_img)
            self.Density_Label.configure(image=density_show)
            self.Density_Label.image = density_show

            label = temp_visual.Llabel
            self.label = label
            self.vis_end_time = datetime.datetime.now()
            max_label = np.max(self.label)
            seg_img = cv2.imread(self.Image_path)
            seg_img = cv2.resize(seg_img, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_AREA)
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
            fig = plt.figure()
            axe = fig.add_subplot(1, 1, 1)
            axe.imshow(seg_img)
            axe.set_axis_off()
            peak_set = Region_Dot(temp_visual, temp_density, self.rects)
            axe.zorder = 0
            axe.contour(label, colors='#FFFF00', linewidths=2.5, corner_mask=True, levels=max_label, zorder=0)
            axe.contour(label, colors='k', linewidths=0.8, linestyles='-', alpha=0.8, corner_mask=True,
                        levels=max_label,
                        zorder=0)
            self.old_peak_set = peak_set
            for peak in peak_set:
                peak_y, peak_x = peak
                circle = plt.Circle((peak_x, peak_y), 2, color='r')
                circle.set_zorder(1)
                axe.add_patch(circle)
            fig.savefig('Visualize.jpg', bbox_inches="tight", dpi=600, pad_inches=0.0)
            visual_img = Image.open('./Visualize.jpg')
            visual_img = visual_img.resize((self.Display_width, self.Display_height))
            visual_show = ImageTk.PhotoImage(visual_img)
            self.Visual_Label.configure(image=visual_show)
            self.Visual_Label.image = visual_show

        else:
            self.visual_couter_list.pop(-1)
            self.density_list.pop(-1)
            self.visual_list.pop(-1)
            self.inter_mask_list.pop(-1)
            assert len(self.visual_couter_list) == len(self.density_list) == len(self.visual_list) == len(self.inter_mask_list) + 1
            self.visual_counter = self.visual_couter_list[-1]
            temp_density = self.density_list[-1]
            temp_visual = self.visual_list[-1]
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.imshow(temp_density)
            fig.savefig('Density_map.jpg', bbox_inches="tight", dpi=600, pad_inches=0.0)
            density_img = Image.open('./Density_map.jpg')
            density_img = density_img.resize((self.Display_width, self.Display_height))
            density_show = ImageTk.PhotoImage(density_img)
            self.Density_Label.configure(image=density_show)
            self.Density_Label.image = density_show

            label = temp_visual.Llabel
            self.label = label
            self.vis_end_time = datetime.datetime.now()
            max_label = np.max(self.label)
            seg_img = cv2.imread(self.Image_path)
            seg_img = cv2.resize(seg_img, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_AREA)
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
            fig = plt.figure()
            axe = fig.add_subplot(1, 1, 1)
            axe.imshow(seg_img)
            axe.set_axis_off()
            peak_set = Region_Dot(temp_visual, temp_density, self.rects)
            axe.zorder = 0
            axe.contour(label, colors='#FFFF00', linewidths=2.5, corner_mask=True, levels=max_label, zorder=0)
            axe.contour(label, colors='k', linewidths=0.8, linestyles='-', alpha=0.8, corner_mask=True,
                        levels=max_label,
                        zorder=0)
            self.old_peak_set = peak_set
            for peak in peak_set:
                peak_y, peak_x = peak
                circle = plt.Circle((peak_x, peak_y), 2, color='r')
                circle.set_zorder(1)
                axe.add_patch(circle)
            fig.savefig('Visualize.jpg', bbox_inches="tight", dpi=600, pad_inches=0.0)
            visual_img = Image.open('./Visualize.jpg')
            visual_img = visual_img.resize((self.Display_width, self.Display_height))
            visual_show = ImageTk.PhotoImage(visual_img)
            self.Visual_Label.configure(image=visual_show)
            self.Visual_Label.image = visual_show

    def interactive_reset(self):
        if not self.INIT_COUNT_FLAG:
            messagebox.showinfo("Initial Counting First", "Please do initial counting first.")
            return
        self.visual_counter = self.visual_couter_list[0]
        temp_density = self.density_list[0]
        temp_visual = self.visual_list[0]
        self.inter_mask_list = []
        self.visual_couter_list = self.visual_couter_list[:1]
        self.density_list = self.density_list[:1]
        self.visual_list = self.visual_list[:1]
        assert len(self.visual_couter_list) == len(self.density_list) == len(self.visual_list) == len(
            self.inter_mask_list) + 1
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axis_off()
        ax.imshow(temp_density)
        fig.savefig('Density_map.jpg', bbox_inches="tight", dpi=600, pad_inches=0.0)
        density_img = Image.open('./Density_map.jpg')
        density_img = density_img.resize((self.Display_width, self.Display_height))
        density_show = ImageTk.PhotoImage(density_img)
        self.Density_Label.configure(image=density_show)
        self.Density_Label.image = density_show

        label = temp_visual.Llabel
        self.label = label
        self.vis_end_time = datetime.datetime.now()
        max_label = np.max(self.label)
        seg_img = cv2.imread(self.Image_path)
        seg_img = cv2.resize(seg_img, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_AREA)
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
        fig = plt.figure()
        axe = fig.add_subplot(1, 1, 1)
        axe.imshow(seg_img)
        axe.set_axis_off()
        peak_set = Region_Dot(temp_visual, temp_density, self.rects)
        axe.zorder = 0
        axe.contour(label, colors='#FFFF00', linewidths=2.5, corner_mask=True, levels=max_label, zorder=0)
        axe.contour(label, colors='k', linewidths=0.8, linestyles='-', alpha=0.8, corner_mask=True,
                    levels=max_label,
                    zorder=0)
        self.old_peak_set = peak_set
        for peak in peak_set:
            peak_y, peak_x = peak
            circle = plt.Circle((peak_x, peak_y), 2, color='r')
            circle.set_zorder(1)
            axe.add_patch(circle)
        fig.savefig('Visualize.jpg', bbox_inches="tight", dpi=600, pad_inches=0.0)
        visual_img = Image.open('./Visualize.jpg')
        visual_img = visual_img.resize((self.Display_width, self.Display_height))
        visual_show = ImageTk.PhotoImage(visual_img)
        self.Visual_Label.configure(image=visual_show)
        self.Visual_Label.image = visual_show


if __name__ == "__main__":
    win = ICACountInterface(None)
    win.mainloop()
