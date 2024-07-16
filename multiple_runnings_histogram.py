import numpy as np
from matplotlib import pyplot as plt, font_manager, patches
from matplotlib.transforms import Bbox
from matplotlib.ticker import FuncFormatter
import matplotlib.lines as lines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import os
from itertools import chain

plt.rcParams['font.serif'] = ['Palatino Linotype']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 27
plt.rcParams['mathtext.fontset'] = 'cm'

data_amount = 10


data_folder = r'C:\Users\PC\PycharmProjects\Thin_Films_Project\Daniel_figure_codes\data\new_data'

output_folder = r'C:\Users\PC\PycharmProjects\Thin_Films_Project\figures'
output_figure_name = 'multiple_running_histogram_figure'
show_or_save = False # None for nothing, True for show, False for save
start_indx = None
end_indx = None

def get_bboxes(axes):
    bboxes = []
    tightbboxes = []
    for i in range(2):
        bboxes.append([])
        for j in range(4):
            bboxes[i].append(axes[i][j].get_position())
    for i in range(2):
        tightbboxes.append([])
        for j in range(4):
            tightbboxes[i].append(Bbox(fig.transFigure.inverted().transform(axes[i][j].get_tightbbox(renderer=frenderer))))
    return bboxes, tightbboxes

def get_label_widths(bboxes, tightbboxes):
    leftpos_ylabel_width = tightbboxes[1][0].width - bboxes[1][0].width
    leftdiff_ylabel_width = tightbboxes[1][1].width - bboxes[1][1].width

    rightpos_ylabel_width = tightbboxes[1][2].width - bboxes[1][2].width
    rightdiff_ylabel_width = tightbboxes[1][3].width - bboxes[1][3].width
    return [leftpos_ylabel_width, leftdiff_ylabel_width, rightpos_ylabel_width, rightdiff_ylabel_width]

def draw_bboxes(bboxes, colors):
    bboxes_flat = chain.from_iterable(bboxes)
    colors_flat = chain.from_iterable(colors)
    for bbox, color in zip(bboxes_flat, colors_flat):
        rect = patches.Rectangle((bbox.x0, bbox.y0), bbox.width, bbox.height, fill=False, color=color, transform=fig.transFigure, figure=fig)
        fig.patches.extend([rect])

def arange(vec):
    a_vec = np.zeros((10,))
    for num in vec:
        index = int(num * 10)
        if index == 10:
            index = 9
        a_vec[index] = a_vec[index] + 1
    return a_vec

def load_results_i(base_dir, i):
    file_name1 = 'layers5_DNN_' + str(i) + '_histogram.npz'
    res1 = dict(np.load(os.path.join(base_dir, file_name1)))
    file_name2 = 'layers5_Transformer_' + str(i) + '_histogram.npz'
    res2 = dict(np.load(os.path.join(base_dir, file_name2)))
    file_name3 = 'unknown_layers_DNN_' + str(i) + '_histogram.npz'
    res3 = dict(np.load(os.path.join(base_dir, file_name3)))
    file_name4 = 'unknown_layers_Transformer_' + str(i) + '_histogram.npz'
    res4 = dict(np.load(os.path.join(base_dir, file_name4)))

    res = {}

    res['layers5_DNN_T_histogram'] = arange(res1['T_histogram'])
    res['layers5_DNN_R_histogram'] = arange(res1['R_histogram'])
    res['layers5_DNN_Structure_histogram'] = arange(res1['Structure_histogram'])
    res['layers5_DNN_Refractive_histogram'] = arange(res1['Refractive_histogram'])

    res['layers5_Transformer_T_histogram'] = arange(res2['T_histogram'])
    res['layers5_Transformer_R_histogram'] = arange(res2['R_histogram'])
    res['layers5_Transformer_Structure_histogram'] = arange(res2['Structure_histogram'])
    res['layers5_Transformer_Refractive_histogram'] = arange(res2['Refractive_histogram'])

    res['unknown_layers_DNN_T_histogram'] = arange(res3['T_histogram'])
    res['unknown_layers_DNN_R_histogram'] = arange(res3['R_histogram'])
    res['unknown_layers_DNN_Structure_histogram'] = arange(res3['Structure_histogram'])
    res['unknown_layers_DNN_Refractive_histogram'] = arange(res3['Refractive_histogram'])

    res['unknown_layers_Transformer_T_histogram'] = arange(res4['T_histogram'])
    res['unknown_layers_Transformer_R_histogram'] = arange(res4['R_histogram'])
    res['unknown_layers_Transformer_Structure_histogram'] = arange(res4['Structure_histogram'])
    res['unknown_layers_Transformer_Refractive_histogram'] = arange(res4['Refractive_histogram'])

    #average

    res['layers5_DNN_T_histogram_average_all'] = np.average(res1['T_histogram']) * 100
    res['layers5_DNN_R_histogram_average_all'] = np.average(res1['R_histogram']) * 100

    res['layers5_Transformer_T_histogram_average_all'] = np.average(res2['T_histogram']) * 100
    res['layers5_Transformer_R_histogram_average_all'] = np.average(res2['R_histogram']) * 100

    res['unknown_layers_DNN_T_histogram_average_all'] = np.average(res3['T_histogram']) * 100
    res['unknown_layers_DNN_R_histogram_average_all'] = np.average(res3['R_histogram']) * 100

    res['unknown_layers_Transformer_T_histogram_average_all'] = np.average(res4['T_histogram']) * 100
    res['unknown_layers_Transformer_R_histogram_average_all'] = np.average(res4['R_histogram']) * 100

    return res

def image_to_Dlist(image):
    lis = []
    vec = image[0]
    prev = image[0,0]
    n = prev
    d = 0
    for i in range(len(vec)):
        if vec[i] == prev:
            d = d + 1
        else:
            lis.append(d)
            d = 1
            prev = vec[i]
    lis.append(d)
    lis = np.array(lis)
    lis = lis * (10**-7)
    return lis, n

def error_correl_filter(n_o, n_p, d_o, d_p):
    if (n_o != n_p) or (len(d_o) != len(d_p)):
        return 1
    sum = 0
    for i in range(len(d_o)):
        L1 = d_o[i]
        L2 = d_p[i]
        sum = sum + np.abs(L1 - L2)/(L1 + L2)
    sum = sum / len(d_o)
    return sum

def NDlist_to_filter(NDlist):
    d = NDlist[:, 1]
    d = np.array(d)
    d = d * (10**-7)
    new_d = []
    for i in range(5):
        if d[i] != 0:
            new_d.append(d[i])
    new_d = np.array(new_d)
    n = NDlist[0,0] * 255.0
    return new_d, n

def clean_dp(d_p):
    new_d = []
    for i in range(5):
        if d_p[i] != 0:
            new_d.append(d_p[i])
    new_d = np.array(new_d)
    return new_d

def calculate_error_structure_i(base_dir, i):
    # file_name1 = 'layers5_DNN_' + str(i) + '_filters.npz'
    # filter1 = dict(np.load(os.path.join(base_dir, file_name1)))
    # file_name2 = 'layers5_Transformer_' + str(i) + '_filters.npz'
    # filter2 = dict(np.load(os.path.join(base_dir, file_name2)))
    # file_name3 = 'unknown_layers_DNN_' + str(i) + '_filters.npz'
    # filter3 = dict(np.load(os.path.join(base_dir, file_name3)))
    # file_name4 = 'unknown_layers_Transformer_' + str(i) + '_filters.npz'
    # filter4 = dict(np.load(os.path.join(base_dir, file_name4)))

    original_file_name1 = 'layers5_y_test.npz'
    original_filter1 = dict(np.load(os.path.join(base_dir, original_file_name1)))['layers5_DNN_y_test']
    original_filter2 = original_filter1
    original_file_name3 = 'unknown_layers_y_test.npz'
    original_filter3 = dict(np.load(os.path.join(base_dir, original_file_name3)))['unknown_layers_DNN_y_test']
    original_filter4 = original_filter3

    pred_file_name1 = 'layers5_DNN_' + str(i) + '_NDlist.npz'
    NDlists1 = dict(np.load(os.path.join(base_dir, pred_file_name1)))
    n1 = NDlists1['n']
    d1 = NDlists1['d']

    pred_file_name2 = 'layers5_Transformer_' + str(i) + '_NDlist.npz'
    NDlists2 = dict(np.load(os.path.join(base_dir, pred_file_name2)))
    n2 = NDlists2['n']
    d2 = NDlists2['d']

    pred_file_name3 = 'unknown_layers_DNN_' + str(i) + '_NDlist.npz'
    NDlists3 = dict(np.load(os.path.join(base_dir, pred_file_name3)))
    n3 = NDlists3['n']
    d3 = NDlists3['d']

    pred_file_name4 = 'unknown_layers_Transformer_' + str(i) + '_NDlist.npz'
    NDlists4 = dict(np.load(os.path.join(base_dir, pred_file_name4)))
    n4 = NDlists4['n']
    d4 = NDlists4['d']

    error1 = []
    error2 = []
    error3 = []
    error4 = []

    print('Errors for waves figure:')

    refractive_error1 = 0
    for index1 in range(1000):
        d_o, n_o = NDlist_to_filter(original_filter1[index1])
        n_p = n1[index1] * 255
        d_p = clean_dp(d1[index1])
        e = error_correl_filter(n_o, n_p, d_o, d_p)
        if e == 1:
            refractive_error1 = refractive_error1 + 1
        if index1 == 40:
            print(round(100 * e, 2))
        error1.append(e)

    refractive_error2 = 0
    for index2 in range(1000):
        d_o, n_o = NDlist_to_filter(original_filter2[index2])
        n_p = n2[index2] * 255
        d_p = clean_dp(d2[index2])
        e = error_correl_filter(n_o, n_p, d_o, d_p)
        if e == 1:
            refractive_error2 = refractive_error2 + 1
        if index2 == 40:
            print(round(100 * e, 2))
        error2.append(e)

    for index3 in range(1000):
        d_o, n_o = NDlist_to_filter(original_filter3[index3])
        n_p = n3[index3] * 255
        d_p = clean_dp(d3[index3])
        e = error_correl_filter(n_o, n_p, d_o, d_p)
        if index3 == 53:
            print(round(100 * e, 2))
        error3.append(e)

    for index4 in range(1000):
        d_o, n_o = NDlist_to_filter(original_filter4[index4])
        n_p = n4[index4] * 255
        d_p = clean_dp(d4[index4])
        e = error_correl_filter(n_o, n_p, d_o, d_p)
        if index4 == 53:
            print(round(100 * e, 2))
        error4.append(e)

    error1 = np.array(error1)
    error2 = np.array(error2)
    error3 = np.array(error3)
    error4 = np.array(error4)
    return error1, error2, error3, error4, refractive_error1, refractive_error2

def get_data_i(base_dir, i):
    res = load_results_i(base_dir, i)
    error1, error2, error3, error4, refractive_error1, refractive_error2 = calculate_error_structure_i(base_dir, i)
    res['error1'] = arange(error1)
    res['error2'] = arange(error2)
    res['error3'] = arange(error3)
    res['error4'] = arange(error4)
    res['layers5_DNN_Refractive_histogram'][0] = res['layers5_DNN_Refractive_histogram'][0] + refractive_error1
    res['layers5_DNN_Refractive_histogram'][9] = res['layers5_DNN_Refractive_histogram'][9] - refractive_error1
    res['layers5_Transformer_Refractive_histogram'][0] = res['layers5_Transformer_Refractive_histogram'][0] + refractive_error2
    res['layers5_Transformer_Refractive_histogram'][9] = res['layers5_Transformer_Refractive_histogram'][9] - refractive_error2


    res['layers5_DNN_Error_histogram_average_all'] = np.average(error1) * 100
    res['layers5_DNN_Refractive_histogram_average_all'] = res['layers5_DNN_Refractive_histogram'][9] / 10

    res['layers5_Transformer_Error_histogram_average_all'] = np.average(error2) * 100
    res['layers5_Transformer_Refractive_histogram_average_all'] = res['layers5_Transformer_Refractive_histogram'][9] / 10

    res['unknown_layers_DNN_Error_histogram_average_all'] = np.average(error3) * 100
    res['unknown_layers_DNN_Refractive_histogram_average_all'] = res['unknown_layers_DNN_Refractive_histogram'][9] / 10

    res['unknown_layers_Transformer_Error_histogram_average_all'] = np.average(error4) * 100
    res['unknown_layers_Transformer_Refractive_histogram_average_all'] = res['unknown_layers_Transformer_Refractive_histogram'][9] / 10

    return res

def histogram_analysis():
    res = {}

    res['layers5_DNN_T_histogram'] = np.zeros((data_amount,10))
    res['layers5_DNN_R_histogram'] = np.zeros((data_amount,10))
    res['layers5_DNN_Error_histogram'] = np.zeros((data_amount,10))
    res['layers5_DNN_Refractive_histogram'] = np.zeros((data_amount,10))

    res['layers5_Transformer_T_histogram'] = np.zeros((data_amount,10))
    res['layers5_Transformer_R_histogram'] = np.zeros((data_amount,10))
    res['layers5_Transformer_Error_histogram'] = np.zeros((data_amount,10))
    res['layers5_Transformer_Refractive_histogram'] = np.zeros((data_amount,10))

    res['unknown_layers_DNN_T_histogram'] = np.zeros((data_amount,10))
    res['unknown_layers_DNN_R_histogram'] = np.zeros((data_amount,10))
    res['unknown_layers_DNN_Error_histogram'] = np.zeros((data_amount,10))
    res['unknown_layers_DNN_Refractive_histogram'] = np.zeros((data_amount,10))

    res['unknown_layers_Transformer_T_histogram'] = np.zeros((data_amount,10))
    res['unknown_layers_Transformer_R_histogram'] = np.zeros((data_amount,10))
    res['unknown_layers_Transformer_Error_histogram'] = np.zeros((data_amount,10))
    res['unknown_layers_Transformer_Refractive_histogram'] = np.zeros((data_amount,10))



    layers5_DNN_T_histogram = []
    layers5_DNN_R_histogram = []
    layers5_DNN_Error_histogram = []
    layers5_DNN_Refractive_histogram = []

    layers5_Transformer_T_histogram = []
    layers5_Transformer_R_histogram = []
    layers5_Transformer_Error_histogram = []
    layers5_Transformer_Refractive_histogram = []

    unknown_layers_DNN_T_histogram = []
    unknown_layers_DNN_R_histogram = []
    unknown_layers_DNN_Error_histogram = []
    unknown_layers_DNN_Refractive_histogram = []

    unknown_layers_Transformer_T_histogram = []
    unknown_layers_Transformer_R_histogram = []
    unknown_layers_Transformer_Error_histogram = []
    unknown_layers_Transformer_Refractive_histogram = []

    for i in range(1, data_amount + 1):
        data_i = get_data_i(data_folder, i)

        res['layers5_DNN_T_histogram'][i - 1,:] = data_i['layers5_DNN_T_histogram']
        res['layers5_DNN_R_histogram'][i - 1,:] = data_i['layers5_DNN_R_histogram']
        res['layers5_DNN_Error_histogram'][i - 1,:] = data_i['error1']
        res['layers5_DNN_Refractive_histogram'][i - 1,:] = data_i['layers5_DNN_Refractive_histogram']

        layers5_DNN_T_histogram.append(data_i['layers5_DNN_T_histogram_average_all'])
        layers5_DNN_R_histogram.append(data_i['layers5_DNN_R_histogram_average_all'])
        layers5_DNN_Error_histogram.append(data_i['layers5_DNN_Error_histogram_average_all'])
        layers5_DNN_Refractive_histogram.append(data_i['layers5_DNN_Refractive_histogram_average_all'])

        res['layers5_Transformer_T_histogram'][i - 1,:] = data_i['layers5_Transformer_T_histogram']
        res['layers5_Transformer_R_histogram'][i - 1,:] = data_i['layers5_Transformer_R_histogram']
        res['layers5_Transformer_Error_histogram'][i - 1,:] = data_i['error2']
        res['layers5_Transformer_Refractive_histogram'][i - 1,:] = data_i['layers5_Transformer_Refractive_histogram']

        layers5_Transformer_T_histogram.append(data_i['layers5_Transformer_T_histogram_average_all'])
        layers5_Transformer_R_histogram.append(data_i['layers5_Transformer_R_histogram_average_all'])
        layers5_Transformer_Error_histogram.append(data_i['layers5_Transformer_Error_histogram_average_all'])
        layers5_Transformer_Refractive_histogram.append(data_i['layers5_Transformer_Refractive_histogram_average_all'])

        res['unknown_layers_DNN_T_histogram'][i - 1,:] = data_i['unknown_layers_DNN_T_histogram']
        res['unknown_layers_DNN_R_histogram'][i - 1,:] = data_i['unknown_layers_DNN_R_histogram']
        res['unknown_layers_DNN_Error_histogram'][i - 1,:] = data_i['error3']
        res['unknown_layers_DNN_Refractive_histogram'][i - 1,:] = data_i['unknown_layers_DNN_Refractive_histogram']

        unknown_layers_DNN_T_histogram.append(data_i['unknown_layers_DNN_T_histogram_average_all'])
        unknown_layers_DNN_R_histogram.append(data_i['unknown_layers_DNN_R_histogram_average_all'])
        unknown_layers_DNN_Error_histogram.append(data_i['unknown_layers_DNN_Error_histogram_average_all'])
        unknown_layers_DNN_Refractive_histogram.append(data_i['unknown_layers_DNN_Refractive_histogram_average_all'])

        res['unknown_layers_Transformer_T_histogram'][i - 1,:] = data_i['unknown_layers_Transformer_T_histogram']
        res['unknown_layers_Transformer_R_histogram'][i - 1,:] = data_i['unknown_layers_Transformer_R_histogram']
        res['unknown_layers_Transformer_Error_histogram'][i - 1,:] = data_i['error4']
        res['unknown_layers_Transformer_Refractive_histogram'][i - 1,:] = data_i['unknown_layers_Transformer_Refractive_histogram']

        unknown_layers_Transformer_T_histogram.append(data_i['unknown_layers_Transformer_T_histogram_average_all'])
        unknown_layers_Transformer_R_histogram.append(data_i['unknown_layers_Transformer_R_histogram_average_all'])
        unknown_layers_Transformer_Error_histogram.append(data_i['unknown_layers_Transformer_Error_histogram_average_all'])
        unknown_layers_Transformer_Refractive_histogram.append(data_i['unknown_layers_Transformer_Refractive_histogram_average_all'])

    res['layers5_DNN_T_histogram_avg_value'] = np.average(np.array(layers5_DNN_T_histogram))
    res['layers5_DNN_R_histogram_avg_value'] = np.average(np.array(layers5_DNN_R_histogram))
    res['layers5_DNN_Error_histogram_avg_value'] = np.average(np.array(layers5_DNN_Error_histogram))
    res['layers5_DNN_Refractive_histogram_avg_value'] = np.average(np.array(layers5_DNN_Refractive_histogram))

    res['layers5_Transformer_T_histogram_avg_value'] = np.average(np.array(layers5_Transformer_T_histogram))
    res['layers5_Transformer_R_histogram_avg_value'] = np.average(np.array(layers5_Transformer_R_histogram))
    res['layers5_Transformer_Error_histogram_avg_value'] = np.average(np.array(layers5_Transformer_Error_histogram))
    res['layers5_Transformer_Refractive_histogram_avg_value'] = np.average(np.array(layers5_Transformer_Refractive_histogram))

    res['unknown_layers_DNN_T_histogram_avg_value'] = np.average(np.array(unknown_layers_DNN_T_histogram))
    res['unknown_layers_DNN_R_histogram_avg_value'] = np.average(np.array(unknown_layers_DNN_R_histogram))
    res['unknown_layers_DNN_Error_histogram_avg_value'] = np.average(np.array(unknown_layers_DNN_Error_histogram))
    res['unknown_layers_DNN_Refractive_histogram_avg_value'] = np.average(np.array(unknown_layers_DNN_Refractive_histogram))

    res['unknown_layers_Transformer_T_histogram_avg_value'] = np.average(np.array(unknown_layers_Transformer_T_histogram))
    res['unknown_layers_Transformer_R_histogram_avg_value'] = np.average(np.array(unknown_layers_Transformer_R_histogram))
    res['unknown_layers_Transformer_Error_histogram_avg_value'] = np.average(np.array(unknown_layers_Transformer_Error_histogram))
    res['unknown_layers_Transformer_Refractive_histogram_avg_value'] = np.average(np.array(unknown_layers_Transformer_Refractive_histogram))
    return res

def avg_std_dict(data_i):
    res = {}

    res['layers5_DNN_T_histogram_avg'] = np.average(data_i['layers5_DNN_T_histogram'], axis = 0)
    res['layers5_DNN_T_histogram_std'] = np.std(data_i['layers5_DNN_T_histogram'], axis=0)

    res['layers5_DNN_R_histogram_avg'] = np.average(data_i['layers5_DNN_R_histogram'], axis = 0)
    res['layers5_DNN_R_histogram_std'] = np.std(data_i['layers5_DNN_R_histogram'], axis=0)

    res['layers5_DNN_Error_histogram_avg'] = np.average(data_i['layers5_DNN_Error_histogram'], axis = 0)
    res['layers5_DNN_Error_histogram_std'] = np.std(data_i['layers5_DNN_Error_histogram'], axis=0)

    res['layers5_DNN_Refractive_histogram_avg'] = np.average(data_i['layers5_DNN_Refractive_histogram'], axis = 0)
    res['layers5_DNN_Refractive_histogram_std'] = np.std(data_i['layers5_DNN_Refractive_histogram'], axis=0)



    res['layers5_Transformer_T_histogram_avg'] = np.average(data_i['layers5_Transformer_T_histogram'], axis = 0)
    res['layers5_Transformer_T_histogram_std'] = np.std(data_i['layers5_Transformer_T_histogram'], axis=0)

    res['layers5_Transformer_R_histogram_avg'] = np.average(data_i['layers5_Transformer_R_histogram'], axis = 0)
    res['layers5_Transformer_R_histogram_std'] = np.std(data_i['layers5_Transformer_R_histogram'], axis=0)

    res['layers5_Transformer_Error_histogram_avg'] = np.average(data_i['layers5_Transformer_Error_histogram'], axis = 0)
    res['layers5_Transformer_Error_histogram_std'] = np.std(data_i['layers5_Transformer_Error_histogram'], axis=0)

    res['layers5_Transformer_Refractive_histogram_avg'] = np.average(data_i['layers5_Transformer_Refractive_histogram'], axis = 0)
    res['layers5_Transformer_Refractive_histogram_std'] = np.std(data_i['layers5_Transformer_Refractive_histogram'], axis = 0)



    res['unknown_layers_DNN_T_histogram_avg'] = np.average(data_i['unknown_layers_DNN_T_histogram'], axis = 0)
    res['unknown_layers_DNN_T_histogram_std'] = np.std(data_i['unknown_layers_DNN_T_histogram'], axis=0)

    res['unknown_layers_DNN_R_histogram_avg'] = np.average(data_i['unknown_layers_DNN_R_histogram'], axis = 0)
    res['unknown_layers_DNN_R_histogram_std'] = np.std(data_i['unknown_layers_DNN_R_histogram'], axis=0)

    res['unknown_layers_DNN_Error_histogram_avg'] = np.average(data_i['unknown_layers_DNN_Error_histogram'], axis = 0)
    res['unknown_layers_DNN_Error_histogram_std'] = np.std(data_i['unknown_layers_DNN_Error_histogram'], axis=0)

    res['unknown_layers_DNN_Refractive_histogram_avg'] = np.average(data_i['unknown_layers_DNN_Refractive_histogram'], axis = 0)
    res['unknown_layers_DNN_Refractive_histogram_std'] = np.std(data_i['unknown_layers_DNN_Refractive_histogram'], axis = 0)



    res['unknown_layers_Transformer_T_histogram_avg'] = np.average(data_i['unknown_layers_Transformer_T_histogram'], axis = 0)
    res['unknown_layers_Transformer_T_histogram_std'] = np.std(data_i['unknown_layers_Transformer_T_histogram'],axis=0)

    res['unknown_layers_Transformer_R_histogram_avg'] = np.average(data_i['unknown_layers_Transformer_R_histogram'], axis = 0)
    res['unknown_layers_Transformer_R_histogram_std'] = np.std(data_i['unknown_layers_Transformer_R_histogram'], axis=0)

    res['unknown_layers_Transformer_Error_histogram_avg'] = np.average(data_i['unknown_layers_Transformer_Error_histogram'], axis = 0)
    res['unknown_layers_Transformer_Error_histogram_std'] = np.std(data_i['unknown_layers_Transformer_Error_histogram'], axis=0)

    res['unknown_layers_Transformer_Refractive_histogram_avg'] = np.average(data_i['unknown_layers_Transformer_Refractive_histogram'], axis = 0)
    res['unknown_layers_Transformer_Refractive_histogram_std'] = np.std(data_i['unknown_layers_Transformer_Refractive_histogram'], axis=0)

    return res

analysed_data = histogram_analysis()
histogram = avg_std_dict(analysed_data)


figure_size = (35, 23)

gt_line_style = '-'
video_pred_line_style = '-'
twoimgs_pred_line_style = '-'
lines_style = '--'
plots_linewidth = 2
lines_dash_capstyle = 'round'
lines_solid_capstyle = 'round'
dashes = (3, 3)

black = (0, 0, 0)
blue = (0, 0, 1)
red = (1, 0, 0)
purple = (0.75, 0.0, 0.75)
green = (0, 0.62, 0.22)

stages_font_size = 28

lines_color = 'k'
lines_width = 3

err_type='diffs_'

diff_line_style = '-k'
legend_frameon = True
pos_legend_loc = 'lower center'
titles_height_above_axes = 0.006
legend_height_above_titles = 0.012

plots_width = 0.21
plots_spacing = 0.03
line_space = 0.0018

fig, axes = plt.subplots(2, 4, sharex=False, figsize=figure_size)
# fig, axes = plt.subplots(3, 2, sharex=True)
frenderer = fig.canvas.get_renderer()

legend_frameon = True
pos_legend_bbox_to_anchor = (0.02, 0.98)

# x_ticklabels_font_dict = {
#                          'fontsize': 18,
#                          # 'fontweight': rcParams['axes.titleweight'],
#                          # 'verticalalignment': 'baseline',
#                          # 'horizontalalignment': loc
#                           }

bar_width = 0.3

####### set plots

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = x

DNN_label = axes[1][0].plot(x*0, gt_line_style, color=red, linewidth=plots_linewidth, label='FC-DNN')
Transformer_label = axes[1][0].plot(x*0, video_pred_line_style, color=blue, linewidth=plots_linewidth, label='Attention')


axes[0,0].bar(x + bar_width/2, histogram['layers5_DNN_T_histogram_avg'], color=red, width = bar_width, label='FC-DNN', yerr=histogram['layers5_DNN_T_histogram_std'])
axes[0,0].bar(x - bar_width/2, histogram['layers5_Transformer_T_histogram_avg'], color=blue, width = bar_width, label='Transformer', yerr=histogram['layers5_Transformer_T_histogram_std'])
axes[0,0].set_xticks(np.arange(10))
axes[0,0].set_xticklabels([f'[{10*i}, {10*(i+1)}]' for i in range(10)])
axes[0,0].tick_params(axis='x', labelsize=22, labelrotation=50)
axes[0,0].set_xlim(-bar_width, 9 + bar_width)
textstr = '\n'.join((
    r'$FC-DNN:%.2f$' % (analysed_data['layers5_DNN_T_histogram_avg_value'], ) + "%",
    r'$Attention:%.2f$' % (analysed_data['layers5_Transformer_T_histogram_avg_value'], ) + "%"))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axes[0,0].text(plots_width, 0.95, textstr, transform=axes[0,0].transAxes, fontsize=30,
        verticalalignment='top', bbox=props)


axes[0,1].bar(x + bar_width/2, histogram['layers5_DNN_R_histogram_avg'], color=red, width = bar_width, label='Random trajectory', yerr=histogram['layers5_DNN_R_histogram_std'])
axes[0,1].bar(x - bar_width/2, histogram['layers5_Transformer_R_histogram_avg'], color=blue, width = bar_width, label='Tortuous trajectory', yerr=histogram['layers5_Transformer_R_histogram_std'])
axes[0,1].set_xticks(np.arange(10))
axes[0,1].set_xticklabels([f'[{10*i}, {10*(i+1)}]' for i in range(10)])
axes[0,1].tick_params(axis='x', labelsize=22, labelrotation=50)
axes[0,1].set_xlim(-bar_width, 9 + bar_width)
textstr = '\n'.join((
    r'$FC-DNN:%.2f$' % (analysed_data['layers5_DNN_R_histogram_avg_value'], ) + "%",
    r'$Attention:%.2f$' % (analysed_data['layers5_Transformer_R_histogram_avg_value'], ) + "%"))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axes[0,1].text(plots_width, 0.95, textstr, transform=axes[0,1].transAxes, fontsize=30,
        verticalalignment='top', bbox=props)

axes[0,2].bar(x + bar_width/2, histogram['unknown_layers_DNN_T_histogram_avg'], color=red, width = bar_width, label='DNN', yerr=histogram['unknown_layers_DNN_T_histogram_std'])
axes[0,2].bar(x - bar_width/2, histogram['unknown_layers_Transformer_T_histogram_avg'], color=blue, width = bar_width, label='Transformer', yerr=histogram['unknown_layers_Transformer_T_histogram_std'])
axes[0,2].set_xticks(np.arange(10))
axes[0,2].set_xticklabels([f'[{10*i}, {10*(i+1)}]' for i in range(10)])
axes[0,2].tick_params(axis='x', labelsize=22, labelrotation=50)
axes[0,2].set_xlim(-bar_width, 9 + bar_width)
textstr = '\n'.join((
    r'$FC-DNN:%.2f$' % (analysed_data['unknown_layers_DNN_T_histogram_avg_value'], ) + "%",
    r'$Attention:%.2f$' % (analysed_data['unknown_layers_Transformer_T_histogram_avg_value'], ) + "%"))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axes[0,2].text(plots_width, 0.95, textstr, transform=axes[0,2].transAxes, fontsize=30,
        verticalalignment='top', bbox=props)


axes[0,3].bar(x + bar_width/2, histogram['unknown_layers_DNN_R_histogram_avg'], color=red, width = bar_width, label='DNN', yerr=histogram['unknown_layers_DNN_R_histogram_std'])
axes[0,3].bar(x - bar_width/2, histogram['unknown_layers_Transformer_R_histogram_avg'], color=blue, width = bar_width, label='Transformer', yerr=histogram['unknown_layers_Transformer_R_histogram_std'])
axes[0,3].set_xticks(np.arange(10))
axes[0,3].set_xticklabels([f'[{10*i}, {10*(i+1)}]' for i in range(10)])
axes[0,3].tick_params(axis='x', labelsize=22, labelrotation=50)
axes[0,3].set_xlim(-bar_width, 9 + bar_width)
textstr = '\n'.join((
    r'$FC-DNN:%.2f$' % (analysed_data['unknown_layers_DNN_R_histogram_avg_value'], ) + "%",
    r'$Attention:%.2f$' % (analysed_data['unknown_layers_Transformer_R_histogram_avg_value'], ) + "%"))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axes[0,3].text(plots_width, 0.95, textstr, transform=axes[0,3].transAxes, fontsize=30,
        verticalalignment='top', bbox=props)
axes[0,3].set_ylim(0.0)

axes[1,0].bar(x + bar_width/2, histogram['layers5_DNN_Error_histogram_avg'], color=red, width = bar_width, label='DNN', yerr=histogram['layers5_DNN_Error_histogram_std'])
axes[1,0].bar(x - bar_width/2, histogram['layers5_Transformer_Error_histogram_avg'], color=blue, width = bar_width, label='Transformer', yerr=histogram['layers5_Transformer_Error_histogram_std'])
axes[1,0].set_xticks(np.arange(10))
axes[1,0].set_xticklabels([f'[{10*i}, {10*(i+1)}]' for i in range(10)])
axes[1,0].tick_params(axis='x', labelsize=22, labelrotation=50)
axes[1,0].set_xlim(-bar_width, 9 + bar_width)
textstr = '\n'.join((
    r'$FC-DNN:%.2f$' % (analysed_data['layers5_DNN_Error_histogram_avg_value'], ) + "%",
    r'$Attention:%.2f$' % (analysed_data['layers5_Transformer_Error_histogram_avg_value'], ) + "%"))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axes[1,0].text(plots_width, 0.95, textstr, transform=axes[1,0].transAxes, fontsize=30,
        verticalalignment='top', bbox=props)
axes[1,0].set_ylim(0.0)


layers5_DNN_Refractive = np.array([histogram['layers5_DNN_Refractive_histogram_avg'][0],histogram['layers5_DNN_Refractive_histogram_avg'][9]])
layers5_Transformer_Refractive = np.array([histogram['layers5_Transformer_Refractive_histogram_avg'][0],histogram['layers5_Transformer_Refractive_histogram_avg'][9]])
layers5_DNN_Refractive_std = np.array([histogram['layers5_DNN_Refractive_histogram_std'][0],histogram['layers5_DNN_Refractive_histogram_std'][9]])
layers5_Transformer_Refractive_std = np.array([histogram['layers5_Transformer_Refractive_histogram_std'][0],histogram['layers5_Transformer_Refractive_histogram_std'][9]])
z = np.array([0, 1])
axes[1,1].bar(z + bar_width/12, layers5_DNN_Refractive, color=red, width = bar_width/6, label='Random trajectory',yerr=layers5_DNN_Refractive_std)
axes[1,1].bar(z - bar_width/12, layers5_Transformer_Refractive, color=blue, width = bar_width/6, label='Tortuous trajectory',yerr=layers5_Transformer_Refractive_std)
axes[1,1].set_xticks(np.arange(2))
li = ["False", "True"]
axes[1,1].set_xticklabels([f'[{li[i]}]' for i in range(2)])
axes[1,1].tick_params(axis='x', labelsize=22, labelrotation=50)
axes[1,1].set_xlim(-bar_width, 1 + bar_width)
textstr = '\n'.join((
    r'$FC-DNN:%.1f$' % (analysed_data['layers5_DNN_Refractive_histogram_avg_value'], ) + "%",
    r'$Attention:%.1f$' % (analysed_data['layers5_Transformer_Refractive_histogram_avg_value'], ) + "%"))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axes[1,1].text(plots_width, 0.95, textstr, transform=axes[1,1].transAxes, fontsize=30,
        verticalalignment='top', bbox=props)

axes[1,2].bar(x + bar_width/2, histogram['unknown_layers_DNN_Error_histogram_avg'], color=red, width = bar_width, label='Random trajectory', yerr=histogram['unknown_layers_DNN_Error_histogram_std'])
axes[1,2].bar(x - bar_width/2, histogram['unknown_layers_Transformer_Error_histogram_avg'], color=blue, width = bar_width, label='Tortuous trajectory', yerr=histogram['unknown_layers_Transformer_Error_histogram_std'])
axes[1,2].set_xticks(np.arange(10))
axes[1,2].set_xticklabels([f'[{10*i}, {10*(i+1)}]' for i in range(10)])
axes[1,2].tick_params(axis='x', labelsize=22, labelrotation=50)
axes[1,2].set_xlim(-bar_width, 9 + bar_width)
textstr = '\n'.join((
    r'$FC-DNN:%.2f$' % (analysed_data['unknown_layers_DNN_Error_histogram_avg_value'], ) + "%",
    r'$Attention:%.2f$' % (analysed_data['unknown_layers_Transformer_Error_histogram_avg_value'], ) + "%"))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axes[1,2].text(plots_width, 0.95, textstr, transform=axes[1,2].transAxes, fontsize=30,
        verticalalignment='top', bbox=props)
axes[1,2].set_ylim(0.0)

unknown_layers_DNN_Refractive = np.array([histogram['unknown_layers_DNN_Refractive_histogram_avg'][0],histogram['unknown_layers_DNN_Refractive_histogram_avg'][9]])
unknown_layers_Transformer_Refractive = np.array([histogram['unknown_layers_Transformer_Refractive_histogram_avg'][0],histogram['unknown_layers_Transformer_Refractive_histogram_avg'][9]])
unknown_layers_DNN_Refractive_std = np.array([histogram['unknown_layers_DNN_Refractive_histogram_std'][0],histogram['unknown_layers_DNN_Refractive_histogram_std'][9]])
unknown_layers_Transformer_Refractive_std = np.array([histogram['unknown_layers_Transformer_Refractive_histogram_std'][0],histogram['unknown_layers_Transformer_Refractive_histogram_std'][9]])
z = np.array([0, 1])
axes[1,3].bar(z + bar_width/12, unknown_layers_DNN_Refractive, color=red, width = bar_width/6, label='Random trajectory',yerr=unknown_layers_DNN_Refractive_std)
axes[1,3].bar(z - bar_width/12, unknown_layers_Transformer_Refractive, color=blue, width = bar_width/6, label='Tortuous trajectory',yerr=unknown_layers_Transformer_Refractive_std)
axes[1,3].set_xticks(np.arange(2))
li = ["False", "True"]
axes[1,3].set_xticklabels([f'[{li[i]}]' for i in range(2)])
axes[1,3].tick_params(axis='x', labelsize=22, labelrotation=50)
axes[1,3].set_xlim(-bar_width, 1 + bar_width)
textstr = '\n'.join((
    r'$FC-DNN:%.1f$' % (analysed_data['unknown_layers_DNN_Refractive_histogram_avg_value'], ) + '%',
    r'$Attention:%.1f$' % (analysed_data['unknown_layers_Transformer_Refractive_histogram_avg_value'], ) + '%'))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axes[1,3].text(plots_width, 0.95, textstr, transform=axes[1,3].transAxes, fontsize=30,
        verticalalignment='top', bbox=props)

axes[0][0].set_xlabel('Transmittance Correlation')
axes[0][1].set_xlabel('Reflectance Correlation')
axes[1][0].set_xlabel('Structure Error')
axes[1][1].set_xlabel('Refractive Correlation')
axes[1][1].yaxis.set_label_position('right')
axes[0][1].yaxis.set_ticks_position('right')
axes[1][1].yaxis.set_ticks_position('right')

axes[0][2].set_xlabel('Transmittance Correlation')
axes[0][3].set_xlabel('Reflectance Correlation')
axes[1][2].set_xlabel('Structure Error')
axes[1][3].set_xlabel('Refractive Correlation')
axes[1][3].yaxis.set_label_position('right')
axes[0][3].yaxis.set_ticks_position('right')
axes[1][3].yaxis.set_ticks_position('right')


[bboxes, tightbboxes] = get_bboxes(axes)
[leftpos_ylabel_width, leftdiff_ylabel_width, rightpos_ylabel_width, rightdiff_ylabel_width] = get_label_widths(bboxes, tightbboxes)
# axes_width = (1 - (leftpos_ylabel_width + leftdiff_ylabel_width + rightpos_ylabel_width + rightdiff_ylabel_width + 2 * plots_spacing + 2 * plots_to_text_spacing + stage_texts_max_width)) / 4


lefts = [
    leftpos_ylabel_width,
    leftpos_ylabel_width + plots_width + plots_spacing,
    1 - rightdiff_ylabel_width - plots_spacing - 2 * plots_width,
    1 - rightdiff_ylabel_width - plots_width
]

for i in range(4):
    for j in range(2):
        axes[j][i].set_position([lefts[i], bboxes[j][i].y0, plots_width, bboxes[j][i].height])

[bboxes2, tightbboxes2] = get_bboxes(axes)
[leftpos_ylabel_width2, leftdiff_ylabel_width2, rightpos_ylabel_width2, rightdiff_ylabel_width2] = get_label_widths(bboxes2, tightbboxes2)
newcenterx = (tightbboxes2[1][1].x1 + tightbboxes2[1][2].x0)/2

###### set texts
### titles
random_title_centerx = (bboxes2[0][0].x1 + bboxes2[0][1].x0) / 2
snake_title_centerx = (bboxes2[0][2].x1 + bboxes2[0][3].x0) / 2
titlesy = max([tightbboxes2[0][i].y1 for i in range(4)]) + titles_height_above_axes
random_title = plt.text(random_title_centerx, titlesy, 'Data Set 1: 5 Layers', va='bottom', ha='center', transform=fig.transFigure)
snake_title = plt.text(snake_title_centerx, titlesy, 'Data Set 2: Unknown Amount Of Layers', va='bottom', ha='center', transform=fig.transFigure)
titlesbboxes = [Bbox(fig.transFigure.inverted().transform(title.get_window_extent(renderer=frenderer))) for title in [random_title, snake_title]]

###### legend
legendy = max([titlebox.y1 for titlebox in titlesbboxes]) + legend_height_above_titles
legend1 = axes[0][0].legend(loc=pos_legend_loc, bbox_to_anchor=(newcenterx, legendy), bbox_transform=fig.transFigure, handles=[DNN_label[0], Transformer_label[0]],
                            frameon=legend_frameon, ncol=2, borderaxespad=0, borderpad=0.2, handletextpad=0.1, columnspacing=0.6, handlelength=1)
for line in legend1.get_lines():
    line.set_linewidth(3)
legend1bbox = Bbox(fig.transFigure.inverted().transform(legend1.get_tightbbox(renderer=frenderer)))

###### lines
linesy = [[legend1bbox.y0 - line_space, tightbboxes2[1][0].y0 + line_space],
          ]
lineslist = [lines.Line2D([newcenterx, newcenterx], liney, transform=fig.transFigure,
                          linestyle=lines_style, linewidth=lines_width, color=lines_color, dash_capstyle=lines_dash_capstyle, solid_capstyle=lines_solid_capstyle, dashes=dashes) for liney in linesy]
for line in lineslist: fig.add_artist(line)

if show_or_save is not None:
    if show_or_save:
        # plt.close(fig)
        plt.show()
    else:
        fig.savefig(os.path.join(output_folder, output_figure_name + '.pdf'), bbox_inches='tight', pad_inches=0)