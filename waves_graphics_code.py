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


data_folder = r'C:\Users\PC\PycharmProjects\Thin_Films_Project\Daniel_figure_codes\data'
correl_examples_name = 'correl_data.npz'

output_folder = r'C:\Users\PC\PycharmProjects\Thin_Films_Project\figures'
output_figure_name = 'waves_figure'
show_or_save = False # None for nothing, True for show, False for save
start_indx = None
end_indx = None

def get_bboxes(axes):
    bboxes = []
    tightbboxes = []
    for i in range(5):
        bboxes.append([])
        for j in range(4):
            bboxes[i].append(axes[i][j].get_position())
    for i in range(5):
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

def subtraction(image_original, image_prediction):
  d = max(image_original.shape[1], image_prediction.shape[1])
  h = min(image_original.shape[0], image_prediction.shape[0])
  sub1 = np.zeros((h,d))
  sub1[:,:image_original.shape[1]] = image_original[:h,:]
  sub2 = np.zeros((h,d))
  sub2[:,:image_prediction.shape[1]] = image_prediction[:h,:]
  sub = np.abs(sub1-sub2)
  sub = (sub > 1) * 255
  sub[:,min(image_original.shape[1], image_prediction.shape[1]):] = 255
  return sub

def load_results(base_dir):
    file_name1 = 'layers5_DNN_examples.npz'
    res1 = dict(np.load(os.path.join(base_dir, file_name1)))
    file_name2 = 'layers5_Transformer_examples.npz'
    res2 = dict(np.load(os.path.join(base_dir, file_name2)))
    file_name3 = 'unknown_layers_DNN_examples.npz'
    res3 = dict(np.load(os.path.join(base_dir, file_name3)))
    file_name4 = 'unknown_layers_Transformer_examples.npz'
    res4 = dict(np.load(os.path.join(base_dir, file_name4)))

    file_name1 = 'layers5_DNN_filters.npz'
    filter1 = dict(np.load(os.path.join(base_dir, file_name1)))
    file_name2 = 'layers5_Transformer_filters.npz'
    filter2 = dict(np.load(os.path.join(base_dir, file_name2)))
    file_name3 = 'unknown_layers_DNN_filters.npz'
    filter3 = dict(np.load(os.path.join(base_dir, file_name3)))
    file_name4 = 'unknown_layers_Transformer_filters.npz'
    filter4 = dict(np.load(os.path.join(base_dir, file_name4)))

    index1 = 40 # 5layers_DNN
    index2 = index1 # 5layers_Transformer
    index3 = 53 # unknown_layers_DNN
    index4 = index3 # unknown_layers_Transformer

    res = {}

    res['t_vec'] = res1['t_vec']

    res['layers5_DNN_t_original'] = res1['unknown_layers_DNN_t_original_example' + str(index1)]
    res['layers5_DNN_t_prediction'] = res1['unknown_layers_DNN_t_prediction_example' + str(index1)]
    res['layers5_DNN_r_original'] = res1['unknown_layers_DNN_r_original_example' + str(index1)]
    res['layers5_DNN_r_prediction'] = res1['unknown_layers_DNN_r_prediction_example' + str(index1)]
    res['layers5_DNN_filter'] = filter1['subtraction' + str(index1)]
    res['layers5_DNN_filter_original'] = filter1['original' + str(index1)]
    res['layers5_DNN_filter_prediction'] = filter1['prediction' + str(index1)]

    res['layers5_Transformer_t_original'] = res2['unknown_layers_Transformer_t_original_example' + str(index2)]
    res['layers5_Transformer_t_prediction'] = res2['unknown_layers_Transformer_t_prediction_example' + str(index2)]
    res['layers5_Transformer_r_original'] = res2['unknown_layers_Transformer_r_original_example' + str(index2)]
    res['layers5_Transformer_r_prediction'] = res2['unknown_layers_Transformer_r_prediction_example' + str(index2)]
    res['layers5_Transformer_filter'] = filter2['subtraction' + str(index2)]
    res['layers5_Transformer_filter_original'] = filter2['original' + str(index2)]
    res['layers5_Transformer_filter_prediction'] = filter2['prediction' + str(index2)]

    res['unknown_layers_DNN_t_original'] = res3['unknown_layers_DNN_t_original_example' + str(index3)]
    res['unknown_layers_DNN_t_prediction'] = res3['unknown_layers_DNN_t_prediction_example' + str(index3)]
    res['unknown_layers_DNN_r_original'] = res3['unknown_layers_DNN_r_original_example' + str(index3)]
    res['unknown_layers_DNN_r_prediction'] = res3['unknown_layers_DNN_r_prediction_example' + str(index3)]
    res['unknown_layers_DNN_filter'] = filter3['subtraction' + str(index3)]
    res['unknown_layers_DNN_filter_original'] = filter3['original' + str(index3)]
    res['unknown_layers_DNN_filter_prediction'] = filter3['prediction' + str(index3)]

    res['unknown_layers_Transformer_t_original'] = res4['unknown_layers_Transformer_t_original_example' + str(index4)]
    res['unknown_layers_Transformer_t_prediction'] = res4['unknown_layers_Transformer_t_prediction_example' + str(index4)]
    res['unknown_layers_Transformer_r_original'] = res4['unknown_layers_Transformer_r_original_example' + str(index4)]
    res['unknown_layers_Transformer_r_prediction'] = res4['unknown_layers_Transformer_r_prediction_example' + str(index4)]
    res['unknown_layers_Transformer_filter'] = subtraction(filter4['original' + str(index4)], filter4['prediction' + str(index4)])
    res['unknown_layers_Transformer_filter_original'] = filter4['original' + str(index4)]
    res['unknown_layers_Transformer_filter_prediction'] = filter4['prediction' + str(index4)]

    return res

def non_zero(x_vec, y_vec, min_num=0.003):
    i = 0
    j = len(x_vec) - 1
    while y_vec[i] < min_num:
        i = i + 1
    while y_vec[j] < min_num:
        j = j - 1
    # k = min(i , len(x_vec) - j)
    return x_vec[i:j], y_vec[i:j]

correl_examples = load_results(data_folder)


# def structure_correl(original, prediction):
#
#
# def structure_correl_histogram(original_vec, prediction_vec):
#     correl_vec = []
#     for i in range(original):


figure_size = (30, 20)

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

plots_width = 0.181
plots_spacing = 0.03
line_space = 0.0018

fig, axes = plt.subplots(5, 4, sharex=False, figsize=figure_size)
# fig, axes = plt.subplots(3, 2, sharex=True)
frenderer = fig.canvas.get_renderer()

####### set plots
axes[0][0].imshow(correl_examples['layers5_DNN_filter_original'])
axes[0][0].get_yaxis().set_visible(False)
axes[0][0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2g}'))
axes[0][0].get_xaxis().set_visible(False)

axes[0][1].imshow(correl_examples['layers5_Transformer_filter_original'])
axes[0][1].get_yaxis().set_visible(False)
axes[0][1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2g}'))
axes[0][1].get_xaxis().set_visible(False)

axes[0][2].imshow(correl_examples['unknown_layers_DNN_filter_original'])
axes[0][2].get_yaxis().set_visible(False)
axes[0][2].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2g}'))
axes[0][2].get_xaxis().set_visible(False)

axes[0][3].imshow(correl_examples['unknown_layers_Transformer_filter_original'])
axes[0][3].get_yaxis().set_visible(False)
axes[0][3].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2g}'))
axes[0][3].get_xaxis().set_visible(False)

axes[1][0].imshow(correl_examples['layers5_DNN_filter_prediction'])
axes[1][0].get_yaxis().set_visible(False)
axes[1][0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2g}'))
axes[1][0].get_xaxis().set_visible(False)

axes[1][1].imshow(correl_examples['layers5_Transformer_filter_prediction'])
axes[1][1].get_yaxis().set_visible(False)
axes[1][1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2g}'))
axes[1][1].get_xaxis().set_visible(False)

axes[1][2].imshow(correl_examples['unknown_layers_DNN_filter_prediction'])
axes[1][2].get_yaxis().set_visible(False)
axes[1][2].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2g}'))
axes[1][2].get_xaxis().set_visible(False)

axes[1][3].imshow(correl_examples['unknown_layers_Transformer_filter_prediction'])
axes[1][3].get_yaxis().set_visible(False)
axes[1][3].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2g}'))
axes[1][3].get_xaxis().set_visible(False)


axes[2][0].imshow(correl_examples['layers5_DNN_filter'])
axes[2][0].get_yaxis().set_visible(False)
axes[2][0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2g}'))
axes[2][0].get_xaxis().set_visible(False)
t_vec, layers5_DNN_t_original = non_zero(correl_examples['t_vec'], correl_examples['layers5_DNN_t_original'])
original_label = axes[3][0].plot(t_vec, layers5_DNN_t_original, gt_line_style, color=red, linewidth=plots_linewidth, label='Ground Truth')
t_vec, layers5_DNN_t_prediction = non_zero(correl_examples['t_vec'], correl_examples['layers5_DNN_t_prediction'])
prediction_label = axes[3][0].plot(t_vec, layers5_DNN_t_prediction, video_pred_line_style, color=blue, linewidth=plots_linewidth, label='Prediction')
axes[3][0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2g}'))
axes[3][0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*(10**12):.2g}'))
t_vec, layers5_DNN_r_original = non_zero(correl_examples['t_vec'], correl_examples['layers5_DNN_r_original'])
axes[4][0].plot(t_vec, layers5_DNN_r_original, gt_line_style, color=red, linewidth=plots_linewidth, label='Original')
t_vec, layers5_DNN_r_prediction = non_zero(correl_examples['t_vec'], correl_examples['layers5_DNN_r_prediction'])
axes[4][0].plot(t_vec, layers5_DNN_r_prediction, video_pred_line_style, color=blue, linewidth=plots_linewidth, label='Prediction')
axes[4][0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2g}'))
axes[4][0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*(10**12):.2g}'))

axes[2][1].imshow(correl_examples['layers5_Transformer_filter'])
axes[2][1].get_yaxis().set_visible(False)
axes[2][1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2g}'))
axes[2][1].get_xaxis().set_visible(False)
t_vec, layers5_Transformer_t_original = non_zero(correl_examples['t_vec'], correl_examples['layers5_Transformer_t_original'])
axes[3][1].plot(t_vec, layers5_Transformer_t_original, gt_line_style, color=red, linewidth=plots_linewidth, label='Original')
t_vec, layers5_Transformer_t_prediction = non_zero(correl_examples['t_vec'], correl_examples['layers5_Transformer_t_prediction'])
axes[3][1].plot(t_vec, layers5_Transformer_t_prediction, video_pred_line_style, color=blue, linewidth=plots_linewidth, label='Prediction')
axes[3][1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2g}'))
axes[3][1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*(10**12):.2g}'))
t_vec, layers5_Transformer_r_original = non_zero(correl_examples['t_vec'], correl_examples['layers5_Transformer_r_original'])
axes[4][1].plot(t_vec, layers5_Transformer_r_original, gt_line_style, color=red, linewidth=plots_linewidth, label='Original')
t_vec, layers5_Transformer_r_prediction = non_zero(correl_examples['t_vec'], correl_examples['layers5_Transformer_r_prediction'])
axes[4][1].plot(t_vec, layers5_Transformer_r_prediction, video_pred_line_style, color=blue, linewidth=plots_linewidth, label='Prediction')
axes[4][1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2g}'))
axes[4][1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*(10**12):.2g}'))

axes[4][0].set_xlabel('t[ps]')
axes[4][1].set_xlabel('t[ps]')
axes[4][1].set_ylabel('Intensity [a.u]')
axes[4][1].yaxis.set_label_position("right")
axes[3][1].set_ylabel('Intensity [a.u]')
axes[3][1].yaxis.set_label_position("right")
axes[3][2].set_ylabel('Intensity [a.u]')
axes[4][2].set_ylabel('Intensity [a.u]')
axes[3][1].yaxis.set_label_position('right')
axes[2][1].yaxis.set_ticks_position('right')
axes[3][1].yaxis.set_ticks_position('right')
axes[4][1].yaxis.set_ticks_position('right')
axes[0][0].set_title('(a) FC-DNN')
axes[0][1].set_title('(b) Attention')
axes[0][2].set_title('(c) FC-DNN')
axes[0][3].set_title('(d) Attention')

axes[2][2].imshow(correl_examples['unknown_layers_DNN_filter'])
axes[2][2].get_yaxis().set_visible(False)
axes[2][2].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2g}'))
axes[2][2].get_xaxis().set_visible(False)
t_vec, unknown_layers_DNN_t_original = non_zero(correl_examples['t_vec'], correl_examples['unknown_layers_DNN_t_original'])
axes[3][2].plot(t_vec, unknown_layers_DNN_t_original, gt_line_style, color=red, linewidth=plots_linewidth, label='Original')
t_vec, unknown_layers_DNN_t_prediction = non_zero(correl_examples['t_vec'], correl_examples['unknown_layers_DNN_t_prediction'])
axes[3][2].plot(t_vec, unknown_layers_DNN_t_prediction, video_pred_line_style, color=blue, linewidth=plots_linewidth, label='Prediction')
axes[3][2].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2g}'))
axes[3][2].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*(10**12):.2g}'))
t_vec, unknown_layers_DNN_r_original = non_zero(correl_examples['t_vec'], correl_examples['unknown_layers_DNN_r_original'])
axes[4][2].plot(t_vec, unknown_layers_DNN_r_original, gt_line_style, color=red, linewidth=plots_linewidth, label='Original')
t_vec, unknown_layers_DNN_r_prediction = non_zero(correl_examples['t_vec'], correl_examples['unknown_layers_DNN_r_prediction'])
axes[4][2].plot(t_vec, unknown_layers_DNN_r_prediction, video_pred_line_style, color=blue, linewidth=plots_linewidth, label='Prediction')
axes[4][2].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2g}'))
axes[4][2].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*(10**12):.2g}'))

axes[2][3].imshow(correl_examples['unknown_layers_Transformer_filter'])
axes[2][3].get_yaxis().set_visible(False)
axes[2][3].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2g}'))
axes[2][3].get_xaxis().set_visible(False)
t_vec, unknown_layers_Transformer_t_original = non_zero(correl_examples['t_vec'], correl_examples['unknown_layers_Transformer_t_original'])
axes[3][3].plot(t_vec, unknown_layers_Transformer_t_original, gt_line_style, color=red, linewidth=plots_linewidth, label='Original')
t_vec, unknown_layers_Transformer_t_prediction = non_zero(correl_examples['t_vec'], correl_examples['unknown_layers_Transformer_t_prediction'])
axes[3][3].plot(t_vec, unknown_layers_Transformer_t_prediction, video_pred_line_style, color=blue, linewidth=plots_linewidth, label='Prediction')
axes[3][3].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2g}'))
axes[3][3].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*(10**12):.2g}'))
t_vec, unknown_layers_Transformer_r_original = non_zero(correl_examples['t_vec'], correl_examples['unknown_layers_Transformer_r_original'])
axes[4][3].plot(t_vec, unknown_layers_Transformer_r_original, gt_line_style, color=red, linewidth=plots_linewidth, label='Original')
t_vec, unknown_layers_Transformer_r_prediction = non_zero(correl_examples['t_vec'], correl_examples['unknown_layers_Transformer_r_prediction'])
axes[4][3].plot(t_vec, unknown_layers_Transformer_r_prediction, video_pred_line_style, color=blue, linewidth=plots_linewidth, label='Prediction')
axes[4][3].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2g}'))
axes[4][3].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*(10**12):.2g}'))

axes[4][2].set_xlabel('t[ps]')
axes[4][3].set_xlabel('t[ps]')
axes[3][3].yaxis.set_label_position('right')
axes[2][3].yaxis.set_ticks_position('right')
axes[3][3].yaxis.set_ticks_position('right')
axes[4][3].yaxis.set_ticks_position('right')

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
    for j in range(5):
        axes[j][i].set_position([lefts[i], bboxes[j][i].y0, plots_width, bboxes[j][i].height])

[bboxes2, tightbboxes2] = get_bboxes(axes)
[leftpos_ylabel_width2, leftdiff_ylabel_width2, rightpos_ylabel_width2, rightdiff_ylabel_width2] = get_label_widths(bboxes2, tightbboxes2)
newcenterx = (tightbboxes2[1][1].x1 + tightbboxes2[1][2].x0)/2

###### set texts
### stages
textsy = [(bboxes2[i][0].y1 + bboxes2[i][0].y0 ) / 2 for i in range(5)]
# stage1txt = plt.text(0.5, textsy[0], 'Stage 1', va='center', ha='center', transform=fig.transFigure, bbox=dict(facecolor='none', edgecolor='red', pad=0))
# stage2txt = plt.text(0.5, textsy[1], 'Stage 2', va='center', ha='center', transform=fig.transFigure, bbox=dict(facecolor='none', edgecolor='red', pad=0))
# stage3txt = plt.text(0.5, textsy[2], 'Stage 3', va='center', ha='center', transform=fig.transFigure, bbox=dict(facecolor='none', edgecolor='red', pad=0))
titles_filter_t_r = ['Ground Truth Filter', 'Predicted Filter' , 'Overlap', 'Transmitted', 'Reflected']
stagetxts = [plt.text(newcenterx, textsy[i], titles_filter_t_r[i], va='center', ha='center', transform=fig.transFigure, fontdict={'size': stages_font_size}) for i in range(5)]
stagetxtsbboxes = [Bbox(fig.transFigure.inverted().transform(stagetext.get_window_extent(renderer=frenderer))) for stagetext in stagetxts]

### titles
random_title_centerx = (bboxes2[0][0].x1 + bboxes2[0][1].x0) / 2
snake_title_centerx = (bboxes2[0][2].x1 + bboxes2[0][3].x0) / 2
titlesy = max([tightbboxes2[0][i].y1 for i in range(4)]) + titles_height_above_axes
random_title = plt.text(random_title_centerx, titlesy, 'Data Set 1: 5 Layers', va='bottom', ha='center', transform=fig.transFigure)
snake_title = plt.text(snake_title_centerx, titlesy, 'Data Set 2: Unknown Amount Of Layers', va='bottom', ha='center', transform=fig.transFigure)
titlesbboxes = [Bbox(fig.transFigure.inverted().transform(title.get_window_extent(renderer=frenderer))) for title in [random_title, snake_title]]

# titlesy = textsy[2]
# random_title = plt.text(random_title_centerx, titlesy, 'Data Set 1: 5 Layers', va='bottom', ha='center', transform=fig.transFigure)
# snake_title = plt.text(snake_title_centerx, titlesy, 'Data Set 2: Unknown Amount Of Layers', va='bottom', ha='center', transform=fig.transFigure)
# titlesbboxes = [Bbox(fig.transFigure.inverted().transform(title.get_window_extent(renderer=frenderer))) for title in [random_title, snake_title]]

###### legend
legendy = max([titlebox.y1 for titlebox in titlesbboxes]) -39 * legend_height_above_titles
legend1 = axes[0][0].legend(loc=pos_legend_loc, bbox_to_anchor=(newcenterx, legendy), bbox_transform=fig.transFigure, handles=[original_label[0], prediction_label[0]],
                            frameon=legend_frameon, ncol=2, borderaxespad=0, borderpad=0.2, handletextpad=0.1, columnspacing=0.6, handlelength=1)
for line in legend1.get_lines():
    line.set_linewidth(3)
legend1bbox = Bbox(fig.transFigure.inverted().transform(legend1.get_tightbbox(renderer=frenderer)))

###### lines
linesy = [[stagetxtsbboxes[0].y1 + 35 * line_space, stagetxtsbboxes[0].y1 + line_space],
          [stagetxtsbboxes[0].y0 - line_space, stagetxtsbboxes[1].y1 + line_space],
          [stagetxtsbboxes[1].y0 - line_space, stagetxtsbboxes[2].y1 + line_space],
          [stagetxtsbboxes[2].y0 - line_space, legend1bbox.y1 + line_space],
          [legend1bbox.y0 - line_space, stagetxtsbboxes[3].y1 + line_space],
          [stagetxtsbboxes[3].y0 - line_space, stagetxtsbboxes[4].y1 + line_space],
          [stagetxtsbboxes[4].y0 - line_space, tightbboxes2[4][0].y0 + line_space],
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
        # fig_diff.savefig(os.path.join(output_folder, 'shape_sensing_results_diffs.png'), bbox_inches='tight', pad_inches=0)