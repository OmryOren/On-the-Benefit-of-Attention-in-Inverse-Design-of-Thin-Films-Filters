import numpy as np
from matplotlib import pyplot as plt, font_manager, patches
from matplotlib.transforms import Bbox
from matplotlib.ticker import FuncFormatter
import matplotlib.lines as lines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import os
from itertools import chain
from matplotlib.colors import ListedColormap

plt.rcParams['font.serif'] = ['Palatino Linotype']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 27
plt.rcParams['mathtext.fontset'] = 'cm'

data_amount = 10


data_folder = r'C:\Users\PC\PycharmProjects\Thin_Films_Project\Daniel_figure_codes\data\new_data'

output_folder = r'C:\Users\PC\PycharmProjects\Thin_Films_Project\figures'
output_figure_name = 'attention_scores_figure'
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

def arange(vec):
    a_vec = np.zeros((10,))
    for num in vec:
        index = int(num * 10)
        if index == 10:
            index = 9
        a_vec[index] = a_vec[index] + 1
    return a_vec

def load_results_i(base_dir):
    file_name1 = 'unknown_layers_Transformer_' + str(11) + '_examples.npz'
    res1 = dict(np.load(os.path.join(base_dir, file_name1)))
    file_name2 = 'unknown_layers_Transformer_' + str(11) + '_attention_score.npz'
    res2 = dict(np.load(os.path.join(base_dir, file_name2)))

    sample1 = 1
    sample2 = 2
    sample3 = 3
    sample4 = 4

    res = {}
    res['t_vec'] = res1['t_vec']
    res['GT1'] = res1['unknown_layers_Transformer_r_original_example' + str(sample1)]
    res['Pred1'] = res1['unknown_layers_Transformer_r_prediction_example' + str(sample1)]
    res['AS_Head1_1'] = res2['attention_scores_head1_' + str(sample1-1)]
    res['AS_Head2_1'] = res2['attention_scores_head2_' + str(sample1-1)]

    res['GT2'] = res1['unknown_layers_Transformer_r_original_example' + str(sample2)]
    res['Pred2'] = res1['unknown_layers_Transformer_r_prediction_example' + str(sample2)]
    res['AS_Head1_2'] = res2['attention_scores_head1_' + str(sample2-1)]
    res['AS_Head2_2'] = res2['attention_scores_head2_' + str(sample2-1)]

    res['GT3'] = res1['unknown_layers_Transformer_r_original_example' + str(sample3)]
    res['Pred3'] = res1['unknown_layers_Transformer_r_prediction_example' + str(sample3)]
    res['AS_Head1_3'] = res2['attention_scores_head1_' + str(sample3-1)]
    res['AS_Head2_3'] = res2['attention_scores_head2_' + str(sample3-1)]

    res['GT4'] = res1['unknown_layers_Transformer_r_original_example' + str(sample4)]
    res['Pred4'] = res1['unknown_layers_Transformer_r_prediction_example' + str(sample4)]
    res['AS_Head1_4'] = res2['attention_scores_head1_' + str(sample4-1)]
    res['AS_Head2_4'] = res2['attention_scores_head2_' + str(sample4-1)]

    return res

res = load_results_i(data_folder)

figure_size = (35, 13)

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

plots_width = 0.17
plots_spacing = 0.02
legend_font_size = "15"

fig, axes = plt.subplots(5, 4, sharex=False, figsize=figure_size, gridspec_kw={'height_ratios': [0.2, 1, 0.3, 0.2, 1]})
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
original_cmap = plt.get_cmap('winter')
# Define the range of values you want to keep
start_value = 0.2
end_value = 0.8
new_cmap = ListedColormap(original_cmap(np.linspace(start_value, end_value, 256)))

hm = np.tile(res['AS_Head1_1'], (100, 1))
heatmap = axes[0, 0].imshow(hm, cmap='jet', interpolation='nearest')
axes[0, 0].get_xaxis().set_visible(False)
axes[0, 0].get_yaxis().set_visible(False)

axes[1, 0].plot(res['t_vec']*1e12, res['GT1'], 'r', linewidth=lines_width, label='Ground Truth')
axes[1, 0].plot(res['t_vec']*1e12, res['Pred1'], 'b', linewidth=lines_width, ls='-',
                label='Prediction')
axes[0, 0].set_title('Sample (a): Head 1', fontsize="27")
axes[1, 0].set_xlabel('t[ps]')
axes[1, 0].legend(fontsize=legend_font_size)



hm = np.tile(res['AS_Head2_1'], (100, 1))
heatmap = axes[3, 0].imshow(hm, cmap='jet', interpolation='nearest')
axes[3, 0].get_xaxis().set_visible(False)
axes[3, 0].get_yaxis().set_visible(False)

axes[4, 0].plot(res['t_vec']*1e12, res['GT1'], 'r', linewidth=lines_width, label='Ground Truth')
axes[4, 0].plot(res['t_vec']*1e12, res['Pred1'], 'b', linewidth=lines_width, ls='-',
                label='Prediction')
axes[3, 0].set_title('Sample (a): Head 2', fontsize="27")
axes[4, 0].set_xlabel('t[ps]')
axes[4, 0].legend(fontsize=legend_font_size)

original_cmap = plt.get_cmap('summer')
# Define the range of values you want to keep
start_value = 0.2
end_value = 0.8
new_cmap = ListedColormap(original_cmap(np.linspace(start_value, end_value, 256)))


hm = np.tile(res['AS_Head1_2'], (100, 1))
heatmap = axes[0, 1].imshow(hm, cmap='jet', interpolation='nearest')
axes[0, 1].get_xaxis().set_visible(False)
axes[0, 1].get_yaxis().set_visible(False)

axes[1, 1].plot(res['t_vec']*1e12, res['GT2'], 'r', linewidth=lines_width, label='Ground Truth')
axes[1, 1].plot(res['t_vec']*1e12, res['Pred2'], 'b', linewidth=lines_width, ls='-',
                label='Prediction')
axes[0, 1].set_title('Sample (b): Head 1', fontsize="27")
axes[1, 1].set_xlabel('t[ps]')
axes[1, 1].legend(fontsize=legend_font_size)


hm = np.tile(res['AS_Head2_2'], (100, 1))
heatmap = axes[3, 1].imshow(hm, cmap='jet', interpolation='nearest')
axes[3, 1].get_xaxis().set_visible(False)
axes[3, 1].get_yaxis().set_visible(False)

axes[4, 1].plot(res['t_vec']*1e12, res['GT2'], 'r', linewidth=lines_width, label='Ground Truth')
axes[4, 1].plot(res['t_vec']*1e12, res['Pred2'], 'b', linewidth=lines_width, ls='-',
                label='Prediction')
axes[3, 1].set_title('Sample (b): Head 2', fontsize="27")
axes[4, 1].set_xlabel('t[ps]')
axes[4, 1].legend(fontsize=legend_font_size)


hm = np.tile(res['AS_Head1_3'], (100, 1))
heatmap = axes[0, 2].imshow(hm, cmap='jet', interpolation='nearest')
axes[0, 2].get_xaxis().set_visible(False)
axes[0, 2].get_yaxis().set_visible(False)

axes[1, 2].plot(res['t_vec']*1e12, res['GT3'], 'r', linewidth=lines_width, label='Ground Truth')
axes[1, 2].plot(res['t_vec']*1e12, res['Pred3'], 'b', linewidth=lines_width, ls='-',
                label='Prediction')
axes[0, 2].set_title('Sample (c): Head 1', fontsize="27")
axes[1, 2].set_xlabel('t[ps]')
axes[1, 2].legend(fontsize=legend_font_size)


hm = np.tile(res['AS_Head2_3'], (100, 1))
heatmap = axes[3, 2].imshow(hm, cmap='jet', interpolation='nearest')
axes[3, 2].get_xaxis().set_visible(False)
axes[3, 2].get_yaxis().set_visible(False)

axes[4, 2].plot(res['t_vec']*1e12, res['GT3'], 'r', linewidth=lines_width, label='Ground Truth')
axes[4, 2].plot(res['t_vec']*1e12, res['Pred3'], 'b', linewidth=lines_width, ls='-',
                label='Prediction')
axes[3, 2].set_title('Sample (c): Head 2', fontsize="27")
axes[4, 2].set_xlabel('t[ps]')
axes[4, 2].legend(fontsize=legend_font_size)



hm = np.tile(res['AS_Head1_4'], (100, 1))
heatmap = axes[0, 3].imshow(hm, cmap='jet', interpolation='nearest')
axes[0, 3].get_xaxis().set_visible(False)
axes[0, 3].get_yaxis().set_visible(False)
axes[1, 3].plot(res['t_vec']*1e12, res['GT4'], 'r', linewidth=lines_width, label='Ground Truth')
axes[1, 3].plot(res['t_vec']*1e12, res['Pred4'], 'b', linewidth=lines_width, ls='-',
                label='Prediction')
axes[0, 3].set_title('Sample (d): Head 1', fontsize="27")
axes[1, 3].set_xlabel('t[ps]')
axes[1, 3].legend(fontsize=legend_font_size)


hm = np.tile(res['AS_Head2_4'], (100, 1))
heatmap = axes[3, 3].imshow(hm, cmap='jet', interpolation='nearest')
axes[3, 3].get_xaxis().set_visible(False)
axes[3, 3].get_yaxis().set_visible(False)

axes[4, 3].plot(res['t_vec']*1e12, res['GT4'], 'r', linewidth=lines_width, label='Ground Truth')
axes[4, 3].plot(res['t_vec']*1e12, res['Pred4'], 'b', linewidth=lines_width, ls='-',
                label='Prediction')
axes[3, 3].set_title('Sample (d): Head 2', fontsize="27")
axes[4, 3].set_xlabel('t[ps]')
axes[4, 3].legend(fontsize=legend_font_size)

axes[2,0].axis('off')
axes[2,1].axis('off')
axes[2,2].axis('off')
axes[2,3].axis('off')

axes[1,0].set_ylabel('Intensity [a.u]')
axes[4,0].set_ylabel('Intensity [a.u]')

[bboxes, tightbboxes] = get_bboxes(axes)
[leftpos_ylabel_width, leftdiff_ylabel_width, rightpos_ylabel_width, rightdiff_ylabel_width] = get_label_widths(bboxes, tightbboxes)
# axes_width = (1 - (leftpos_ylabel_width + leftdiff_ylabel_width + rightpos_ylabel_width + rightdiff_ylabel_width + 2 * plots_spacing + 2 * plots_to_text_spacing + stage_texts_max_width)) / 4


lefts = [
    leftpos_ylabel_width,
    leftpos_ylabel_width + plots_width + plots_spacing,
    leftpos_ylabel_width + 2 * (plots_width + plots_spacing),
    leftpos_ylabel_width + 3 * (plots_width + plots_spacing)
]

for i in range(4):
    for j in range(2):
        axes[j][i].set_position([lefts[i], bboxes[j][i].y0, plots_width, bboxes[j][i].height])

[bboxes2, tightbboxes2] = get_bboxes(axes)
[leftpos_ylabel_width2, leftdiff_ylabel_width2, rightpos_ylabel_width2, rightdiff_ylabel_width2] = get_label_widths(bboxes2, tightbboxes2)
newcenterx = (tightbboxes2[1][1].x1 + tightbboxes2[1][2].x0)/2

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.25, 0.015, 0.5])
cbar = fig.colorbar(heatmap, cax=cbar_ax)
cbar.set_label('Attention Scores [%]', rotation=90)
fig.text(0.817, 0.76, '100%')
fig.text(0.821, 0.22, '0%')
cbar.set_ticks([])

pos1 = axes[0, 0].get_position() # get the original position
pos2 = [pos1.x0, pos1.y0 - 0.032,  pos1.width, pos1.height]
axes[0, 0].set_position(pos2) # set a new position

pos1 = axes[0, 1].get_position() # get the original position
pos2 = [pos1.x0, pos1.y0 - 0.032,  pos1.width, pos1.height]
axes[0, 1].set_position(pos2) # set a new position

pos1 = axes[0, 2].get_position() # get the original position
pos2 = [pos1.x0, pos1.y0 - 0.032,  pos1.width, pos1.height]
axes[0, 2].set_position(pos2) # set a new position

pos1 = axes[0, 3].get_position() # get the original position
pos2 = [pos1.x0, pos1.y0 - 0.032,  pos1.width, pos1.height]
axes[0, 3].set_position(pos2) # set a new position

pos1 = axes[3, 0].get_position() # get the original position
pos2 = [pos1.x0, pos1.y0 - 0.032,  pos1.width, pos1.height]
axes[3, 0].set_position(pos2) # set a new position

pos1 = axes[3, 1].get_position() # get the original position
pos2 = [pos1.x0, pos1.y0 - 0.032,  pos1.width, pos1.height]
axes[3, 1].set_position(pos2) # set a new position

pos1 = axes[3, 2].get_position() # get the original position
pos2 = [pos1.x0, pos1.y0 - 0.032,  pos1.width, pos1.height]
axes[3, 2].set_position(pos2) # set a new position

pos1 = axes[3, 3].get_position() # get the original position
pos2 = [pos1.x0, pos1.y0 - 0.032,  pos1.width, pos1.height]
axes[3, 3].set_position(pos2) # set a new position


if show_or_save is not None:
    if show_or_save:
        # plt.close(fig)
        plt.show()
    else:
        fig.savefig(os.path.join(output_folder, output_figure_name + '.pdf'), bbox_inches='tight', pad_inches=0)