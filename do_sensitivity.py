import argparse
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
from plotnine import *
from sklearn.metrics import mean_squared_error


def calc_beta_shapes(g, alpha):
    alpha_shape = g*((1/alpha)-1)
    beta_shape = (1-g)*((1/alpha)-1)
    return alpha_shape, beta_shape


def calc_delta(alpha, g, bias):
    alpha_shape, beta_shape = calc_beta_shapes(g, alpha)
    bias_term = sp.special.digamma(alpha_shape+1) - sp.special.digamma(
        beta_shape) - sp.special.digamma(alpha_shape) + sp.special.digamma(beta_shape+1)
    delta = bias/np.mean(bias_term)
    return delta


def calc_Rsq(row, g, Q, t, y):
    alpha = row['alpha']
    delta = row['delta']
    alpha_shape, beta_shape = calc_beta_shapes(g, alpha)
    Rsq_num = delta**2*np.mean(sp.special.polygamma(1, alpha_shape +
                                                    t) + sp.special.polygamma(1, beta_shape+(1-t)))
    Rsq_den = mean_squared_error(y, Q)
    return Rsq_num/Rsq_den


def calc_Rsqhat(y, Qhat, Q):
    Rsqhat = (mean_squared_error(y, Qhat)-mean_squared_error(y, Q)) / \
        (mean_squared_error(y, Qhat))
    if Rsqhat < 0:
        Rsqhat = 0
    return Rsqhat


def calc_ahat(ghat, g):
    ahat = 1-(np.mean(g*(1-g))/np.mean(ghat*(1-ghat)))
    if ahat < 0:
        ahat = 0
    return ahat


def plot_sensitivity_graph(input_df, bias, covariate_params, do_att):
    # Calculate alpha, delta, Rsq
    if do_att == True:
        input_df_line = input_df[input_df['t'] == 1]
        print('Calculating bias for ATT.....')
    else:
        input_df_line = input_df.copy(deep=True)
        print('Calculating bias for ATE.....')
    plot_coords = pd.DataFrame({'alpha': np.arange(0.0001, 1, 0.0005)})
    plot_coords['delta'] = plot_coords['alpha'].apply(
        calc_delta, args=(input_df_line['g'], bias))
    plot_coords['Rsq'] = plot_coords.apply(
        calc_Rsq, axis=1, args=(input_df_line['g'], input_df_line['Q'], input_df_line['t'], input_df_line['y']))
    p = (ggplot(data=plot_coords, mapping=aes(x='alpha', y='Rsq'))
         + geom_line(color='#585858', size=1, na_rm=True)
         + theme_light()
         + theme(figure_size=(3.5, 3.5),
                 legend_key=element_blank(),
                 axis_title=element_text(size=12),
                 axis_text=element_text(color='black', size=10),
                 plot_title=element_text(size=12),
                 legend_text=element_text(size=12))
         + labs(x='Influence on treatment ' + r'($\mathregular{\alpha}$)',
                fill='',
                y='Influence on outcome ' +
                r'(partial $R^2$)',
                title=f"Bias = {bias}")
         + scale_x_continuous(expand=[0, 0, 0, 0], limits=(-0.03, 1))
         + scale_y_continuous(expand=[0, 0, 0, 0], limits=(-0.03, 1))
         )
    if covariate_params is not None:
        variable_importances = defaultdict(list)
        for k, v in covariate_params.items():
            variable_importances['covariate_name'].append(k)
            if k != 'treatment':
                variable_importances['ahat'].append(
                    calc_ahat(v[0], input_df['g']))
            else:
                variable_importances['ahat'].append(np.nan)
            variable_importances['Rsqhat'].append(
                calc_Rsqhat(input_df['y'], v[1], input_df['Q']))

        variable_importances_df = pd.DataFrame(variable_importances)
        variable_importances_plot = variable_importances_df[
            variable_importances_df['covariate_name'] != 'treatment']
        scale_fill = ['#D55E00', '#E69F00', '#0072B2',
                      '#009E73', '#F0E442', '#CC79A7', '#56B4E9']
        if variable_importances_plot.shape[0] > len(scale_fill):
            filler = ['#000000'] * \
                (variable_importances_plot.shape[0]-len(scale_fill))
            scale_fill = scale_fill + filler
            print("Number of covariates to plot is greater than the number of colors in the palette - will plot remaining as black.")
        p = p + geom_point(data=variable_importances_plot,
                           mapping=aes(x='ahat',
                                       y='Rsqhat',
                                       fill='covariate_name'),
                           color='black',
                           alpha=0.8,
                           size=4) + scale_fill_manual(scale_fill)
    elif covariate_params is None:
        variable_importances_df = pd.DataFrame()
    return p, plot_coords, variable_importances_df


def main():
    command = ' '.join(sys.argv[0:])
    myparser = argparse.ArgumentParser()
    myparser.add_argument('-input_csv', required=True)
    myparser.add_argument('-bias', required=True, type=float)
    myparser.add_argument('-output_dir', required=True)
    myparser.add_argument('-covariate_dir',
                          required=False, default=None)
    myparser.add_argument('-do_att', required=False,
                          default=False, type=bool)

    args = myparser.parse_args()
    Path(args.output_dir).mkdir(exist_ok=True)
    input_df = pd.read_csv(args.input_csv)
    if args.covariate_dir is not None:
        covariate_params = defaultdict(list)
        for f in glob.glob(os.path.join(args.covariate_dir, '*.csv')):
            name = Path(f).stem
            df = pd.read_csv(f)
            covariate_params[name] = [df['g'].values, df['Q'].values]
    else:
        covariate_params = None
    p, plot_coords, variable_importances_df = plot_sensitivity_graph(
        input_df, args.bias, covariate_params, args.do_att)
    frac_bad_rows = len(plot_coords[plot_coords['Rsq'] >= 1])/len(plot_coords)
    if frac_bad_rows > 0.5:
        print('Warning: Bias this large may not be compatible with the data')
    p.save(os.path.join(args.output_dir, 'austen_plot.png'),
           dpi=500, verbose=False)
    with open(os.path.join(args.output_dir, 'austen_plot_coordinates.csv'), 'w+', newline='\n') as file:
        file.write(f'#{command}\n')
        plot_coords.to_csv(file, index=False)
    with open(os.path.join(args.output_dir, 'variable_importances.csv'), 'w+', newline='\n') as file:
        file.write(f'#{command}\n')
        variable_importances_df.to_csv(file, index=False)
    print('Done.....')
    return None


if __name__ == "__main__":
    main()
