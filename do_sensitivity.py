import argparse
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path
from functools import reduce
from multiprocessing import Pool
from tqdm import tqdm


import numpy as np
import pandas as pd
import scipy as sp
from plotnine import *
from sklearn.metrics import mean_squared_error


def calc_beta_shapes(g, alpha):
    alpha_shape = g * ((1 / alpha) - 1)
    beta_shape = (1 - g) * ((1 / alpha) - 1)
    return alpha_shape, beta_shape


def calc_delta(alpha, g, bias):
    alpha_shape, beta_shape = calc_beta_shapes(g, alpha)
    bias_term = sp.special.digamma(alpha_shape + 1) - sp.special.digamma(
        beta_shape) - sp.special.digamma(alpha_shape) + sp.special.digamma(beta_shape + 1)
    delta = bias / np.mean(bias_term)
    return delta


def calc_Rsq(row, g, Q, t, y):
    alpha = row['alpha']
    delta = row['delta']
    alpha_shape, beta_shape = calc_beta_shapes(g, alpha)
    Rsq_num = delta**2 * np.mean(sp.special.polygamma(1, alpha_shape +
                                                      t) + sp.special.polygamma(1, beta_shape + (1 - t)))
    Rsq_den = mean_squared_error(y, Q)
    return Rsq_num / Rsq_den


def calc_Rsqhat(y, Qhat, Q):
    Rsqhat = (mean_squared_error(y, Qhat) - mean_squared_error(y, Q)) / \
        (mean_squared_error(y, Qhat))
    if Rsqhat < 0:
        Rsqhat = 0
    return Rsqhat


def calc_ahat(ghat, g):
    ahat = 1 - (np.mean(g * (1 - g)) / np.mean(ghat * (1 - ghat)))
    if ahat < 0:
        ahat = 0
    return ahat


def calc_ci(row, alpha_ci):
    boot_values = row[[col for col in list(row.index) if '_' in col]].values
    main_value = row[[col for col in list(
        row.index) if '_' not in col]].values[0]
    boot_values_d = boot_values - main_value
    lower_quant = np.quantile(boot_values_d, (1 - alpha_ci) / 2)
    upper_quant = np.quantile(boot_values_d, alpha_ci + ((1 - alpha_ci) / 2))
    lower_lim = main_value - upper_quant
    upper_lim = main_value - lower_quant
    if lower_lim < 0:
        lower_lim = 0
    if upper_lim > 1:
        upper_lim = 1
    return main_value, lower_lim, upper_lim


def get_color_palette(n, verbose):
    scale_fill = ['#D55E00', '#E69F00', '#0072B2',
                  '#009E73', '#F0E442', '#CC79A7', '#56B4E9']
    if n > len(scale_fill):
        filler = ['#000000'] * \
            (n - len(scale_fill))
        scale_fill = scale_fill + filler
        if verbose:
            print("Number of covariates to plot is greater than the number of colors in the palette - will plot remaining as black.")
    return scale_fill


def modify_theme(p, bias):
    p = p + theme_light() + theme(figure_size=(3.5, 3.5),
                                  legend_key=element_blank(),
                                  axis_title=element_text(size=12),
                                  axis_text=element_text(
                                      color='black', size=10),
                                  plot_title=element_text(size=12),
                                  legend_text=element_text(size=12)) + labs(x='Influence on treatment ' + r'($\mathregular{\alpha}$)',
                                                                            fill='',
                                                                            y='Influence on outcome ' +
                                                                            r'(partial $R^2$)',
                                                                            title=f"Bias = {bias}")
    return p


def modify_Rsq_plot(df, col):
    """
    Makes the largest Rsq that is greater than 1, equal to 1 for prettier plots. 
    """
    try:
        Rsq_to_modify = df.loc[df[col] > 1, col].idxmin()
        df.loc[Rsq_to_modify, col] = 1
    except ValueError:
        pass
    return df


def plot_sensitivity_graph(input_df, bias, covariate_params, do_att, verbose):
    """Creates a austen plot without bootstrap

    Args:
        input_df (pd.DataFrame): input dataframe
        bias (int): bias to be used in calculation
        covariate_params (dict): dictionary of covariate name:[ghat, qhat]
        do_att (bool): If ATE or ATT to be calculated
        verbose (bool): If verbose

    Returns:
        p: Plot
        plot_coords: DataFrame with co-ordinates of the resulting plot
        variable_importances_df: DataFrame with co-ordinates of the plots representing missing covariates

    """
    # Truncate input df if ATT is to be calculated
    if do_att == True:
        input_df_line = input_df[input_df['t'] == 1]
        if verbose:
            print('Calculating bias for ATT.....')
    else:
        input_df_line = input_df.copy(deep=True)
        if verbose:
            print('Calculating bias for ATE.....')

    # Calculate alpha, delta and Rsq
    plot_coords = pd.DataFrame({'alpha': np.arange(0.0001, 1, 0.0005)})
    plot_coords['delta'] = plot_coords['alpha'].apply(
        calc_delta, args=(input_df_line['g'], bias))
    plot_coords['Rsq'] = plot_coords.apply(
        calc_Rsq, axis=1, args=(input_df_line['g'], input_df_line['Q'], input_df_line['t'], input_df_line['y']))

    # make largest Rsq value=1
    plot_coords = modify_Rsq_plot(plot_coords, 'Rsq')

    # plot the line
    p = (ggplot(data=plot_coords, mapping=aes(x='alpha', y='Rsq'))
         + geom_line(color='#585858', size=1, na_rm=True)
         + scale_x_continuous(expand=[0, 0, 0, 0], limits=(-0.03, 1))
         + scale_y_continuous(expand=[0, 0, 0, 0], limits=(-0.03, 1))
         )
    p = modify_theme(p, bias)

    # if covariate information provided, calculate co-ordinates of the covariates
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

        # drop treatment from covariates to plot
        variable_importances_plot = variable_importances_df[
            variable_importances_df['covariate_name'] != 'treatment']
        scale_fill = get_color_palette(
            variable_importances_df.shape[0], verbose)

        # Add variables to plot
        p = p + geom_point(data=variable_importances_plot,
                           mapping=aes(x='ahat',
                                       y='Rsqhat',
                                       fill='covariate_name'),
                           color='black',
                           alpha=0.8,
                           size=4) + scale_fill_manual(scale_fill)
    elif covariate_params is None:
        variable_importances_df = None
    return p, plot_coords, variable_importances_df


def plot_bootstrap_sensitivity_graph(plot_coords, variable_coords, boot_plot_coords, boot_variable_coords, plot_variables, bootstrap_cutoff, bias):
    # merge into single df
    boot_plot_coords = reduce(lambda left, right: pd.merge(left, right, on='alpha',
                                                           how='inner'), boot_plot_coords)
    plot_coords = plot_coords.drop(columns='delta')
    boot_plot_coords = pd.merge(
        boot_plot_coords, plot_coords, how='left', on='alpha')

    # calculate confidence intervals for Rsq
    boot_plot_coords[['main', 'ci_lower', 'ci_upper']] = boot_plot_coords.filter(regex='^Rsq', axis=1).apply(
        calc_ci, axis=1, result_type='expand', args=(bootstrap_cutoff,))

    # make largest Rsq = 1
    boot_plot_coords = modify_Rsq_plot(boot_plot_coords, 'main')

    if plot_variables == 'both':
        # merge main plot co-ordinates with boot plot co-ordinates
        boot_variable_coords = reduce(lambda left, right: pd.merge(left, right, on='covariate_name',
                                                                   how='inner'), boot_variable_coords)

        boot_variable_coords = pd.merge(
            boot_variable_coords, variable_coords, how='left', on='covariate_name')

        # calc ci
        boot_variable_coords[['Rsqhat_main', 'Rsqhat_ci_lower', 'Rsqhat_ci_upper']] = boot_variable_coords.filter(regex='^Rsqhat', axis=1).apply(
            calc_ci, axis=1, result_type='expand', args=(bootstrap_cutoff,))

        boot_variable_coords[['ahat_main', 'ahat_ci_lower', 'ahat_ci_upper']] = boot_variable_coords.filter(regex='^ahat', axis=1).apply(
            calc_ci, axis=1, result_type='expand', args=(bootstrap_cutoff,))
        boot_variable_coords_plot = boot_variable_coords[
            boot_variable_coords['covariate_name'] != 'treatment']
    if plot_variables == 'main':
        variable_coords_plot = variable_coords[variable_coords['covariate_name'] != 'treatment']

    p = (ggplot(data=boot_plot_coords, mapping=aes(x='alpha', y='Rsq'))
         + geom_ribbon(aes(ymin='ci_lower', ymax='ci_upper'), fill='#D3D3D3')
         + geom_line(color='#585858', size=1, na_rm=True)
         + scale_x_continuous(expand=[0, 0, 0, 0], limits=(-0.05, 1))
         + scale_y_continuous(expand=[0, 0, 0, 0], limits=(-0.05, 1)))

    p = modify_theme(p, bias)
    if plot_variables == 'main':
        scale_fill = get_color_palette(variable_coords_plot.shape[0], False)
        p = p + geom_point(data=variable_coords_plot, mapping=aes(x='ahat', y='Rsqhat',
                                                                  fill='factor(covariate_name)'), inherit_aes=False, color='black', alpha=1, size=2.5, stroke=0.5)\
            + scale_fill_manual(scale_fill)
        return p, boot_plot_coords, variable_coords

    if plot_variables == 'both':
        scale_fill = get_color_palette(
            boot_variable_coords_plot.shape[0], False)
        p = p + geom_errorbar(data=boot_variable_coords_plot, mapping=aes(x='ahat_main', ymin='Rsqhat_ci_lower',
                                                                          ymax='Rsqhat_ci_upper'), inherit_aes=False, width=0.02, size=0.5)\
            + geom_errorbarh(data=boot_variable_coords_plot, mapping=aes(y='Rsqhat_main', xmin='ahat_ci_lower',
                                                                         xmax='ahat_ci_upper'), inherit_aes=False, height=0.02, size=0.5)\
            + geom_point(data=boot_variable_coords_plot, mapping=aes(x='ahat_main', y='Rsqhat_main',
                                                                     fill='factor(covariate_name)'), inherit_aes=False, color='black', alpha=1, size=2.5, stroke=0.5)\
            + scale_fill_manual(scale_fill)
        return p, boot_plot_coords, boot_variable_coords
    else:
        return p, boot_plot_coords, None


def do_single_sensitivity(input_csv, bias, output_dir, covariate_dir, do_att, command, verbose=True):
    """Creates and saves a single austen plot without bootstrap values

    Args:
        input_csv (str): path to input csv
        bias (int): bias to use for calculation
        output_dir (str): path to the output dir. The directory will be created if it doesn't exist
        covariate_dir (str): path to covariate dir containing individual csv files of parameters calculated after removing the covariate
        do_att (bool): Use att for calculation of sensitivity parameters
        command (str): Input command given
        verbose (bool, optional): If output should be verbose. Defaults to True.

    Returns:
        plot_coords: DataFrame with co-ordinates of the resulting plot
        variable_importances_df: DataFrame with co-ordinates of the plots representing missing covariates
    """
    input_df = pd.read_csv(input_csv)

    # create a dictionary of covariate name:[ghat, qhat]
    if covariate_dir is not None:
        covariate_params = defaultdict(list)
        for f in glob.glob(os.path.join(covariate_dir, '*.csv')):
            name = Path(f).stem
            df = pd.read_csv(f)
            covariate_params[name] = [df['g'].values, df['Q'].values]
    else:
        covariate_params = None

    p, plot_coords, variable_importances_df = plot_sensitivity_graph(
        input_df, bias, covariate_params, do_att, verbose)
    frac_bad_rows = len(
        plot_coords[plot_coords['Rsq'] >= 1]) / len(plot_coords)
    if (frac_bad_rows > 0.5) and verbose:
        print('Warning: Bias this large may not be compatible with the data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    p.save(os.path.join(output_dir, 'austen_plot.png'),
           dpi=500, verbose=False)
    with open(os.path.join(output_dir, 'austen_plot_coordinates.csv'), 'w+', newline='\n') as file:
        file.write(f'#{command}\n')
        plot_coords.to_csv(file, index=False)
    if variable_importances_df is not None:
        with open(os.path.join(output_dir, 'variable_importances.csv'), 'w+', newline='\n') as file:
            file.write(f'#{command}\n')
            variable_importances_df.to_csv(file, index=False)
    return plot_coords, variable_importances_df


def has_none(l):
    for item in l:
        if item is None:
            return True
    return False


def get_bootstrap_folder_names(subdir, input_name, covariate_dir_name, output_dir_name):
    input_csv = os.path.join(subdir, input_name)
    if covariate_dir_name is not None:
        boot_covariate_dir = os.path.join(subdir, covariate_dir_name)
        if not os.path.exists(boot_covariate_dir):
            boot_covariate_dir = None
    else:
        boot_covariate_dir = None
    boot_output_dir = os.path.join(subdir, output_dir_name)
    return input_csv, boot_output_dir, boot_covariate_dir


def do_bootstrap_sensitivity(n, subdir_list, input_name, covariate_dir_name, output_dir_name, command, args):
    """Calculates and saves a single sensitivity plot for a single bootstrapped dataset. 

    Args:
        n (int): iteration numbers
        subdir_list (str): list of paths to input subdirs
        input_name (str): name of input_csv
        covariate_dir_name (str): name of covariate dir
        output_dir_name (str): name of output dir
        command (str): input command given
        args (): input arguments

    Returns:
        plot_coords: DataFrame with co-ordinates of the resulting plot
        variable_coords: DataFrame with co-ordinates of the plots representing missing covariates. Is None if no covariate information was passed

    """
    subdir = subdir_list[n]
    input_csv, boot_output_dir, boot_covariate_dir = get_bootstrap_folder_names(
        subdir, input_name, covariate_dir_name, output_dir_name)
    plot_coords, variable_coords = do_single_sensitivity(input_csv, args.bias, boot_output_dir,
                                                         boot_covariate_dir, args.do_att, command, verbose=False)
    # rename columns to had a suffix of the iteration number
    plot_coords = plot_coords.drop(columns='delta').rename(
        columns={'Rsq': f"Rsq_{n}"})
    if variable_coords is not None:
        variable_coords = variable_coords.rename(
            columns={'ahat': f"ahat_{n}", 'Rsqhat': f'Rsqhat_{n}'})
        return plot_coords, variable_coords
    else:
        return plot_coords, None


def main():
    # collect args
    command = ' '.join(sys.argv[0:])
    myparser = argparse.ArgumentParser()
    myparser.add_argument('-i', '--input_csv', required=True)
    myparser.add_argument('-b', '--bias', required=True, type=float)
    myparser.add_argument('-o', '--output_dir', required=True)
    myparser.add_argument('-cov', '--covariate_dir',
                          required=False, default=None)
    myparser.add_argument('-boot', '--bootstrap_dir',
                          required=False, default=None)
    myparser.add_argument('-cut', '--bootstrap_cutoff',
                          required=False, type=float, default=0.95)
    myparser.add_argument('-do_att', '--do_att', required=False,
                          default=False, type=bool)
    myparser.add_argument('-multi', '--multi',
                          required=False, default=1, type=int)
    args = myparser.parse_args()

    # plot single sensitivity without bootstrap
    main_plot_coords, main_variable_coords = do_single_sensitivity(args.input_csv, args.bias, args.output_dir,
                                                                   args.covariate_dir, args.do_att, command)

    # if bootstrap values provided
    if args.bootstrap_dir is not None:
        # extract names of input_csv, covariate dir and output dir
        input_name = os.path.basename(os.path.normpath(args.input_csv))
        if args.covariate_dir is not None:
            covariate_dir_name = os.path.basename(
                os.path.normpath(args.covariate_dir))
        else:
            covariate_dir_name = None
        output_dir_name = os.path.basename(os.path.normpath(args.output_dir))

        # create paths of input subdirectories
        subdir_list = [os.path.join(args.bootstrap_dir, subdir)
                       for subdir in os.listdir(args.bootstrap_dir)]
        print('Calculating outputs for individual bootstrapped datasets')
        # Create pool for multiprocessing
        pool = Pool(args.multi)
        # Create progress bar
        pbar = tqdm(total=len(subdir_list))

        # Calculate bootstrap co-ordinates for all provided bootstrap values and combine them into a list of dataframes
        res = [pool.apply_async(do_bootstrap_sensitivity, args=(
            n, subdir_list), kwds={'input_name': input_name, 'covariate_dir_name': covariate_dir_name, 'output_dir_name': output_dir_name, 'command': command, 'args': args}, callback=lambda _: pbar.update(1)) for n in range(len(subdir_list))]
        boot_plot_coords, boot_variable_coords = list(
            zip(*[pool.get() for pool in res]))
        pbar.close()
        boot_plot_coords = list(boot_plot_coords)
        boot_variable_coords = list(boot_variable_coords)

        # which variables to plot?
        if args.covariate_dir is None:
            plot_variables = 'none'
        else:
            if has_none(boot_variable_coords):
                plot_variables = 'main'
            else:
                plot_variables = 'both'
        print('Plotting combined bootstrap graph')

        p, plot_coords_out, variable_coords_out = plot_bootstrap_sensitivity_graph(
            main_plot_coords, main_variable_coords, boot_plot_coords, boot_variable_coords, plot_variables, args.bootstrap_cutoff, args.bias)

        # Save bootstrap output files
        p.save(os.path.join(args.output_dir,
                            'austen_plot_bootstrap.png'), dpi=500, verbose=False)

        with open(os.path.join(args.output_dir, 'austen_plot_coordinates_bootstrap.csv'), 'w+', newline='\n') as file:
            file.write(f'#{command}\n')
            plot_coords_out.to_csv(file, index=False)
        if variable_coords_out is not None:
            with open(os.path.join(args.output_dir, 'variable_importances_bootstrap.csv'), 'w+', newline='\n') as file:
                file.write(f'#{command}\n')
                variable_coords_out.to_csv(file, index=False)

    return None


if __name__ == "__main__":
    main()
