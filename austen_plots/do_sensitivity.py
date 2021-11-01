import glob
import os
from collections import defaultdict
from pathlib import Path
from functools import reduce
import numpy as np
import pandas as pd
import scipy as sp
import plotnine as pn
from tqdm import tqdm
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
    Rsq_num = delta ** 2 * np.mean(sp.special.polygamma(1, alpha_shape +
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
            print(
                "Number of covariates to plot is greater than the number of colors in the palette - will plot remaining as black.")
    return scale_fill


def modify_theme(p, bias):
    p = p + pn.theme_light() + pn.theme(figure_size=(3.5, 3.5),
                                  legend_key=pn.element_blank(),
                                  axis_title=pn.element_text(size=12),
                                  axis_text=pn.element_text(
                                      color='black', size=10),
                                  plot_title=pn.element_text(size=12),
                                  legend_text=pn.element_text(size=12)) + pn.labs(
        x='Influence on treatment ' + r'($\mathregular{\alpha}$)',
        fill='',
        y='Influence on outcome ' +
          r'(partial $R^2$)',
        title=f"Bias = {bias}")
    return p


def modify_Rsq_plot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Makes the smallest Rsq that is greater than 1, equal to 1 for to avoid discontinuous lines.
    """
    try:
        Rsq_to_modify = df.loc[df[col] > 1, col].idxmin()
        df.loc[Rsq_to_modify, col] = 1
    except ValueError:
        pass
    return df


def calculate_sensitivity_graph(input_df, bias, covariate_params, do_att, verbose):
    # Truncate input df if ATT is to be calculated
    if do_att:
        input_df_plot = input_df[input_df['t'] == 1]
    else:
        input_df_plot = input_df.copy(deep=True)
    if verbose:
        print(f"Calculating bias for {'ATT' if do_att else 'ATE'}...")

    # Calculate alpha, delta and Rsq
    plot_coords = pd.DataFrame({'alpha': np.arange(0.0001, 1, 0.0005)})
    plot_coords['delta'] = plot_coords['alpha'].apply(
        calc_delta, args=(input_df_plot['g'], bias))
    plot_coords['Rsq'] = plot_coords.apply(
        calc_Rsq, axis=1, args=(input_df_plot['g'], input_df_plot['Q'], input_df_plot['t'], input_df_plot['y']))

    # make largest Rsq value=1
    plot_coords = modify_Rsq_plot(plot_coords, 'Rsq')

    # if covariate information provided, calculate co-ordinates of the covariates
    if covariate_params:
        variable_importances = []
        for k, v in covariate_params.items():
            variable_importances_sub = {'covariate_name': k,
                                        'ahat': np.nan if k == 'treatment' else calc_ahat(v[0], input_df['g']),
                                        'Rsqhat': calc_Rsqhat(input_df['y'], v[1], input_df['Q'])}
            variable_importances.append(variable_importances_sub)
        variable_importances_df = pd.DataFrame(variable_importances)

    else:
        variable_importances_df = None

    return plot_coords, variable_importances_df


def plot_sensitivity_graph(plot_coords, variable_importances_df, bias, verbose):
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
    # plot the line
    p = (pn.ggplot(data=plot_coords, mapping=pn.aes(x='alpha', y='Rsq'))
         + pn.geom_line(color='#585858', size=1, na_rm=True)
         + pn.scale_x_continuous(expand=[0, 0, 0, 0], limits=(-0.03, 1))
         + pn.scale_y_continuous(expand=[0, 0, 0, 0], limits=(-0.03, 1))
         )
    p = modify_theme(p, bias)

    if variable_importances_df is not None:
        # drop treatment from covariates to plot
        variable_importances_plot = variable_importances_df[
            variable_importances_df['covariate_name'] != 'treatment']
        scale_fill = get_color_palette(
            variable_importances_plot.shape[0], verbose)

        # Add variables to plot
        p = p + pn.geom_point(data=variable_importances_plot,
                           mapping=pn.aes(x='ahat',
                                       y='Rsqhat',
                                       fill='covariate_name'),
                           color='black',
                           alpha=0.8,
                           size=4) + pn.scale_fill_manual(scale_fill)

    return p


def calculate_bootstrap_sensitivity_graph(plot_coords, variable_coords, boot_plot_coords, boot_variable_coords,
                                          plot_variables, bootstrap_cutoff):
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
        boot_variable_coords[['Rsqhat_main', 'Rsqhat_ci_lower', 'Rsqhat_ci_upper']] = boot_variable_coords.filter(
            regex='^Rsqhat', axis=1).apply(
            calc_ci, axis=1, result_type='expand', args=(bootstrap_cutoff,))

        boot_variable_coords[['ahat_main', 'ahat_ci_lower', 'ahat_ci_upper']] = boot_variable_coords.filter(
            regex='^ahat', axis=1).apply(
            calc_ci, axis=1, result_type='expand', args=(bootstrap_cutoff,))

        return boot_plot_coords, boot_variable_coords

    elif plot_variables == 'main':
        return boot_plot_coords, variable_coords

    elif plot_variables == 'none':
        return boot_plot_coords, None


def plot_bootstrap_sensitivity_graph(plot_coords, variable_coords, plot_variables, bias):
    p = (pn.ggplot(data=plot_coords, mapping=pn.aes(x='alpha', y='Rsq'))
         + pn.geom_ribbon(pn.aes(ymin='ci_lower', ymax='ci_upper'), fill='#D3D3D3')
         + pn.geom_line(color='#585858', size=1, na_rm=True)
         + pn.scale_x_continuous(expand=[0, 0, 0, 0], limits=(-0.05, 1))
         + pn.scale_y_continuous(expand=[0, 0, 0, 0], limits=(-0.05, 1)))

    p = modify_theme(p, bias)

    if plot_variables != 'none':
        variable_coords_plot = variable_coords[variable_coords['covariate_name'] != 'treatment']
        scale_fill = get_color_palette(variable_coords_plot.shape[0], False)

        if plot_variables == 'main':
            variable_coords_plot = variable_coords_plot.rename(columns={'Rsqhat': 'Rsqhat_main', 'ahat': 'ahat_main'})

        if plot_variables == 'both':
            p = p + pn.geom_errorbar(data=variable_coords_plot, mapping=pn.aes(x='ahat_main', ymin='Rsqhat_ci_lower',
                                                                         ymax='Rsqhat_ci_upper'), inherit_aes=False,
                                  width=0.02, size=0.5) \
                + pn.geom_errorbarh(data=variable_coords_plot, mapping=pn.aes(y='Rsqhat_main', xmin='ahat_ci_lower',
                                                                        xmax='ahat_ci_upper'), inherit_aes=False,
                                 height=0.02, size=0.5)

        p = p + pn.geom_point(data=variable_coords_plot, mapping=pn.aes(x='ahat_main', y='Rsqhat_main',
                                                                  fill='factor(covariate_name)'), inherit_aes=False,
                           color='black', alpha=1, size=2.5, stroke=0.5) \
            + pn.scale_fill_manual(scale_fill)

    return p


def do_single_sensitivity(input_csv, bias, output_dir, covariate_dir, do_att, command, plot_graph=True, verbose=True):
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
    if covariate_dir:
        covariate_params = defaultdict(list)
        for f in glob.glob(os.path.join(covariate_dir, '*.csv')):
            name = Path(f).stem
            df = pd.read_csv(f)
            covariate_params[name] = [df['g'].values, df['Q'].values]
    else:
        covariate_params = None

    plot_coords, variable_importances_df = calculate_sensitivity_graph(input_df, bias, covariate_params, do_att,
                                                                       verbose)

    if verbose:
        frac_bad_rows = len(
            plot_coords[plot_coords['Rsq'] >= 1]) / len(plot_coords)
        if (frac_bad_rows > 0.5):
            print('Warning: Bias this large may not be compatible with the data')

    if plot_graph:
        p = plot_sensitivity_graph(
            plot_coords, variable_importances_df, bias, verbose)
    else:
        p = None

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if plot_graph:
            p.save(os.path.join(output_dir, 'austen_plot.png'),
                   dpi=500, verbose=False)

        with open(os.path.join(output_dir, 'austen_plot_coordinates.csv'), 'w+', newline='\n') as file:
            file.write(f'#{command}\n')
            plot_coords.to_csv(file, index=False)
        if variable_importances_df is not None:
            with open(os.path.join(output_dir, 'variable_importances.csv'), 'w+', newline='\n') as file:
                file.write(f'#{command}\n')
                variable_importances_df.to_csv(file, index=False)

    return p, plot_coords, variable_importances_df


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


def do_single_bootstrap_sensitivity(n, subdir, input_name, covariate_dir_name, output_dir_name, bias, do_att, command):
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
    input_csv, boot_output_dir, boot_covariate_dir = get_bootstrap_folder_names(
        subdir, input_name, covariate_dir_name, output_dir_name)
    _, plot_coords, variable_coords = do_single_sensitivity(input_csv, bias, boot_output_dir,
                                                         boot_covariate_dir, do_att, command, plot_graph=False,
                                                         verbose=False)
    # rename columns to have a suffix of the iteration number
    plot_coords = plot_coords.drop(columns='delta').rename(
        columns={'Rsq': f"Rsq_{n}"})
    if variable_coords is not None:
        variable_coords = variable_coords.rename(
            columns={'ahat': f"ahat_{n}", 'Rsqhat': f'Rsqhat_{n}'})
        return plot_coords, variable_coords
    else:
        return plot_coords, None


def do_sensitivity(input_csv, bias, output_dir, covariate_dir=None, bootstrap_dir=None, bootstrap_cutoff=0.95,
                   do_att=False,
                   multi=1):
    # plot single sensitivity without bootstrap
    command = f'do_sensitivity.py --input_csv {input_csv} --bias {bias} --output_dir {output_dir} --covariate_dir {covariate_dir} --bootstrap_dir {bootstrap_dir} --bootstrap_dir {bootstrap_dir} --bootstrap_cutoff {bootstrap_cutoff}, --do_att {do_att} --multi {multi}'
    _, main_plot_coords, main_variable_coords = do_single_sensitivity(input_csv, bias, output_dir,
                                                                   covariate_dir, do_att, command)

    # if bootstrap values provided
    if bootstrap_dir is not None:
        # extract names of input_csv, covariate dir and output dir
        input_name = os.path.basename(os.path.normpath(input_csv))
        if covariate_dir is not None:
            covariate_dir_name = os.path.basename(
                os.path.normpath(covariate_dir))
        else:
            covariate_dir_name = None
        output_dir_name = os.path.basename(os.path.normpath(output_dir))

        # create paths of input subdirectories
        subdir_list = sorted([os.path.join(bootstrap_dir, subdir)
                       for subdir in os.listdir(bootstrap_dir) if not subdir.startswith('.')])
        print('Calculating outputs for individual bootstrapped datasets')

        boot_plot_coords = []
        boot_variable_coords = []

        for n, subdir in tqdm(enumerate(subdir_list)):
            plot_coords, variable_coords = do_single_bootstrap_sensitivity(n, subdir, input_name, covariate_dir_name,
                                                                    output_dir_name, bias, do_att, command)
            boot_plot_coords.append(plot_coords)
            boot_variable_coords.append(variable_coords)

        #
        # # Create pool for multiprocessing
        # pool = Pool(multi)
        # # Create progress bar
        # pbar = tqdm(total=len(subdir_list))
        #
        # # Calculate bootstrap co-ordinates for all provided bootstrap values and combine them into a list of dataframes
        # res = [pool.apply_async(do_bootstrap_sensitivity, args=(
        #     n,), kwds={'subdir_list': subdir_list, 'input_name': input_name, 'covariate_dir_name': covariate_dir_name,
        #                            'output_dir_name': output_dir_name, 'bias': bias, 'do_att': do_att,
        #                            'command': command},
        #                         callback=lambda _: pbar.update(1)) for n in range(len(subdir_list))]
        #
        # boot_plot_coords, boot_variable_coords = list(
        #     zip(*[p.get() for p in res]))
        # pbar.close()
        # boot_plot_coords = list(boot_plot_coords)
        # boot_variable_coords = list(boot_variable_coords)

        # which variables to plot?
        if covariate_dir is None:
            plot_variables = 'none'
        else:
            if has_none(boot_variable_coords):
                plot_variables = 'main'
            else:
                plot_variables = 'both'
        print('Plotting combined bootstrap graph')

        combined_plot_coords, combined_variable_coords = calculate_bootstrap_sensitivity_graph(main_plot_coords,
                                                                                               main_variable_coords,
                                                                                               boot_plot_coords,
                                                                                               boot_variable_coords,
                                                                                               plot_variables,
                                                                                               bootstrap_cutoff)

        p = plot_bootstrap_sensitivity_graph(combined_plot_coords, combined_variable_coords, plot_variables, bias)

        # Save bootstrap output files
        p.save(os.path.join(output_dir,
                            'austen_plot_bootstrap.png'), dpi=500, verbose=False)

        with open(os.path.join(output_dir, 'austen_plot_coordinates_bootstrap.csv'), 'w+', newline='\n') as file:
            file.write(f'#{command}\n')
            combined_plot_coords.to_csv(file, index=False)
        if combined_variable_coords is not None:
            with open(os.path.join(output_dir, 'variable_importances_bootstrap.csv'), 'w+', newline='\n') as file:
                file.write(f'#{command}\n')
                combined_variable_coords.to_csv(file, index=False)



do_sensitivity('example/input_df.csv', 2, 'example/sensitivity_output', 'example/covariates', 'example/bootstrap')