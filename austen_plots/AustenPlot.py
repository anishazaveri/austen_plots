import glob
import os
import warnings
from collections import defaultdict
from pathlib import Path
from functools import reduce
import numpy as np
import pandas as pd
from pandas import DataFrame
import scipy as sp
import plotnine as pn
from tqdm import tqdm
from typing import List, Tuple, Optional
from austen_plots.plot_helpers import get_color_palette, modify_theme, modify_Rsq_plot
from austen_plots.math_helpers import calc_Rsq, calc_delta, calc_ci, calc_Rsqhat, calc_ahat


# TODO: add treatment in covs and docstring
class AustenPlot:
    def __init__(self, input_df_path: str, covariate_dir_path: str = None, bootstrap_dir_path: str = None):
        """
        :param input_df_path: Path to input df which should be a .csv with the following columns:
            'g' (propensity_score), 'Q' (conditional expected outcome), 't' (treatment), 'y' (outcome)
        :param covariate_dir_path: Path to directory containing coviariates.
            Each file within the directory should be a "covariate_name.csv" with the same columns as input_df.
            The files should have 'g' and 'Q'  calculated from data where the covariate has been dropped before prediction
        :param bootstrap_dir_path: Path to directory containing bootstrapped data
            Each sub-directory under bootstrap_dir_path should have an input_df with the same name and format as that in input_df_path
            Each sub-directory may also optionally have a directory with covariates with the same name and format as covariate_dir_path
        """
        self.input_df_path = input_df_path
        self.input_name = os.path.basename(os.path.normpath(self.input_df_path))
        self.covariate_dir_path = covariate_dir_path
        self.covariate_dir_name = os.path.basename(
            os.path.normpath(self.covariate_dir_path)) if self.covariate_dir_path else None
        self.bootstrap_dir_path = bootstrap_dir_path
        if self.bootstrap_dir_path:
            self.bootstrap_subdir_list = sorted([os.path.join(self.bootstrap_dir_path, subdir)
                                                 for subdir in os.listdir(self.bootstrap_dir_path) if
                                                 not subdir.startswith('.')])  # ignore hidden files such as .DS_Store

    def _fit_single_sensitivity_analysis(self, input_df_path: str,
                                         bias: float,
                                         covariate_dir_path: str,
                                         do_att: bool,
                                         plot_graph: bool = True,
                                         verbose: bool = True) -> \
            Tuple[Optional[pn.ggplot], DataFrame, DataFrame]:
        input_df = pd.read_csv(input_df_path)
        if covariate_dir_path:
            covariate_params = {}
            for f in glob.glob(os.path.join(covariate_dir_path, '*.csv')):
                name = Path(f).stem
                cov_df = pd.read_csv(f)
                covariate_params[name] = {'g': cov_df['g'].values, 'Q': cov_df['Q'].values}
        else:
            covariate_params = None

        if verbose:
            print("Fitting main dataset")
        plot_coords, variable_importances_df = self._calculate_sensitivity_graph(input_df, bias, covariate_params,
                                                                                 do_att)

        frac_bad_rows = len(
            plot_coords[plot_coords['Rsq'] >= 1]) / len(plot_coords)
        if frac_bad_rows > 0.5:
            warnings.warn('Bias this large may not be compatible with the data')

        if plot_graph:
            p = self._plot_sensitivity_graph(
                plot_coords, variable_importances_df, bias)
        else:
            p = None

        return p, plot_coords, variable_importances_df

    def _calculate_sensitivity_graph(self, input_df: DataFrame, bias: float, covariate_params: dict, do_att: bool) -> \
            Tuple[DataFrame, Optional[DataFrame]]:
        # Truncate input df if ATT is to be calculated
        if do_att:
            input_df_plot = input_df[input_df['t'] == 1]
        else:
            input_df_plot = input_df.copy(deep=True)

        # Calculate alpha, delta and Rsq
        plot_coords = pd.DataFrame({'alpha': np.arange(0.0001, 1, 0.0005)})
        plot_coords['delta'] = plot_coords['alpha'].apply(
            calc_delta, args=(input_df_plot['g'], bias))
        plot_coords['Rsq'] = plot_coords.apply(
            calc_Rsq, axis=1, args=(input_df_plot['g'], input_df_plot['Q'], input_df_plot['t'], input_df_plot['y']))

        # make smallest Rsq value>1 to be equal to 1 to prevent plot discontinuity
        plot_coords = modify_Rsq_plot(plot_coords, 'Rsq')

        # if covariate information provided, calculate co-ordinates of the covariates
        # note that restricting to t==1 is not required here when do_att, see Appendix A in paper
        if covariate_params:
            variable_importances = []
            for k, v in covariate_params.items():
                variable_importances_sub = {'covariate_name': k,
                                            'ahat': np.nan if k == 'treatment' else calc_ahat(v['g'], input_df['g']),
                                            'Rsqhat': calc_Rsqhat(input_df['y'], v['Q'], input_df['Q'])}
                variable_importances.append(variable_importances_sub)
            variable_importances_df = pd.DataFrame(variable_importances)

        else:
            variable_importances_df = None

        return plot_coords, variable_importances_df

    def _plot_sensitivity_graph(self, plot_coords: DataFrame, variable_importances_df: DataFrame,
                                bias: float) -> pn.ggplot:
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
                variable_importances_plot.shape[0])

            # Add variables to plot
            p = p + pn.geom_point(data=variable_importances_plot,
                                  mapping=pn.aes(x='ahat',
                                                 y='Rsqhat',
                                                 fill='covariate_name'),
                                  color='black',
                                  alpha=0.8,
                                  size=4) + pn.scale_fill_manual(scale_fill)

        return p

    def _fit_single_bootstrap_sensitivity_analysis(self, subdir: str, bias: float, do_att: bool) -> Tuple[
        DataFrame, Optional[DataFrame]]:
        if self.covariate_dir_name is not None:
            boot_covariate_dir = os.path.join(subdir, self.covariate_dir_name)
            if not os.path.exists(boot_covariate_dir):
                boot_covariate_dir = None
        else:
            boot_covariate_dir = None

        _, plot_coords, variable_coords = self._fit_single_sensitivity_analysis(os.path.join(subdir, self.input_name),
                                                                                bias,
                                                                                boot_covariate_dir,
                                                                                do_att,
                                                                                plot_graph=False,
                                                                                verbose=False)
        # rename columns to have a suffix of the iteration number
        suffix = Path(subdir).stem
        plot_coords = plot_coords.drop(columns=['delta']).rename(columns={'Rsq': f"Rsq_{suffix}"})
        if variable_coords is not None:
            variable_coords = variable_coords.rename(
                columns={'ahat': f"ahat_{suffix}", 'Rsqhat': f'Rsqhat_{suffix}'})
            return plot_coords, variable_coords
        else:
            return plot_coords, None

    def _calculate_bootstrap_sensitivity_graph(self, main_plot_coords: DataFrame,
                                               main_variable_coords: DataFrame,
                                               boot_plot_coords: List[DataFrame],
                                               boot_variable_coords: List[DataFrame],
                                               plot_variables_condition: str,
                                               ci_cutoff: float) -> Tuple[DataFrame, Optional[DataFrame]]:
        # merge into single df
        boot_plot_coords = reduce(lambda left, right: pd.merge(left, right, on='alpha',
                                                               how='inner'), boot_plot_coords)
        main_plot_coords = main_plot_coords.drop(columns='delta')
        boot_plot_coords = pd.merge(
            boot_plot_coords, main_plot_coords, how='left', on='alpha')

        # calculate confidence intervals for Rsq
        boot_plot_coords[['main', 'ci_lower', 'ci_upper']] = boot_plot_coords.filter(regex='^Rsq', axis=1).apply(
            calc_ci, axis=1, result_type='expand', args=(ci_cutoff,))

        # make largest Rsq = 1
        boot_plot_coords = modify_Rsq_plot(boot_plot_coords, 'main')

        if plot_variables_condition == 'both':
            # merge main plot co-ordinates with boot plot co-ordinates
            boot_variable_coords = reduce(lambda left, right: pd.merge(left, right, on='covariate_name',
                                                                       how='inner'), boot_variable_coords)

            boot_variable_coords = pd.merge(
                boot_variable_coords, main_variable_coords, how='left', on='covariate_name')

            # calc ci for variables
            boot_variable_coords[['Rsqhat_main', 'Rsqhat_ci_lower', 'Rsqhat_ci_upper']] = boot_variable_coords.filter(
                regex='^Rsqhat', axis=1).apply(
                calc_ci, axis=1, result_type='expand', args=(ci_cutoff,))

            boot_variable_coords[['ahat_main', 'ahat_ci_lower', 'ahat_ci_upper']] = boot_variable_coords.filter(
                regex='^ahat', axis=1).apply(
                calc_ci, axis=1, result_type='expand', args=(ci_cutoff,))

            return boot_plot_coords, boot_variable_coords

        elif plot_variables_condition == 'main':
            return boot_plot_coords, main_variable_coords

        elif plot_variables_condition == 'none':
            return boot_plot_coords, None

    def _plot_bootstrap_sensitivity_graph(self, plot_coords: DataFrame, variable_coords: DataFrame,
                                          plot_variables_condition: str, bias: float) -> pn.ggplot:

        p = (pn.ggplot(data=plot_coords, mapping=pn.aes(x='alpha', y='Rsq'))
             + pn.geom_ribbon(pn.aes(ymin='ci_lower', ymax='ci_upper'), fill='#D3D3D3')
             + pn.geom_line(color='#585858', size=1, na_rm=True)
             + pn.scale_x_continuous(expand=[0, 0, 0, 0], limits=(-0.05, 1))
             + pn.scale_y_continuous(expand=[0, 0, 0, 0], limits=(-0.05, 1)))

        p = modify_theme(p, bias)

        if plot_variables_condition != 'none':
            variable_coords_plot = variable_coords[variable_coords['covariate_name'] != 'treatment']
            scale_fill = get_color_palette(variable_coords_plot.shape[0])

            if plot_variables_condition == 'main':
                variable_coords_plot = variable_coords_plot.rename(
                    columns={'Rsqhat': 'Rsqhat_main', 'ahat': 'ahat_main'})

            if plot_variables_condition == 'both':
                p = p + pn.geom_errorbar(data=variable_coords_plot,
                                         mapping=pn.aes(x='ahat_main', ymin='Rsqhat_ci_lower',
                                                        ymax='Rsqhat_ci_upper'), inherit_aes=False,
                                         width=0.02, size=0.5) \
                    + pn.geom_errorbarh(data=variable_coords_plot, mapping=pn.aes(y='Rsqhat_main', xmin='ahat_ci_lower',
                                                                                  xmax='ahat_ci_upper'),
                                        inherit_aes=False,
                                        height=0.02, size=0.5)

            p = p + pn.geom_point(data=variable_coords_plot, mapping=pn.aes(x='ahat_main', y='Rsqhat_main',
                                                                            fill='factor(covariate_name)'),
                                  inherit_aes=False,
                                  color='black', alpha=1, size=2.5, stroke=0.5) \
                + pn.scale_fill_manual(scale_fill)

        return p

    def fit(self, bias: float, do_bootstrap: bool = False, ci_cutoff: float = 0.95, do_att: bool = False) -> \
            Tuple[pn.ggplot, DataFrame, Optional[DataFrame]]:
        """
        Fits an Austen plot
        :param bias: The desired amount of bias to test for
        :param do_bootstrap: If bootstrapped data should be used for analysis
        :param ci_cutoff: Confidence interval cutoff if using bootstrapped data
        :param do_att: If ATT should be used instead of ATE
        :returns: Austen plot, plot_co-ordinates and optionally variable co-ordinates
        """
        # plot single sensitivity without bootstrap
        if not do_bootstrap:
            p, main_plot_coords, main_variable_coords = self._fit_single_sensitivity_analysis(self.input_df_path, bias,
                                                                                              self.covariate_dir_path,
                                                                                              do_att)
            return p, main_plot_coords, main_variable_coords

        else:
            if self.bootstrap_dir_path is None:
                raise ValueError("bootstrap_dir_path is not provided")

            else:
                _, main_plot_coords, main_variable_coords = self._fit_single_sensitivity_analysis(self.input_df_path,
                                                                                                  bias,
                                                                                                  self.covariate_dir_path,
                                                                                                  do_att,
                                                                                                  plot_graph=False)
                print('Fitting bootstrapped datasets')

                boot_plot_coords, boot_variable_coords = [], []
                for subdir in tqdm(self.bootstrap_subdir_list):
                    plot_coords, variable_coords = self._fit_single_bootstrap_sensitivity_analysis(subdir, bias,
                                                                                                   do_att)
                    boot_plot_coords.append(plot_coords)
                    boot_variable_coords.append(variable_coords)

                # which variables to plot?
                if not self.covariate_dir_path:
                    plot_variables_condition = 'none'
                    boot_variable_coords_filtered = None
                else:
                    boot_variable_coords_filtered = [v for v in boot_variable_coords if v is not None]
                    if len(boot_variable_coords_filtered) < 3:
                        warnings.warn("Need at last three bootstrapped covariates for plotting confidence intervals")
                        plot_variables_condition = 'main'
                    else:
                        plot_variables_condition = 'both'
                print('Plotting combined bootstrap graph')

                combined_plot_coords, combined_variable_coords = self._calculate_bootstrap_sensitivity_graph(
                    main_plot_coords,
                    main_variable_coords,
                    boot_plot_coords,
                    boot_variable_coords_filtered,
                    plot_variables_condition,
                    ci_cutoff)

                p = self._plot_bootstrap_sensitivity_graph(combined_plot_coords, combined_variable_coords,
                                                           plot_variables_condition, bias)

                return p, combined_plot_coords, combined_variable_coords
