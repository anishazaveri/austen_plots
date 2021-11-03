import plotnine as pn
import pandas as pd
import warnings


def get_color_palette(n: int):
    scale_fill = ['#D55E00', '#E69F00', '#0072B2',
                  '#009E73', '#F0E442', '#CC79A7', '#56B4E9']
    if n > len(scale_fill):
        filler = ['#000000'] * \
                 (n - len(scale_fill))
        scale_fill = scale_fill + filler
        warnings.warn("Number of covariates to plot is greater than the number of colors in the palette")
    return scale_fill


def modify_theme(p: pn.ggplot, bias: float):
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
