import logging
import matplotlib.pyplot as plt
import seaborn as sns


# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO, 
    format="%(name)s - %(asctime)s - %(levelname)s - %(message)s"
)

# Get the logger for this script (one logger per module)
logger = logging.getLogger(__name__)


def set_plotting_config(fontsize: int = 10, aspect_ratio: float = 1.618, width_fraction: float = 1.0, text_usetex: bool = True,
                        latex_text_width_in_pt: int = 468):
    """ Set global plotting configuration for Matplotlib and Seaborn. 
    
    Args:   
        fontsize (int, optional): Font size for text elements. Defaults to 10.
        aspect_ratio (float, optional): Aspect ratio of the figure. Defaults to 1.618.
        width_fraction (float, optional): Fraction of the text width to use for the figure width in latex. Defaults to 1.0.
        text_usetex (bool, optional): Use LaTeX for text rendering. Defaults to True.
        latex_text_width_in_pt (int, optional): LaTeX text width in points. Defaults to 468 (from Physical Review B).
    """
    latex_text_width_in_in = width_fraction * latex_text_width_in_pt / 72  # Convert pt to inches
    scale_factor = width_fraction + 0.25  if width_fraction < 1.0 else 1.0


    # Set Matplotlib rcParams
    plt.rcParams.update({
        "font.family": "serif" if text_usetex else "sans-serif",
        "text.usetex": text_usetex,
        'font.size': fontsize * scale_factor,  
        'text.latex.preamble': r'\usepackage{lmodern}',
        "axes.labelsize": fontsize * scale_factor,
        "axes.titlesize": fontsize * scale_factor,
        "xtick.labelsize": (fontsize - 2) * scale_factor,
        "ytick.labelsize": (fontsize - 2) * scale_factor,
        "legend.fontsize": (fontsize - 2) * scale_factor,
        "axes.linewidth": 0.8 * scale_factor,
        "lines.linewidth": 0.8 * scale_factor,
        "grid.linewidth": 0.6 * scale_factor,
        'lines.markersize': 5 * width_fraction,
        "figure.autolayout": True,
        "figure.figsize": (latex_text_width_in_in, latex_text_width_in_in / aspect_ratio),
    }) 

    # Set color palette
    sns.set_palette("colorblind")