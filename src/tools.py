import math
from plotly.io import write_image
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import plotly
import numpy as np

# from src.params import project_path, local_path
from src.params import writepath


# def _get_savename(savename):
#     if project_path.is_dir():
#         savepath = project_path / 'Dataset' / 'simluation_output'
#         savename = savepath / f'{savename}.pkl'
#
#     elif local_path.is_dir():
#         savename = local_path / f'{savename}.pkl'
#
#     else:
#         savename = Path.cwd() / f'{savename}.pkl'
#
#     return savename


# def save_data(data: dict, savename: str):
#     savefile = _get_savename(savename)
#     if not savefile.parent.is_dir():
#         savefile.parent.mkdir(parents=True)
#
#     with open(savefile.as_posix(), 'wb') as f:
#         pickle.dump(data, f, protocol=4)


# def load_data(savename):
#     savefile = _get_savename(savename)
#     with open(savefile.as_posix(), 'rb') as f:
#         data = pickle.load(f)
#     return data
#
#
# def is_saved(savename):
#     savefile = _get_savename(savename)
#     return savefile.is_file()


# def setup_environment():
#     req_file = project_path / 'Code' / 'requirements.txt'
#
#     process = subprocess.Popen(f'pip install --upgrade pip',
#             stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, shell=True)
#     output = process.communicate()
#
#     process = subprocess.Popen(f'pip install -r {req_file.as_posix()}',
#             stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, shell=True)
#     output = process.communicate()
#
#     sys.path.append((project_path / 'Code').as_posix())


class PBar:
    def __init__(self, text, n_ticks, n_counts, n_decimals=0):
        self.text = text
        self.n_ticks = n_ticks
        self.n_counts = n_counts
        self.n_chars = len(text) + n_ticks
        self.count = 0
        self.n_decimals = n_decimals

    def print(self, count):
        self.count = count
        if count == 0:
            perc = 0
        else:
            perc = self.count / self.n_counts
        tick = math.ceil(perc * self.n_ticks)
        pstr = '\r' + self.text + ': |'

        for _ in range(tick):
            pstr += '+'
        for _ in range(self.n_ticks - tick):
            pstr += '_'

        perc_print = f'{perc * 100:.20f}'
        # pstr += '|' + f' {perc_print[:self.n_decimals + 1]} %'
        pstr += f'|' + f' {perc*100:.0f} %'
        # pstr += f'{count, self.n_counts}'
        print(pstr, end='')

    def print_1(self):
        self.print(self.count + 1)

    def close(self):
        self.print(count=self.n_counts)
        print('')
    

def make_simulation_gif_video(folder, root, frames=None, duration=500):
    if frames is None:
        frames = [
            Image.open(image)
            for image in sorted(
                glob.glob(f"{folder}/{root}*.png")
            )
        ]
    frame_one = frames[0]
    frame_one.save(
        f"{root}.gif",
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=0,
    )

def save_fig(fig, savename: str, formats=None, scale=1, verbose=True):
    savename = writepath / savename
    if formats is None:
        formats = ['png']

    if not savename.parent.is_dir():
        savename.parent.mkdir(parents=True)

    for f in formats:
        file = savename.parent / (savename.name + f'.{f}')
        write_image(
            fig=fig,
            file=file,
            format=f,
            scale=1 if f != 'png' else 10,  # not used when exporting to svg
            engine='kaleido',
            # width=300,
            # height=180,
        )

        if verbose:
            print(f'saved: {file}')


def make_fig(width, height, x_domains, y_domains, **kwargs):
    for i, k in kwargs.items():
        assert i in ['subplot_titles', 'specs', 'equal_width_height', 'equal_width_height_axes'], f'{i}'

    # Check dimensions of x_domains and y_domains
    for row in x_domains.keys():
        assert row in y_domains.keys()
        assert len(x_domains[row]) == len(y_domains[row])
        for i in x_domains[row]:
            assert len(i) == 2
        for i in y_domains[row]:
            assert len(i) == 2

    font_famliy = 'calibri'
    figwidth_pxl = 498 * width
    figheight_pxl = 842 * 0.25 * height
    if 'equal_width_height' in kwargs.keys():
        eqw = kwargs['equal_width_height']
        assert eqw in [None, 'x', 'y', 'shortest']
    else:
        eqw = None

    if 'equal_width_height_axes' not in kwargs.keys():
        eqw_axes = 'all'
    else:
        eqw_axes = kwargs['equal_width_height_axes']

    specs_f = dict()
    if 'shared_xaxes' in kwargs.keys():
        specs_f['shared_xaxes'] = kwargs['shared_xaxes']
    # if 'subplot_titles' in kwargs.keys():
    #     specs_f['subplot_titles'] = kwargs['subplot_titles']
    if 'specs' in kwargs.keys():
        specs_f['specs'] = kwargs['specs']

    # Detect the nr of cols
    n_cols = 0
    for row, specs in x_domains.items():
        n_cols = np.max([n_cols, len(specs)])

    n_cols = np.int(n_cols)
    n_rows = len(x_domains.keys())

    fig = make_subplots(rows=n_rows, cols=n_cols, **specs_f)

    fig.update_layout(
        # Figure dimensions
        autosize=False,
        width=figwidth_pxl,
        height=figheight_pxl,
        margin=dict(l=0, t=0, b=0, r=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            borderwidth=0.5,
            bordercolor='black',
            font=dict(
                family=font_famliy,
                size=6,
            )
        ),
    )

    for row_i in range(n_rows):
        n_cols = len(x_domains[row_i+1])

        for col_i in range(n_cols):

            if 'subplot_titles' in kwargs.keys():
                # print(kwargs['subplot_titles'], ax_tick-1)
                fig.add_annotation(
                    x=x_domains[row_i+1][col_i][0],
                    y=y_domains[row_i+1][col_i][1],
                    text=kwargs['subplot_titles'][row_i+1][col_i],
                    font=dict(size=8, family=font_famliy),
                    showarrow=False,
                    xanchor='left', yanchor='bottom',
                    xref='paper', yref='paper',
                )

            x0, x1 = x_domains[row_i+1][col_i]
            y0, y1 = y_domains[row_i+1][col_i]
            if eqw_axes == 'all' or [row_i + 1, col_i + 1] in eqw_axes:
                if eqw == 'x':
                    n_px_x = (x1-x0) * figwidth_pxl
                    dy = n_px_x / figheight_pxl
                    y1 = y0 + dy
                elif eqw == 'y':
                    n_px_y = (y1-y0) * figheight_pxl
                    dx = n_px_y / figwidth_pxl
                    x1 = x0 + dx


            fig.update_xaxes(
                row=row_i+1,
                col=col_i+1,
                # automargin=False,
                domain=[x0, x1],

                # Xaxis ticks
                tickmode='array',
                tickvals=[],
                tickwidth=0.5,
                ticklen=0.1,
                tickangle=0,
                # tickcolor='crimson',
                tickfont=dict(
                    size=6,
                    family=font_famliy,
                ),

                # Xaxis line
                linewidth=0.5,
                linecolor='black',
                showline=True,

                # Title properties
                title=dict(
                    standoff=5,
                    font=dict(
                        size=8,
                        family=font_famliy,
                    )
                )

            )
            fig.update_yaxes(
                row=row_i+1,
                col=col_i+1,
                automargin=False,
                domain=[y0, y1],

                # Xaxis ticks
                tickmode='array',
                tickvals=[],
                tickwidth=0.5,
                ticklen=0.1,
                # tickcolor='crimson',
                tickfont=dict(
                    size=6,
                    family=font_famliy,
                ),

                # Xaxis line
                linewidth=0.5,
                linecolor='black',
                showline=True,

                # Title properties
                title=dict(
                    standoff=0,
                    font=dict(
                        size=8,
                        family=font_famliy,
                    )
                )
            )

    return fig


def interp_color(nsteps, step_nr, scalename, alpha, inverted=False):
    cols = getattr(getattr(plotly.colors, scalename[0]), scalename[1])
    if inverted:
        cols = cols[::-1]
    xx = np.linspace(0, 1, len(cols))

    if 'rgb' in cols[0]:
        r = [int(t.split('(')[1].split(',')[0]) for t in cols]
        g = [int(t.split('(')[1].split(',')[1]) for t in cols]
        b = [int(t.split('(')[1].split(',')[2].split(')')[0]) for t in cols]
    else:
        r = [int(tuple(int(t.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[0]) for t in cols]
        g = [int(tuple(int(t.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[1]) for t in cols]
        b = [int(tuple(int(t.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[2]) for t in cols]

    fr = interp1d(xx, r)
    fg = interp1d(xx, g)
    fb = interp1d(xx, b)

    ii = step_nr / nsteps
    return f'rgba({fr(ii)}, {fg(ii)}, {fb(ii)}, {alpha})'




