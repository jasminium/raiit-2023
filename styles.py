import seaborn as sns

fig_size = (4, 4 / 1.6)

def style_setter():
    sns.set_style('whitegrid')
    sns.set_context('talk')
    sns.set_palette('colorblind')