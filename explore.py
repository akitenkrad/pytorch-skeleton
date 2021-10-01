from typing import List, Union, Tuple
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import seaborn as sns

from utils.utils import Path, glob, tqdm, np

sns.set()

class Feat(object):
    def __init__(self, data_path:Path, id:str, label:int, out:float, feat:np.ndarray):
        super().__init__()
        self.data_path = data_path
        self.id = id
        self.label = label
        self.out = out
        self.feat = feat
        self.ipca_feat = None
        
    def __str__(self):
        return f'<Feat id:{self.id} out:{self.out:.2f} label:{self.label}>'
    def __repr__(self):
        return self.__str__()

    def is_correct(self, thresh=0.5):
        return int(self.out > thresh) == self.label
    
def load_features(data_path:Union[str, Path], feat_dir:Union[str, Path]) -> List[Feat]:
    data_path = Path(data_path)
    feat_dir = Path(feat_dir)
    feat_id = feat_dir.parent.name

    feat_cache = Path('__cache__/') / feat_id / 'features.npy'
    ipca_cache = Path('__cache__/') / feat_id / 'ipca.npy'
    feat_cache.parent.mkdir(parents=True, exist_ok=True)

    if feat_cache.exists():
        cached_features = np.load(str(feat_cache), allow_pickle=True)
        print('loaded from cache', str(feat_cache.resolve().absolute()))
    else:
        cached_features = [np.load(f, allow_pickle=True) for f in tqdm(glob(str(feat_dir / '*.npy')), desc='load data...')]
        np.save(str(feat_cache), cached_features)
        print('loaded from feature directory')

    features = []
    for feat in tqdm(cached_features, desc='extract features...'):
        feat = feat.item()
        features.append(Feat(data_path, feat['item'].id, feat['item'].label, feat['out'], feat['feat']))

    if ipca_cache.exists():
        ipca = np.load(str(ipca_cache), allow_pickle=True)
    else:
        ipca = None
    return features, ipca

def calculate_ipca(data_path:Union[str, Path], feat_dir:Union[str, Path], n_components=10, batch_size=4) -> Tuple[List[Feat], IncrementalPCA]:
    features, ipca = load_features(data_path, feat_dir)
    feat_for_pca = np.concatenate([feat.feat for feat in features], axis=0)
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    print('start IPCA fit-transform')
    feat_ipca = ipca.fit_transform(feat_for_pca)
    for feat, ipca_feat in zip(features, feat_ipca):
        feat.ipca_feat = ipca_feat

    feat_id = Path(feat_dir).parent.name
    feat_cache = Path('__cache__') / feat_id / 'features.npy'
    ipca_cache = Path('__cache__') / feat_id / 'ipca.npy'
    np.save(str(feat_cache), features)
    np.save(str(ipca_cache), ipca)
    print('saved feature cache:', str(feat_cache.resolve().absolute()))
    print('saved ipca cache:', str(ipca_cache.resolve().absolute()))

    return features, ipca

def visualize(feat:Feat, colors=('black', 'red', 'green'), signal_names=('LIGO Hanford', 'LIGO Livingston', 'Virgo')):

    fig = plt.figure(figsize=(18, 30))
    grid = gridspec.GridSpec(nrows=7, ncols=2, figure=fig)

    # visualize wave
    wave = feat.load_wave()
    for i in range(3):
        ax = fig.add_subplot(grid[i, 0])
        ax.plot(wave[i], color=colors[i])
        ax.legend([signal_names[i]], fontsize=12, loc='lower right')

        ax = fig.subplot(grid[3, 0])
        ax.plot(wave[i], color=colors[i])
    
    ax = fig.subplot(grid[3, 0])
    ax.legend(signal_names, fontsize=12, loc='lower right')

    plt.show()

def show_components(features:List[Feat], name='animation.gif', target_axis=[0, 1, 2], thresh=0.5):
    
    assert name.endswith('.gif')
    assert len(target_axis) == 3
    axis_0, axis_1, axis_2 = target_axis
    
    # 1. prepare data
    print('prepare data')
    t_data = [feat for feat in features if feat.is_correct(thresh=thresh) == True]
    f_data = [feat for feat in features if feat.is_correct(thresh=thresh) == False]

    np.random.shuffle(t_data)
    np.random.shuffle(f_data)
    t_data = list(t_data)[:1000]
    f_data = list(f_data)[:1000]

    x_t = [float(feat.feat[axis_0]) for feat in t_data]
    y_t = [float(feat.feat[axis_1]) for feat in t_data]
    z_t = [float(feat.feat[axis_2]) for feat in t_data]

    x_f = [float(feat.feat[axis_0]) for feat in f_data]
    y_f = [float(feat.feat[axis_1]) for feat in f_data]
    z_f = [float(feat.feat[axis_2]) for feat in f_data]

    fig = plt.figure(figsize=(18, 18))
    ax_t = fig.add_subplot(121, projection='3d')
    ax_f = fig.add_subplot(122, projection='3d')
    
    # 2. define animation functions
    def init():
        ax_t.scatter(x_t, y_t, z_t, color='green', marker='o', alpha=0.5, s=60, label='True')
        ax_f.scatter(x_f, y_f, z_f, color='red', marker='x', alpha=0.5, s=120, label='False')
        ax_t.legend()
        ax_f.legend()
        ax_t.set_title(f'True plot: P{axis_0} x P{axis_1} x P{axis_2}')
        ax_f.set_title(f'False plot: P{axis_0} x P{axis_1} x P{axis_2}')
        return fig, 

    def animate(i):
        ax_t.view_init(elev=20.0, azim=3.6*i)
        ax_f.view_init(elev=20.0, azim=3.6*i)
        return fig,

    # 3. draw animation
    print('start to draw animation')
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=100, blit=True) 
    writergif = animation.PillowWriter(fps=5)
    anim.save(name, writer=writergif)
    print('Done ->', name)