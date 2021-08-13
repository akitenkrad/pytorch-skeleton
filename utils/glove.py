from pathlib import Path
import pickle
import operator
from tqdm import tqdm
import numpy as np
import urllib.request as request
import zipfile
import progressbar
from .logger import get_logger

pbar = None
def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def download_glove_weights(path:Path):
    path.mkdir(parents=True, exist_ok=True)
    url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    request.urlretrieve(url, str(path / 'glove.6B.zip'), show_progress)

def load_glove(weights_path:str, no_cache:bool=False):
    '''load pretrained glove from http://nlp.stanford.edu
    vectors:    np.ndarray (n_words, dim)
    words:      [list of words]
    word2idx:   {word: idx}
    
    Args:
        weights_path: path to glove.xxx.txt (ex. glove.6B.300d.txt)
    
    Return:
        (vectors, words, word2idx)
    '''
    logger = get_logger('load_glove')
    weights_path = Path(weights_path)
    weights_zip_path = weights_path.parent / 'glove.6B.zip'
    
    if not weights_zip_path.exists():
        logger.info('download glove weights from the Internet.')
        download_glove_weights(weights_path.parent)

    # cache path
    _, tokens, dim = weights_path.stem.split('.')
    cache_dir = Path('__cache__/glove') / f'glove.{tokens}'
    vector_cache = cache_dir / f'glove.{tokens}.{dim}_vectors.pickle'
    words_cache = cache_dir / f'glove.{tokens}.{dim}_words.pickle'
    word2idx_cache = cache_dir / f'glove.{tokens}.{dim}_word2idx.pickle'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if no_cache == False and vector_cache.exists() and words_cache.exists() and word2idx_cache.exists():
        logger.info('restore weights from cache.')
        vectors = pickle.load(open(str(vector_cache), 'rb'))
        words = pickle.load(open(str(words_cache), 'rb'))
        word2idx = pickle.load(open(str(word2idx_cache), 'rb'))
    else:
        logger.info('construct weights from the zip file.')
        words = []
        idx = 0
        word2idx = {}
        vectors = []
        with zipfile.ZipFile(str(weights_zip_path)) as zip_f:
            with zip_f.open(weights_path.name) as f:

                # get file size
                f_len = sum([1 for _ in f])
                f.seek(0)

                # load weights
                for l in tqdm(f, desc='loading glove weights', total=f_len):
                    line = [i.strip() for i in l.strip().split()]
                    word = line[0]   
                    words.append(word)
                    word2idx[word] = idx
                    idx += 1         
                    vect = np.array(line[1:]).astype(np.float)
                    vectors.append(vect)
            vectors = np.array(vectors)
        
        # cache weights
        pickle.dump(vectors, open(str(vector_cache), 'wb'))
        pickle.dump(words, open(str(words_cache), 'wb'))
        pickle.dump(word2idx, open(str(word2idx_cache), 'wb'))
        
    logger.info(f'Finished loading glove weights: total={len(words)} words.')
    return np.array(vectors), words, word2idx

def load_glove_weights_matrix(weights_path:str, no_cache:bool=False):
    '''load pretrained glove weights matrix from http://nlp.stanford.edu
    vectors:        np.ndarray (n_words, dim)
    words:          [list of words]
    word2idx:       {word: idx}
    weights_matrix: np.ndarray
    
    Args:
        weights_path: path to glove.xxx.txt
        vocabulary: words list
    
    Return:
        weights_matrix
    '''
    # load glove weights
    vectors, words ,word2idx = load_glove(weights_path, no_cache)
    glove = {w: vectors[word2idx[w]] for w in words}
    embedding_dim = vectors.shape[-1]
                                                           
    # sort word2idx             
    word2idx = {k: v for k, v in sorted(word2idx.items(), key=operator.itemgetter(1))}
                                
    # construct weights_matrix
    vocab = list(set(words))
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, embedding_dim))
        
    # load weights
    for i, word in enumerate(vocab):
        if word in glove:
            weights_matrix[i] = glove[word]
        else:
            weights_matrix[i] = np.zeros(embedding_dim)
     
    return weights_matrix
