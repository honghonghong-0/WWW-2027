"""Register RedditMovie in UCCR_infer (dataset class, language, model resource + kgsf model dir symlink) and write the www27 config."""
import os, re, shutil
U = "/path/to/base/projects/crs/external/UCCR_infer"
def bak(p):
    if not os.path.exists(p + ".bak_www27u"): shutil.copy(p, p + ".bak_www27u")
os.makedirs(f"{U}/crslab/data/dataset/reddit", exist_ok=True)
open(f"{U}/crslab/data/dataset/reddit/reddit.py", "w").write('''# WWW27: Reddit-Movie in ReDial format (copied by build_uccr_reddit.py; integer conv/user ids)
import os
from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset
from crslab.data.dataset.redial.redial import ReDialDataset
from crslab.data.dataset.redial.resources import resources
class RedditDataset(ReDialDataset):
    def __init__(self, opt, tokenize, restore=False, save=False):
        resource = resources[tokenize]
        self.special_token_idx = resource["special_token_idx"]; self.unk_token_idx = self.special_token_idx["unk"]
        dpath = os.path.join(DATASET_PATH, "reddit", tokenize)
        BaseDataset.__init__(self, opt, dpath, resource, restore, save)
''')
open(f"{U}/crslab/data/dataset/reddit/__init__.py", "w").write("from .reddit import RedditDataset\n")
p = f"{U}/crslab/data/dataset/__init__.py"; bak(p); s = open(p).read()
if "RedditDataset" not in s: s += "\nfrom .reddit import RedditDataset  # WWW27\n"; open(p, "w").write(s)
p = f"{U}/crslab/data/__init__.py"; bak(p); s = open(p).read()
if "'RedditMovie'" not in s:
    s = s.replace("    'ReDial': ReDialDataset,", "    'ReDial': ReDialDataset,\n    'RedditMovie': RedditDataset,  # WWW27", 1)
    s = s.replace("    'ReDial': 'en',", "    'ReDial': 'en',\n    'RedditMovie': 'en',  # WWW27", 1)
    open(p, "w").write(s)
p = f"{U}/crslab/model/crs/uccr/resources.py"; bak(p); s = open(p).read()
if "RedditMovie" not in s: s += "\nresources['RedditMovie'] = resources['ReDial']  # WWW27: same KGSF resource (no download needed if dir exists)\n"; open(p, "w").write(s)
lk = f"{U}/data/model/kgsf/RedditMovie"
if not os.path.exists(lk): os.symlink(f"{U}/data/model/kgsf/ReDial", lk)
c = open(f"{U}/config/crs/uccr/tgredial_www27_recOnly.yaml").read()
c = c.replace("dataset: TGReDial", "dataset: RedditMovie").replace("tokenize: pkuseg", "tokenize: nltk")
c = re.sub(r"embedding: .*\n", "", c)
open(f"{U}/config/crs/uccr/reddit_www27_recOnly.yaml", "w").write("# WWW27: UCCR rec-only on Reddit-Movie; same hyperparameters as TG run; no pretrained word embedding (as KGSF Reddit)\n" + c)
print(open(f"{U}/config/crs/uccr/reddit_www27_recOnly.yaml").read()[:400]); print("OK")
