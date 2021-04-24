
# install kaggle (skip if already installed)
pip install kaggle

# prepare kaggle authentication key (download from https://www.kaggle.com/[username]/account into current folder)
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# download data by competitions name
mkdir -p coleridge_data
cd coleridge_data
kaggle competitions download -c coleridgeinitiative-show-us-the-data
unzip coleridgeinitiative-show-us-the-data.zip
cd ..


# write cleaner function (given by kaggle)
echo "
import re
def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())
    
" > utils.py
