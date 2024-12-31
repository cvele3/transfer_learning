tensorinjo = "tensorflow-2.11.0-cp38-cp38-win_amd64.whl"
rdkitinjo = "rdkit_pypi-2022.9.5-cp38-cp38-win_amd64.whl"
stelargrfinjo = "stellargraph-1.2.1-py3-none-any.whl"

import pip

def install_whl(path):
   pip.main(['install', path])

install_whl(tensorinjo)
install_whl(rdkitinjo)
install_whl(stelargrfinjo)