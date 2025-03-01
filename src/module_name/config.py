from pathlib import Path

module_dir = Path(__file__).parent
src_dir = module_dir.parent
root_dir = src_dir.parent
data_dir = root_dir/"data"
vir_dir = root_dir/"Viridien"

def test():
    print("Root directory:\t\t", root_dir)
    print("Data folder directory:\t", data_dir)
    print("Source directory:\t", src_dir)
    print("Module directory:\t", module_dir)
