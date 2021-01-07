import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_ROOT = os.path.join(ROOT_DIR, 'data')
EMBD_FILE = os.path.join(ROOT_DIR, 'data/saved_embd.pt')


if __name__ == '__main__':
    print(EMBD_FILE)
