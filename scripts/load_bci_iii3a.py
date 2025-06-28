import scipy.io
import numpy as np

def load_bci_iii3a_mat(filepath):
    mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    s = mat['s']  # EEG data, shape (samples, channels)
    hdr = mat['HDR']
    events = hdr.EVENT.POS  # Event positions (sample indices)
    types = hdr.EVENT.TYP   # Event types (codes)
    labels = getattr(hdr, 'Classlabel', None)
    return s, events, types, labels