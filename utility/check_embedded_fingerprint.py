# This script compares the embedded fingerprint to the non-embedded fingerprint
# by plotting the user specific binary code-vectors.

# ------------------------------------- Import Libraries and Modules ---------------------------------------------------

import os
import h5py
import matplotlib.pyplot as plt

# set paths
embed_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result_09"))
non_embed_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result_non"))
plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "embedded_fingerprint.png"))

# ---------------------------------------- Save and Load Functions -----------------------------------------------------

def load_from_hdf5(filename):
    with h5py.File(filename, 'r') as hdf5_file:
        signature = hdf5_file['signature'][:]
        fingerprint = hdf5_file['fingerprint'][:]
    return signature, fingerprint

# --------------------------------------------- Load Data --------------------------------------------------------------

# load the embedded fingerprint
embed_fingerprint_fname = os.path.join(embed_model_path, "extracted_fingerprint.h5")
embed_signature, embed_fingerprint = load_from_hdf5(embed_fingerprint_fname)

# load the non-embedded fingerprint
non_fingerprint_fname = os.path.join(non_embed_model_path, "non_extracted_fingerprint.h5")
non_embed_signature, non_embed_fingerprint = load_from_hdf5(non_fingerprint_fname)

# ----------------------------- Visualizing Differences in Fingerprint Distributions -----------------------------------

# plot the histograms of the original and pruned signatures
plt.hist(embed_signature, bins=25, alpha=0.5, label='Embedded', color='gray')
plt.hist(non_embed_signature, bins=80, alpha=0.5, label='Non-Embedded', color='orange')
plt.xlabel('Extracted Fingerprint')
plt.ylabel('Frequency')
plt.xlim(-0.1, 1.1)
plt.title(' ')
plt.legend(loc='upper center')

# show the figure
#plt.show()

# save the plot to file
plt.savefig(plot_path, dpi=300, bbox_inches='tight')











