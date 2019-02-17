# RESIDUAL FEATURE EXTRACTION PIPELINE
A pipeline that carries out feature extraction of residual substructure within the residual images produced by popular galaxy structural-fitting routines such as GALFIT, GIM2D, etc. This pipeline works on model-subtracted residual images by popular structural fitting routines (e.g., GALFIT, GIM2D, etc) to extract faint low surface brightness features by isolating flux-wise and area-wise significant contiguous pixels regions by rigourous masking routine. This routine accepts the image cubes (original image, model image, residual image) and generates several data products:

1. An Image with Extracted features.
2. Source extraction based segmentation map.
3. The background sky mask and the residual extraction mask.
4. A montecarlo approach based area threshold above which the extracted features are identified.
5. A catalog entry indicating the surface brightness and its error.

**Author:** Kameswara Bharadwaj Mantha
**email:** km4n6@mail.umkc.edu

**Publication:** 
Studying the Physical Properties of Tidal Features I. Extracting Morphological Substructure in CANDELS Observations and VELA Simulations.

**Corresponding Author:** 
Kameswara Bharadwaj Mantha

**Co-authors:** 
Daniel H. McIntosh, Cody P. Ciaschi, Rubyet Evan, Henry C. Ferguson, Logan B. Fries, Yicheng Guo, Luther D. Landry, Elizabeth J. McGrath, Raymond C. Simons, Gregory F. Snyder, Scott E. Thompson, Eric F. Bell, Daniel Ceverino, Nimish P. Hathi, Anton M. Koekemoer, Camilla Pacifici, Joel R. Primack, Marc Rafelski, Vicente Rodriguez-Gomez.

# README OF RESIDUAL FEATURE EXTRACTION PIPELINE

You will use the python script: `Tidal_feature_finder.py`. This python file uses two other python scripts which have all the necessary functions that carry out the task.
