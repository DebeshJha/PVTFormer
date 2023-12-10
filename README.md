# CT Liver Segmentation Via PVT-based Encoding and Refined Decoding

## Overview
PVTFormer is a groundbreaking encoder-decoder framework designed for precise liver segmentation from CT scans. At its core, it utilizes the Pyramid Vision Transformer (PVT v2) as a pretrained encoder, enhancing the segmentation process with its unique ability to handle variable-resolution input images and produce multi-scale representations.

## Key Features
-**Innovative Encoder-Decoder Framework:** Incorporates PVT v2 for efficient and rich feature extraction.

-**Hierarchical Decoding Strategy:** Enhances semantic features for high-quality segmentation masks.

-**Efficient Feature Processing:** Combines residual learning with Transformer mechanisms for optimal feature representation.

-**High Performance Metrics:** Achieves impressive dice coefficients and mean IoUs, outperforming state-of-the-art methods.




## TransRupNet Architecture 
<p align="center">
<img src="Img/PVTFormer1.png">
</p>

## Up Block and Decoder

<p align="center">
<img src="Img/PVTFormer_block.jpg">
</p>

## Architecture Advantages:
- Improved accuracy for medical image segmentation.
- Efficient learning of hierarchical features.
- Ability to capture long-range spatial dependencies.

  
## Applications
PVTFormer is highly effective for healthy liver segmentation, with potential applications in other medical imaging areas. It represents a significant advancement in medical image segmentation, offering a robust solution for accurate diagnosis and treatment planning.


## Uses of PVTFormer:
- Medical Image Segmentation 
- General Image Segmentation
- Anomaly Detection in Medical Images 
- Comparative Studies

## Dataset 
LiTS dataset


## Results
 **Qualitative results comparison of the SOTA methods** <br/>
<p align="center">
<img src="Img/PVTformer_results.png">
</p>

## Citation
Please cite our paper if you find the work useful: 
<pre>
@INPROCEEDINGS{8959021,
  author={D. {Jha},  K. {Biswas}, N. {Tomar}, U. {Bagci}},
  title={CT Liver Segmentation Via PVT-based Encoding and Refined Decoding}, 
  year={2023}
</pre>

## Contact
Please contact debesh.jha@northwestern.edu for any further questions.



