# SATO: Straighten Any 3D Tubular Object

This is the official code of [SATO: Straighten Any 3D Tubular Object](https://) (TMI 2025.5).

# ***Let's straighten any curved 3D tubular object!***

<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SATO/blob/main/figure/Qualitative%20Show%201.png" width="100%" >
</p>

<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SATO/blob/main/figure/Qualitative%20Show%203.png" width="100%" >
</p>

## How to Use
- Straightening a tubular object requires two elements: raw image (mask) and centerline
- [demo_image](https://)  provides various examples of tubular objects
- [straighten_img.py](https://github.com/Yanfeng-Zhou/SATO/blob/main/straighten/straighten_img.py) and [straighten_seg.py](https://github.com/Yanfeng-Zhou/SATO/blob/main/straighten/straighten_seg.py) are the implementation of the straightening algorithm
- [parallel_straighten.py](https://github.com/Yanfeng-Zhou/SATO/blob/main/straighten/parallel_straighten.py) is a multi-processing optimized version based on the [tqdm_pathos library](https://github.com/mdmould/tqdm_pathos) for straightening multiple tubular objects.


## Straightening Pipeline
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SATO/blob/main/figure/Straightening%20Pipeline.png" width="100%" >
</p>

## Zhou's Swept Frame
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SATO/blob/main/figure/Zhou's%20Swept%20Frame.png" width="50%" >
</p>


<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SATO/blob/main/figure/Definition%20of%20Zhou's%20Swept%20Frame.png" width="50%" >
</p>

## Comparison of Different Swept Frame
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SATO/blob/main/figure/Comparison%20of%20Different%20Swept%20Frame.png" width="100%" >
</p>


## Comparison of Different Straightening Methods
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SATO/blob/main/figure/Comparison%20of%20Different%20Straightening%20Methods.png" width="100%" >
</p>

## Requirements
```
numpy==1.21.6
scikit_image==0.19.3
scipy==1.7.3
SimpleITK==2.5.0
tqdm_pathos==0.4
```

## Citation
If our work is useful for your research, please cite our paper:
```

```
