
# *Node* - Size Matcher

Takes two images and matches their sizes using various methods, detailed below.

## cover_crop_center

> Expand the smaller image to cover the larger image without changing the aspect ratio. Then center it and crop the edges to match the size of the larger image.
> 
> <details>
> <summary> &nbsp; Expand </summary>
>
> 
> ![alt text](wiki/demo/size-match/cover-crop-center.png)
>
> </details>

## cover_crop

> Expand the smaller image to cover the larger image without changing the aspect ratio. Then crop the edges to match the size of the larger image.
>
> 
> <details>
> <summary>&nbsp; Expand </summary>
>
> 
> ![alt text](wiki/demo/size-match/cover-crop.png)
>
> </details>

## fit_center

> Expand the smaller image as much as possible to fit inside the larger image without changing the aspect ratio. Then center it.
>
> 
> <details>
> <summary>&nbsp; Expand </summary>
>
> 
> ![alt text](wiki/demo/size-match/fit-center.png)
>
> </details>

## center_dont_resize

> Center the smaller image inside the larger image without changing either sizes.
>
> 
> <details>
> <summary>&nbsp; Expand </summary>
>
> 
> ![alt text](wiki/demo/size-match/center-dont-resize.png)
>
> </details>

## fill

> Expand the smaller image to exactly match the size of the larger image, allowing the aspect ratio to change.
>
> 
> <details>
> <summary>&nbsp; Expand </summary>
>
> 
> ![alt text](wiki/demo/size-match/fill.png)
>
> </details>

## crop_larger_center

> Crop the larger image to match the size of the smaller image. Then center it.
>
> 
> <details>
> <summary>&nbsp; Expand </summary>
>
> 
> ![alt text](wiki/demo/size-match/crop-larger-center.png)
>
> </details>

## crop_larger_topleft

> Crop the larger image to match the size of the smaller image. Then align it to the top-left corner.
> 
> <details>
> <summary>&nbsp; Expand </summary>
>
> 
>
> ![alt text](wiki/demo/size-match/crop-larger-topleft.png)
>
> </details>
>

&nbsp;

# *Node* - Paste Cutout on Base Image (Compositing)


- Automatically matches size of two images with various size matching methods
- If the cutout doesn't have an alpha channel (not really a cutout), the bg is automatically inferred and made transparent
- Invert option



## Base Layer Composite with Alpha Layer

> 
> <details>
> <summary>&nbsp; Expand </summary>
>
> 
> ![paste-cutout](wiki/demo/composite/paste-cutout.png)
>
> </details>


## BG Being Inferred when Alpha Layer is Missing



> 
> <details>
> <summary>&nbsp; Expand </summary>
>
> 
>
> ![inferred-bg](wiki/demo/composite/inferred-bg.png)
> 
>
> </details>

## Using with Auto Segmentation

> 
> <details>
> <summary>&nbsp; Expand </summary>
>
> 
>
> ![with-auto-segmentation](wiki/demo/composite/with-auto-segmentation.png)
>
>
> </details>


## With Chromakeying

> 
> <details>
> <summary>&nbsp; Expand </summary>
>
> 
> 
> ![with-chromakeying](wiki/demo/composite/with-chromakeying.png)
>
>
> </details>


# Installation

1. `cd` into `ComfyUI/custom_nodes`
2. `git clone` this repo

# Requirements

- Python3.10+