# 🏰 monoscule-c

monoscule-c is a minimalist, dependency-free computer vision library designed for microcontrollers and other resource-constrained devices. It focuses on **grayscale** images and provides modern, practical algorithms that fit in a few kilobytes of code. Single-header design, integer-based operations, pure C99.

## Features

* Image operations: copy, crop, resize (bilinear), downsample
* Filtering: blur, Sobel edges, thresholding (global, Otsu, adaptive)
* Morphology: erosion, dilation
* Geometry: connected components, perspective warp
* Features: FAST/ORB keypoints and descriptors (object tracking)
* Local binary patterns: LBP cascades to detect faces, vehicles etc
* Utilities: PGM read/write

As usual, no dependencies, no dynamic memory allocation, no C++, no surprises. Just a single header file under 1KLOC.

Check out the [examples](examples) folder for more!


## Quickstart

```c
#include "monoscule-c.h"

struct ms_image img = ms_read_pgm("input.pgm");
struct ms_image blurred = ms_alloc(img.w, img.h);
struct ms_image binary = ms_alloc(img.w, img.h);

ms_blur(blurred, img, 2);
ms_threshold(binary, blurred, ms_otsu_theshold(blurred));

ms_write_pgm(binary, "output.pgm");
ms_free(img);
ms_free(blurred);
ms_free(binary);
```

_Note that `ms_alloc`/`ms_free` are optional helpers; you can allocate image pixel buffers any way you like._

## API Reference

```c
struct ms_image { unsigned w, h; uint8_t *data; };
struct ms_rect { unsigned x, y, w, h; }; // ROI
struct ms_point { unsigned x, y; }; // corners

uint8_t ms_get(struct ms_image img, unsigned x, unsigned y);
void ms_set(struct ms_image img, unsigned x, unsigned y, uint8_t value);
void ms_crop(struct ms_image dst, struct ms_image src, struct ms_rect roi);
void ms_copy(struct ms_image dst, struct ms_image src);
void ms_resize(struct ms_image dst, struct ms_image src);
void ms_downsample(struct ms_image dst, struct ms_image src);

// Thresholding
void ms_histogram(struct ms_image img, unsigned hist[256]);
void ms_threshold(struct ms_image img, uint8_t threshold);
uint8_t ms_otsu_threshold(struct ms_image img);
void ms_adaptive_threshold(struct ms_image dst, struct ms_image src, unsigned radius, int c);

// Filters
void ms_blur(struct ms_image dst, struct ms_image src, unsigned radius);
void ms_erode(struct ms_image dst, struct ms_image src);
void ms_dilate(struct ms_image dst, struct ms_image src);
void ms_sobel(struct ms_image dst, struct ms_image src);

// Blobs (connected components) and contours
typedef uint16_t ms_label;
struct ms_blob { ms_label label; unsigned area; struct ms_rect box; struct ms_point centroid; };
struct ms_contour { struct ms_rect box; struct ms_point start; unsigned length; };
unsigned ms_blobs(struct ms_image img, ms_label *labels, struct ms_blob *blobs, unsigned nblobs);
void ms_blob_corners(struct ms_image img, ms_label *labels, struct ms_blob *b, struct ms_point c[4]);
void ms_perspective_correct(struct ms_image dst, struct ms_image src, struct ms_point c[4]);
void ms_trace_contour(struct ms_image img, struct ms_image visited, struct ms_contour *c);

// FAST/ORB
struct ms_keypoint { struct ms_point pt; unsigned response; float angle; uint32_t descriptor[8]; };
struct ms_match { unsigned idx1, idx2; unsigned distance; };
unsigned ms_fast(struct ms_image img, struct ms_image scoremap, struct ms_keypoint *kps, unsigned nkps, unsigned threshold);
float ms_compute_orientation(struct ms_image img, unsigned x, unsigned y, unsigned r);
void ms_brief_descriptor(struct ms_image img, struct ms_keypoint *kp);
unsigned ms_orb_extract(struct ms_image img, struct ms_keypoint *kps, unsigned nkps, unsigned threshold, uint8_t *scoremap_buffer);
unsigned ms_match_orb(const struct ms_keypoint *kps1, unsigned n1, const struct ms_keypoint *kps2, unsigned n2, struct ms_match *matches, unsigned max_matches, float max_distance);

// LBP cascades
struct ms_lbp_cascade { uint16_t window_w, window_h; uint16_t nfeatures, nweaks, nstages; const int8_t *features; /* [nfeatures * 4] */ const uint16_t *weak_feature_idx; const float *weak_left_val, *weak_right_val; const uint16_t *weak_subset_offset, *weak_num_subsets; const int32_t *subsets; const uint16_t *stage_weak_start, *stage_nweaks; const float *stage_threshold; };
void ms_integral(struct ms_image src, unsigned *ii);
unsigned ms_lbp_window(const struct ms_lbp_cascade *c, const unsigned *ii, unsigned iw, unsigned ih, int x, int y, float scale);
unsigned ms_lbp_detect(const struct ms_lbp_cascade *c, const unsigned *ii, unsigned iw, unsigned ih, struct ms_rect *rects, unsigned max_rects, float scale_factor, float min_scale, float max_scale, int step);

// Optional:
struct ms_image ms_alloc(unsigned w, unsigned h);
void ms_free(struct ms_image img);
struct ms_image ms_read_pgm(const char *path);
int ms_write_pgm(struct ms_image img, const char *path);
```

## License

This project is licensed under the MIT License. Feel free to use in research, products, and your next embedded vision project!
