#ifndef monoscule_H
#define monoscule_H

#include <limits.h>
#include <stdint.h>

#ifndef ms_API
#define ms_API static inline
#endif

#define ms_MIN(a, b) ((a) < (b) ? (a) : (b))
#define ms_MAX(a, b) ((a) > (b) ? (a) : (b))

struct ms_image {
  unsigned w, h;
  uint8_t *data;
};

struct ms_rect {
  unsigned x, y, w, h;
};

struct ms_point {
  unsigned x, y;
};

typedef uint16_t ms_label;

struct ms_blob {
  ms_label label;
  unsigned area;
  struct ms_rect box;
  struct ms_point centroid;
};

struct ms_contour {
  struct ms_rect box;
  struct ms_point start;
  unsigned length;
};

struct ms_keypoint {
  struct ms_point pt;
  unsigned response;
  float angle;
  uint32_t descriptor[8];
};

struct ms_match {
  unsigned idx1, idx2;
  unsigned distance;
};

struct ms_lbp_cascade {
  uint16_t window_w, window_h;
  uint16_t nfeatures, nweaks, nstages;
  const int8_t *features; /* [nfeatures * 4] */
  const uint16_t *weak_feature_idx;
  const float *weak_left_val, *weak_right_val;
  const uint16_t *weak_subset_offset, *weak_num_subsets;
  const int32_t *subsets;
  const uint16_t *stage_weak_start, *stage_nweaks;
  const float *stage_threshold;
};

static inline int ms_valid(struct ms_image img) { return img.data && img.w > 0 && img.h > 0; }

#ifdef ms_NO_STDLIB  // no asserts, no memory allocation, no file I/O
#define ms_assert(cond)
static inline float ms_atan2(float y, float x)
{
    union { float f; uint32_t i; } ux = {x}, uy = {y};

    uint32_t sign_x = ux.i & 0x80000000u;
    uint32_t sign_y = uy.i & 0x80000000u;

    ux.i &= 0x7FFFFFFFu;
    uy.i &= 0x7FFFFFFFu;

    float ax = ux.f;
    float ay = uy.f;

    // Compute once
    int swap = ay > ax;

    float a = swap ? ax : ay;
    float b = swap ? ay : ax;

    if (b == 0.0f)
        return 0.0f;

    float r  = a / b;
    float r2 = r * r;

    // Polynomial atan(r)
    float angle =
        (((-0.0464964749f * r2 + 0.15931422f) * r2
           - 0.327622764f) * r2) * r + r;

    // Quadrant correction
    if (swap)
        angle = 1.570796327f - angle;

    if (x < 0.0f)
        angle = 3.141592654f - angle;

    // Restore sign of y
    union { float f; uint32_t i; } ures = { angle };
    ures.i = (ures.i & 0x7FFFFFFFu) | sign_y;

    return ures.f;
}

float ms_sin(float var) {
    const float PI      = 3.14159265f;
    const float TAU     = 6.28318531f;
    const float INV_TAU = 0.15915494f; 
    const float PI_SQ   = 9.86960440f;

    float n = (float)((int)(var * INV_TAU + (var >= 0 ? 0.5f : -0.5f)));
    var -= n * TAU;

    union { float f; uint32_t i; } u = { var };
    float sign = (u.i & 0x80000000u) ? -1.0f : 1.0f;
    u.i &= 0x7FFFFFFFu;
    var = u.f;

    float arc = var * (PI - var);
    return sign * (16.0f * arc) / ((5.0f * PI_SQ) - (4.0f * arc));
}
#else
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ms_assert(cond)                               \
  if (!(cond)) {                                      \
    fprintf(stderr, "Assertion failed: %s\n", #cond); \
    abort();                                          \
  }

static inline float ms_atan2(float y, float x) { return atan2f(y, x); }
static inline float ms_sin(float x) { return sinf(x); }

ms_API struct ms_image ms_alloc(unsigned w, unsigned h) {
  if (w == 0 || h == 0) return (struct ms_image){0, 0, NULL};
  uint8_t *data = (uint8_t *)calloc(w * h, sizeof(uint8_t));
  return (struct ms_image){w, h, data};
}

ms_API void ms_free(struct ms_image img) { free(img.data); }

ms_API struct ms_image ms_read_pgm(const char *path) {
  struct ms_image img = {0, 0, NULL};
  FILE *f = (path[0] == '-' && !path[1]) ? stdin : fopen(path, "rb");
  if (!f) return img;
  unsigned w, h, maxval;
  if (fscanf(f, "P5\n%u %u\n%u\n", &w, &h, &maxval) != 3 || maxval != 255) goto end;
  img = ms_alloc(w, h);
  if (!ms_valid(img)) goto end;
  if (fread(img.data, sizeof(uint8_t), w * h, f) != (size_t)(w * h)) {
    ms_free(img);
    img = (struct ms_image){0, 0, NULL};
  }
end:
  fclose(f);
  return img;
}

ms_API int ms_write_pgm(struct ms_image img, const char *path) {
  if (!ms_valid(img)) return -1;
  FILE *f = (path[0] == '-' && !path[1]) ? stdout : fopen(path, "wb");
  if (!f) return -1;
  fprintf(f, "P5\n%u %u\n255\n", img.w, img.h);
  size_t written = fwrite(img.data, sizeof(uint8_t), img.w * img.h, f);
  fclose(f);
  return (written == (size_t)(img.w * img.h)) ? 0 : -1;
}
#endif  // ms_NO_STDLIB

#define ms_for(img, x, y)                \
  for (unsigned y = 0; y < (img).h; y++) \
    for (unsigned x = 0; x < (img).w; x++)

ms_API uint8_t ms_get(struct ms_image img, unsigned x, unsigned y) {
  return (ms_valid(img) && x < img.w && y < img.h) ? img.data[y * img.w + x] : 0;
}
ms_API void ms_set(struct ms_image img, unsigned x, unsigned y, uint8_t value) {
  if (ms_valid(img) && x < img.w && y < img.h) img.data[y * img.w + x] = value;
}

//
// Image processing
//

ms_API void ms_crop(struct ms_image dst, struct ms_image src, struct ms_rect roi) {
  ms_assert(ms_valid(dst) && ms_valid(src) && roi.x + roi.w <= src.w && roi.y + roi.h <= src.h &&
            dst.w == roi.w && dst.h == roi.h);
  ms_for(roi, x, y) ms_set(dst, x, y, ms_get(src, roi.x + x, roi.y + y));
}

ms_API void ms_copy(struct ms_image dst, struct ms_image src) {
  ms_crop(dst, src, (struct ms_rect){0, 0, src.w, src.h});
}

ms_API void ms_resize_nn(struct ms_image dst, struct ms_image src) {
  ms_for(dst, x, y) {
    unsigned sx = x * src.w / dst.w, sy = y * src.h / dst.h;
    ms_set(dst, x, y, ms_get(src, sx, sy));
  }
}

ms_API void ms_resize(struct ms_image dst, struct ms_image src) {
  ms_assert(ms_valid(dst) && ms_valid(src));
  ms_for(dst, x, y) {
    float sx = ((float)x + 0.5f) * src.w / dst.w - 0.5f;  // 0.5f centers the pixel
    float sy = ((float)y + 0.5f) * src.h / dst.h - 0.5f;
    sx = ms_MAX(0.0f, ms_MIN(sx, src.w - 1.0f));
    sy = ms_MAX(0.0f, ms_MIN(sy, src.h - 1.0f));
    unsigned sx_int = (unsigned)sx, sy_int = (unsigned)sy;
    unsigned sx1 = ms_MIN(sx_int + 1, src.w - 1), sy1 = ms_MIN(sy_int + 1, src.h - 1);
    float dx = sx - sx_int, dy = sy - sy_int;
    uint8_t c00 = ms_get(src, sx_int, sy_int), c01 = ms_get(src, sx1, sy_int),
            c10 = ms_get(src, sx_int, sy1), c11 = ms_get(src, sx1, sy1);
    uint8_t p = (c00 * (1 - dx) * (1 - dy)) + (c01 * dx * (1 - dy)) + (c10 * (1 - dx) * dy) +
                (c11 * dx * dy);
    ms_set(dst, x, y, p);
  }
}

ms_API void ms_downsample(struct ms_image dst, struct ms_image src) {
  ms_assert(ms_valid(src) && ms_valid(dst) && dst.w == src.w / 2 && dst.h == src.h / 2);
  ms_for(dst, x, y) {
    unsigned src_x = x * 2, src_y = y * 2;
    unsigned sum = ms_get(src, src_x, src_y) + ms_get(src, src_x + 1, src_y) +
                   ms_get(src, src_x, src_y + 1) + ms_get(src, src_x + 1, src_y + 1);
    ms_set(dst, x, y, (uint8_t)(sum / 4));
  }
}

ms_API void ms_histogram(struct ms_image img, unsigned hist[256]) {
  ms_assert(ms_valid(img) && hist != NULL);
  for (unsigned i = 0; i < 256; i++) hist[i] = 0;
  for (unsigned i = 0; i < img.w * img.h; i++) hist[img.data[i]]++;
}

ms_API uint8_t ms_otsu_threshold(struct ms_image img) {
  ms_assert(ms_valid(img));
  unsigned hist[256] = {0}, wb = 0, wf = 0, threshold = 0;
  ms_histogram(img, hist);
  float sum = 0, sumB = 0, varMax = -1.0;
  for (unsigned i = 0; i < 256; i++) sum += (float)i * hist[i];
  for (unsigned t = 0; t < 256; t++) {
    wb += hist[t];
    if (wb == 0) continue;
    wf = img.w * img.h - wb;
    if (wf == 0) break;
    sumB += (float)t * hist[t];
    float mB = (float)sumB / wb;
    float mF = (float)(sum - sumB) / wf;
    float varBetween = (float)wb * (float)wf * (mB - mF) * (mB - mF);
    if (varBetween > varMax) varMax = varBetween, threshold = t;
  }
  return (uint8_t)threshold;
}

ms_API void ms_threshold(struct ms_image img, uint8_t thresh) {
  ms_assert(ms_valid(img));
  for (unsigned i = 0; i < img.w * img.h; i++) img.data[i] = (img.data[i] > thresh) ? 255 : 0;
}

ms_API void ms_adaptive_threshold(struct ms_image dst, struct ms_image src, unsigned radius,
                                  int c) {
  ms_assert(ms_valid(dst) && ms_valid(src) && dst.w == src.w && dst.h == src.h);
  ms_for(src, x, y) {
    unsigned sum = 0, count = 0;
    for (int dy = -radius; dy <= (int)radius; dy++) {
      for (int dx = -radius; dx <= (int)radius; dx++) {
        int sy = (int)y + dy, sx = (int)x + dx;
        if (sy >= 0 && sy < (int)src.h && sx >= 0 && sx < (int)src.w) {
          sum += ms_get(src, sx, sy);
          count++;
        }
      }
    }
    int threshold = sum / count - c;
    ms_set(dst, x, y, (ms_get(src, x, y) > threshold) ? 255 : 0);
  }
}

#define ms_sharpen ((struct ms_image){3, 3, (uint8_t[]){0, -1, 0, -1, 5, -1, 0, -1, 0}})  // norm 1
#define ms_emboss ((struct ms_image){3, 3, (uint8_t[]){-2, -1, 0, -1, 1, 1, 0, 1, 2}})    // norm 1
#define ms_blur_box ((struct ms_image){3, 3, (uint8_t[]){1, 1, 1, 1, 1, 1, 1, 1, 1}})     // norm 9
#define ms_blur_gaussian \
  ((struct ms_image){3, 3, (uint8_t[]){1, 2, 1, 2, 4, 2, 1, 2, 1}})  // norm 16

ms_API void ms_filter(struct ms_image dst, struct ms_image src, struct ms_image kernel,
                      unsigned norm) {
  ms_assert(ms_valid(src) && ms_valid(dst) && dst.w == src.w && dst.h == src.h && norm > 0);
  ms_for(dst, x, y) {
    int sum = 0;
    ms_for(kernel, i, j) {
      sum += ms_get(src, x + i - kernel.w / 2, y + j - kernel.h / 2) * (int8_t)ms_get(kernel, i, j);
    }
    sum = sum / norm;
    ms_set(dst, x, y, ms_MIN(255, ms_MAX(0, sum)));
  }
}

ms_API void ms_blur(struct ms_image dst, struct ms_image src, unsigned radius) {
  ms_assert(ms_valid(src) && ms_valid(dst) && dst.w == src.w && dst.h == src.h);
  ms_for(src, x, y) {
    unsigned sum = 0, count = 0;
    for (int dy = -radius; dy <= (int)radius; dy++) {
      for (int dx = -radius; dx <= (int)radius; dx++) {
        int sy = y + dy, sx = x + dx;
        if (sy >= 0 && sy < (int)src.h && sx >= 0 && sx < (int)src.w) {
          sum += ms_get(src, sx, sy);
          count++;
        }
      }
    }
    ms_set(dst, x, y, (uint8_t)(sum / count));
  }
}

enum { ms_ERODE, ms_DILATE };
static inline void ms_morph(struct ms_image dst, struct ms_image src, int op) {
  ms_assert(ms_valid(dst) && ms_valid(src) && dst.w == src.w && dst.h == src.h);
  ms_for(src, x, y) {
    uint8_t val = op == ms_ERODE ? 255 : 0;
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        int sy = (int)y + dy, sx = (int)x + dx;
        if (sy >= 0 && sy < (int)src.h && sx >= 0 && sx < (int)src.w) {
          uint8_t pixel = ms_get(src, sx, sy);
          if (op == ms_DILATE && pixel > val) val = pixel;
          if (op == ms_ERODE && pixel < val) val = pixel;
        }
      }
    }
    ms_set(dst, x, y, val);
  }
}
ms_API void ms_erode(struct ms_image dst, struct ms_image src) { ms_morph(dst, src, ms_ERODE); }
ms_API void ms_dilate(struct ms_image dst, struct ms_image src) { ms_morph(dst, src, ms_DILATE); }

ms_API void ms_sobel(struct ms_image dst, struct ms_image src) {
  ms_assert(ms_valid(dst) && ms_valid(src) && dst.w == src.w && dst.h == src.h);
  for (unsigned y = 1; y < src.h - 1; y++) {
    for (unsigned x = 1; x < src.w - 1; x++) {
      int gx = -src.data[(y - 1) * src.w + (x - 1)] + src.data[(y - 1) * src.w + (x + 1)] -
               2 * src.data[y * src.w + (x - 1)] + 2 * src.data[y * src.w + (x + 1)] -
               src.data[(y + 1) * src.w + (x - 1)] + src.data[(y + 1) * src.w + (x + 1)];
      int gy = -src.data[(y - 1) * src.w + (x - 1)] - 2 * src.data[(y - 1) * src.w + x] -
               src.data[(y - 1) * src.w + (x + 1)] + src.data[(y + 1) * src.w + (x - 1)] +
               2 * src.data[(y + 1) * src.w + x] + src.data[(y + 1) * src.w + (x + 1)];
      int magnitude = ((gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy)) / 2;
      dst.data[y * dst.w + x] = (uint8_t)ms_MAX(0, ms_MIN(magnitude, 255));
    }
  }
}

//
// Connected components (blobs)
//
static inline ms_label ms_root(ms_label x, ms_label *parents) {
  while (parents[x] != x) x = parents[x] = parents[parents[x]];
  return x;
}

ms_API unsigned ms_blobs(struct ms_image img, ms_label *labels, struct ms_blob *blobs,
                         unsigned nblobs) {
  ms_assert(ms_valid(img) && labels != NULL && blobs != NULL && nblobs > 0);
  unsigned w = img.w;
  ms_label next = 1, parents[nblobs + 1];
  unsigned cx[nblobs], cy[nblobs];
  for (unsigned i = 0; i < img.w * img.h; i++) labels[i] = 0;
  for (unsigned i = 0; i < nblobs; i++)
    blobs[i] = (struct ms_blob){0, 0, {UINT_MAX, UINT_MAX, 0, 0}, {0, 0}};
  for (unsigned i = 0; i <= nblobs; i++) parents[i] = i;
  // first pass: label and union
  ms_for(img, x, y) {
    if (ms_get(img, x, y) < 128) continue;  // skip background pixels
    ms_label left = (x > 0) ? labels[y * w + (x - 1)] : 0;
    ms_label top = (y > 0) ? labels[(y - 1) * w + x] : 0;
    // 4-connectivity: pick smallest from left and top, if any is non-zero
    ms_label n = (left && top ? ms_MIN(left, top) : (left ? left : (top ? top : 0)));
    if (!n) {                       // new component
      if (next > nblobs) continue;  // out of labels
      blobs[next - 1] = (struct ms_blob){next, 1, {x, y, x, y}, {x, y}};
      cx[next - 1] = x, cy[next - 1] = y;
      labels[y * w + x] = next++;
    } else {  // existing component
      labels[y * w + x] = n;
      struct ms_blob *b = &blobs[n - 1];
      cx[n - 1] += x, cy[n - 1] += y;
      b->area++;
      b->box.x = ms_MIN(x, b->box.x), b->box.y = ms_MIN(y, b->box.y);
      // keep bottom-right point coordinates in w/h of the rect, adjust later
      b->box.w = ms_MAX(x, b->box.w), b->box.h = ms_MAX(y, b->box.h);
      // union if labels are different
      if (left && top && left != top) {
        ms_label root1 = ms_root(left, parents), root2 = ms_root(top, parents);
        if (root1 != root2) parents[ms_MAX(root1, root2)] = ms_MIN(root1, root2);
      }
    }
  }
  // merge blobs
  for (int i = 0; i < next - 1; i++) {
    ms_label root = ms_root(blobs[i].label, parents);
    if (root != blobs[i].label) {
      struct ms_blob *broot = &blobs[root - 1];
      broot->area += blobs[i].area;
      broot->box.x = ms_MIN(broot->box.x, blobs[i].box.x);
      broot->box.y = ms_MIN(broot->box.y, blobs[i].box.y);
      broot->box.w = ms_MAX(broot->box.w, blobs[i].box.w);
      broot->box.h = ms_MAX(broot->box.h, blobs[i].box.h);
      cx[root - 1] += cx[i], cy[root - 1] += cy[i];
      blobs[i].area = 0;
    }
  }
  // second pass: update labels
  ms_for(img, x, y) {
    ms_label l = labels[y * w + x];
    if (l) labels[y * w + x] = ms_root(l, parents);
  }

  // compact blobs
  unsigned m = 0;
  for (int i = 0; i < next - 1; i++) {
    if (blobs[i].area == 0) continue;
    // fix rect width/height from bottom-right point to actual width/height
    blobs[i].box.w = blobs[i].box.w - blobs[i].box.x + 1;
    blobs[i].box.h = blobs[i].box.h - blobs[i].box.y + 1;
    // calculate centroids
    blobs[i].centroid.x = cx[i] / blobs[i].area;
    blobs[i].centroid.y = cy[i] / blobs[i].area;
    // move to compacted position
    blobs[m++] = blobs[i];
  }

  return m;  // number of non-empty blobs
}

ms_API void ms_blob_corners(struct ms_image img, ms_label *labels, struct ms_blob *b,
                            struct ms_point c[4]) {
  ms_assert(ms_valid(img) && b && labels);
  struct ms_point tl = b->centroid, tr = b->centroid, br = b->centroid, bl = b->centroid;
  int min_sum = INT_MAX, max_sum = INT_MIN, min_diff = INT_MAX, max_diff = INT_MIN;
  for (unsigned y = b->box.y; y < b->box.y + b->box.h; y++) {
    for (unsigned x = b->box.x; x < b->box.x + b->box.w; x++) {
      if (ms_get(img, x, y) < 128) continue;  // skip background pixels
      if (labels[y * img.w + x] != b->label) continue;
      int sum = (int)x + (int)y, diff = (int)x - (int)y;
      if (sum < min_sum) min_sum = sum, tl = (struct ms_point){x, y};
      if (sum > max_sum) max_sum = sum, br = (struct ms_point){x, y};
      if (diff < min_diff) min_diff = diff, bl = (struct ms_point){x, y};
      if (diff > max_diff) max_diff = diff, tr = (struct ms_point){x, y};
    }
  }
  c[0] = tl, c[1] = tr, c[2] = br, c[3] = bl;
}

ms_API void ms_perspective_correct(struct ms_image dst, struct ms_image src, struct ms_point c[4]) {
  ms_assert(ms_valid(dst) && ms_valid(src));
  float w = dst.w - 1.0f, h = dst.h - 1.0f;
  ms_for(dst, x, y) {
    float u = x / w, v = y / h;
    float top_x = c[0].x * (1 - u) + c[1].x * u;
    float top_y = c[0].y * (1 - u) + c[1].y * u;
    float bot_x = c[3].x * (1 - u) + c[2].x * u;
    float bot_y = c[3].y * (1 - u) + c[2].y * u;
    float src_x = top_x * (1 - v) + bot_x * v;
    float src_y = top_y * (1 - v) + bot_y * v;
    src_x = ms_MAX(0.0f, ms_MIN(src_x, src.w - 1.0f));
    src_y = ms_MAX(0.0f, ms_MIN(src_y, src.h - 1.0f));
    unsigned sx = (unsigned)src_x, sy = (unsigned)src_y;
    unsigned sx1 = ms_MIN(sx + 1, src.w - 1), sy1 = ms_MIN(sy + 1, src.h - 1);
    float dx = src_x - sx, dy = src_y - sy;
    uint8_t c00 = ms_get(src, sx, sy), c01 = ms_get(src, sx1, sy), c10 = ms_get(src, sx, sy1),
            c11 = ms_get(src, sx1, sy1);
    dst.data[y * dst.w + x] = (uint8_t)((c00 * (1 - dx) * (1 - dy)) + (c01 * dx * (1 - dy)) +
                                        (c10 * (1 - dx) * dy) + (c11 * dx * dy));
  }
}

ms_API void ms_trace_contour(struct ms_image img, struct ms_image visited, struct ms_contour *c) {
  ms_assert(ms_valid(img) && ms_valid(visited) && img.w == visited.w && img.h == visited.h);
  static const int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
  static const int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};

  c->length = 0;
  c->box = (struct ms_rect){c->start.x, c->start.y, 1, 1};

  struct ms_point p = c->start;
  unsigned dir = 7, seenstart = 0;

  for (;;) {
    if (!ms_get(visited, p.x, p.y)) c->length++;
    ms_set(visited, p.x, p.y, 255);
    int ndir = (dir + 1) % 8, found = 0;
    for (int i = 0; i < 8; i++) {
      int d = (ndir + i) % 8, nx = p.x + dx[d], ny = p.y + dy[d];
      if (nx >= 0 && nx < (int)img.w && ny >= 0 && ny < (int)img.h && ms_get(img, nx, ny) > 128) {
        p = (struct ms_point){(unsigned)nx, (unsigned)ny};
        dir = (d + 6) % 8;
        found = 1;
        break;
      }
    }
    if (!found) break;  // open contour
    c->box.x = ms_MIN(c->box.x, p.x);
    c->box.y = ms_MIN(c->box.y, p.y);
    c->box.w = ms_MAX(c->box.w, p.x - c->box.x + 1);
    c->box.h = ms_MAX(c->box.h, p.y - c->box.y + 1);
    if (p.x == c->start.x && p.y == c->start.y) {
      if (seenstart) break;  // stop: second time at the starting point
      seenstart = 1;
    }
  }
}

ms_API unsigned ms_fast(struct ms_image img, struct ms_image scoremap, struct ms_keypoint *kps,
                        unsigned nkps, unsigned threshold) {
  ms_assert(ms_valid(img) && kps && nkps > 0);
  static const int dx[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
  static const int dy[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};
  unsigned n = 0;
  // first pass: compute score map
  for (unsigned y = 3; y < img.h - 3; y++) {
    for (unsigned x = 3; x < img.w - 3; x++) {
      uint8_t p = ms_get(img, x, y);
      int run = 0, score = 0;
      for (int i = 0; i < 16 + 9; i++) {
        int idx = (i % 16);
        uint8_t v = ms_get(img, x + dx[idx], y + dy[idx]);
        if (v > p + threshold) {
          run = (run > 0) ? run + 1 : 1;
        } else if (v < p - threshold) {
          run = (run < 0) ? run - 1 : -1;
        } else {
          run = 0;
        }
        if (run >= 9 || run <= -9) {
          score = 255;
          for (int j = 0; j < 16; j++) {
            int d = ms_get(img, x + dx[j], y + dy[j]) - p;
            if (d < 0) d = -d;
            if (d < score) score = d;
          }
          break;
        }
      }
      ms_set(scoremap, x, y, score);
    }
  }
  // second pass: non-maximum suppression
  for (unsigned y = 3; y < img.h - 3; y++) {
    for (unsigned x = 3; x < img.w - 3; x++) {
      int s = ms_get(scoremap, x, y), is_max = 1;
      if (s == 0) continue;
      for (int yy = -1; yy <= 1 && is_max; yy++) {
        for (int xx = -1; xx <= 1; xx++) {
          if (xx == 0 && yy == 0) continue;
          if (ms_get(scoremap, x + xx, y + yy) > s) {
            is_max = 0;
            break;
          }
        }
      }
      if (is_max && n < nkps) kps[n++] = (struct ms_keypoint){{x, y}, (unsigned)s, 0, {0}};
    }
  }
  return n;
}

//
// ORB (Oriented FAST and Rotated BRIEF)
//

// clang-format: off
static const int ms_brief_pattern[256][4] = {
    {1, 0, 1, 3},       {0, 0, 3, 2},       {-1, 1, -1, -1},     {0, -4, -3, -1},
    {-2, 1, -2, -3},    {3, 0, 0, -3},      {-1, 0, -2, 1},      {-1, -1, -1, 4},
    {0, -2, 2, -2},     {0, -4, -3, 0},     {1, 0, 0, -1},       {-3, -1, -1, 2},
    {1, -4, 1, -1},     {-1, 1, 2, 2},      {-2, -1, 1, 2},      {-1, 0, -2, -2},
    {2, 3, 0, 2},       {1, -1, 1, 3},      {0, 3, -5, 2},       {0, -1, 0, -4},
    {0, 1, 3, -1},      {-2, -1, 2, 1},     {-1, 1, 0, 2},       {-1, -1, -1, -3},
    {1, 1, 0, 0},       {-3, -1, -1, -2},   {0, 1, 4, 0},        {1, 0, -4, 0},
    {0, 5, 0, 1},       {0, -2, 2, 2},      {2, -2, 3, -3},      {1, 4, -2, -1},
    {0, -1, -3, 0},     {-2, 1, -2, 3},     {-2, -1, 2, -2},     {0, 3, -3, 0},
    {1, 2, -2, -3},     {1, 1, 1, 1},       {-1, 0, 1, -1},      {4, 1, -2, 1},
    {-2, 2, 2, -2},     {2, 1, 2, 4},       {0, -2, -2, -2},     {0, 1, 1, 2},
    {0, 3, -1, 5},      {1, -2, -2, 1},     {0, 1, 1, 0},        {-2, -3, -1, 2},
    {0, -2, 0, 1},      {-2, 0, 0, -2},     {1, 1, 2, 2},        {-3, -2, 1, 1},
    {1, 8, 1, 2},       {2, 1, -1, 2},      {-2, 0, -1, 0},      {5, -4, 1, -3},
    {-1, 2, 0, -2},     {-1, 1, -1, 0},     {0, -1, 4, 1},       {-4, 0, -1, 2},
    {-2, 0, 1, 2},      {-2, -1, -1, -1},   {4, 1, -3, 2},       {4, 2, -3, -1},
    {3, -1, 1, 2},      {-2, 0, -6, -2},    {-1, -2, 3, -3},     {-1, 0, 3, -3},
    {2, 0, -2, 1},      {0, -1, 0, -1},     {0, 1, 3, -2},       {4, -4, 0, 1},
    {1, -1, 0, -1},     {-1, 2, 1, -1},     {2, 1, 2, 1},        {-2, -1, 1, 1},
    {0, 0, 3, -1},      {1, 0, 0, 2},       {2, 2, 3, 0},        {1, -1, 1, 0},
    {0, 1, -2, 4},      {-2, -2, 2, 2},     {1, 1, 0, -2},       {0, -1, 2, 0},
    {-2, -1, 1, -1},    {-2, 0, 0, -1},     {-1, 0, -3, -3},     {-1, 0, 1, 3},
    {2, 0, 0, -2},      {0, -1, 1, -2},     {1, 3, 0, 1},        {1, -1, 0, 0},
    {0, -2, 0, 1},      {3, 2, 4, -2},      {2, 0, 4, -2},       {-2, -1, -4, -1},
    {-2, 0, 1, 4},      {2, -1, -2, 1},     {-3, 4, 2, -1},      {-3, 3, 0, 2},
    {-3, -1, 0, 0},     {-1, 1, -2, 0},     {0, 1, 1, -2},       {-3, 3, 1, -1},
    {3, 0, 2, 0},       {4, 4, 0, 2},       {1, 3, -2, 1},       {2, -4, -2, -4},
    {-1, 1, 3, 0},      {3, -3, -3, 0},     {1, 0, -4, 0},       {-3, 1, 1, -2},
    {-1, -2, 0, 2},     {-2, 1, -1, -2},    {0, -2, -1, -2},     {4, 0, -1, 0},
    {0, 0, 1, 2},       {-1, -1, -1, -5},   {-3, 3, 3, 0},       {1, 1, 6, 2},
    {0, -2, -3, 0},     {-2, -3, -1, -2},   {3, 2, 0, 3},        {0, -2, 3, 1},
    {-2, 0, -2, -3},    {2, 4, -3, 1},      {-1, -1, -1, -2},    {0, -2, 1, 0},
    {15, -10, -14, 4},  {12, -5, -12, -1},  {-10, 6, 1, 14},     {8, -10, 3, 14},
    {9, -14, -1, -5},   {-8, 10, 3, -3},    {-4, -11, -10, 10},  {6, -12, 3, 4},
    {-15, 4, 1, -4},    {-1, -15, 10, -2},  {-10, -11, 14, -5},  {15, -12, -3, -5},
    {-13, -15, -10, 2}, {8, -6, -11, 7},    {-6, -4, -14, -3},   {-8, -14, 4, -15},
    {15, -11, -7, 1},   {-7, -5, -1, 8},    {-10, 7, -13, 14},   {15, 1, -11, 14},
    {12, -4, 2, -2},    {5, 8, -5, -7},     {-14, -4, -13, -13}, {-15, -8, 6, 12},
    {13, -8, -5, -7},   {-11, -2, 12, 14},  {-13, 5, -11, -11},  {3, 11, -2, 10},
    {14, -12, 9, -3},   {-6, 9, 2, -8},     {-8, -9, -8, -2},    {3, 13, -10, -15},
    {7, 15, -1, -15},   {9, 1, -15, -1},    {7, -14, -2, 5},     {-8, -8, 3, -9},
    {3, -10, -10, -13}, {-9, 3, -8, -6},    {4, -1, -1, 13},     {-15, 4, 14, -9},
    {11, -12, 13, -10}, {9, -15, 13, -11},  {11, 7, -15, 14},    {-12, 6, -14, -6},
    {-11, 11, -6, -15}, {6, -10, -3, 15},   {-1, -12, -3, 8},    {4, 8, -1, 13},
    {-8, -11, 13, -1},  {-12, -4, -3, -14}, {11, 15, 3, 3},      {-12, -12, 10, -5},
    {11, -11, 4, -5},   {14, -6, -8, -10},  {-10, -8, 7, -1},    {10, -2, -5, -4},
    {10, -3, -8, 14},   {2, 9, -15, -1},    {-8, 12, -5, -4},    {-4, -12, 0, -12},
    {-11, 8, -11, -8},  {15, -6, 1, 12},    {15, 10, -7, 6},     {3, 13, -2, -8},
    {11, -7, 0, 3},     {1, 3, -6, 11},     {1, 5, -7, 7},       {3, 11, -10, -7},
    {-2, 1, 12, -6},    {-7, 1, -12, -7},   {1, -1, -4, -2},     {3, 1, 1, -5},
    {1, 5, -4, 0},      {-14, 4, 6, -7},    {3, 8, -2, 5},       {-6, 3, -7, 10},
    {-5, -5, 3, -5},    {-3, 9, -11, -2},   {-8, 1, 1, -8},      {-1, 2, 0, -2},
    {4, -3, 3, -8},     {8, -12, -11, 7},   {0, 9, -4, 0},       {-5, 8, 7, -6},
    {-2, -9, 12, -1},   {3, -9, 14, -5},    {-2, 2, 5, 3},       {-1, -10, 9, 9},
    {-8, -10, 9, -6},   {-5, 8, -8, 10},    {1, -1, 1, -6},      {4, -5, 4, -1},
    {9, 8, 9, -1},      {3, 7, -8, -1},     {-4, -11, 1, 7},     {-9, 5, 2, -2},
    {-4, -10, -12, -2}, {-12, 0, -2, 1},    {-1, -8, 2, 2},      {0, 5, 0, 11},
    {-10, 0, 5, -8},    {1, -7, -4, 5},     {6, 13, 0, -2},      {1, -2, 6, -4},
    {-9, -7, -11, 9},   {9, 11, -1, 8},     {4, 7, 7, -11},      {8, 12, -10, 2},
    {-3, 5, -2, -7},    {-9, 2, 2, 1},      {1, 0, 1, 1},        {2, -5, 4, -14},
    {-11, -1, 2, -1},   {-7, -9, -2, -11},  {10, -1, -8, -11},   {10, 3, 10, 3},
    {9, 0, -9, 1},      {4, 4, 4, 11},      {-2, 1, 0, -12},     {-2, 0, -5, -7},
    {-7, 8, -9, 1},     {-13, -3, -6, 4},   {3, -9, -4, -7},     {-11, -1, 5, -5},
    {-7, 2, 15, 0},     {-3, 2, 13, 6},     {1, 0, 2, 1},        {-7, -4, -4, 3}};
// clang-format: on

ms_API float ms_compute_orientation(struct ms_image img, unsigned x, unsigned y, unsigned r) {
  ms_assert(ms_valid(img) && x >= r && y >= r && x < img.w - r && y < img.h - r);
  float m01 = 0, m10 = 0;
  for (int dy = -(int)r; dy <= (int)r; dy++) {
    for (int dx = -(int)r; dx <= (int)r; dx++) {
      if (dx * dx + dy * dy <= (int)(r * r)) {
        uint8_t intensity = ms_get(img, x + dx, y + dy);
        m01 += dy * intensity;
        m10 += dx * intensity;
      }
    }
  }
  return ms_atan2(m01, m10);
}

ms_API void ms_brief_descriptor(struct ms_image img, struct ms_keypoint *kp) {
  ms_assert(ms_valid(img) && kp);
  int x = kp->pt.x, y = kp->pt.y;
  float angle = kp->angle, sin_a = ms_sin(angle), cos_a = ms_sin((float)(angle + 1.57079f));
  for (int i = 0; i < 8; i++) kp->descriptor[i] = 0;
  for (int i = 0; i < 256; i++) {
    float dx1 = ms_brief_pattern[i][0] * cos_a - ms_brief_pattern[i][1] * sin_a;
    float dy1 = ms_brief_pattern[i][0] * sin_a + ms_brief_pattern[i][1] * cos_a;
    float dx2 = ms_brief_pattern[i][2] * cos_a - ms_brief_pattern[i][3] * sin_a;
    float dy2 = ms_brief_pattern[i][2] * sin_a + ms_brief_pattern[i][3] * cos_a;
    int x1 = x + (int)dx1, y1 = y + (int)dy1, x2 = x + (int)dx2, y2 = y + (int)dy2;
    uint8_t intensity1 = ms_get(img, x1, y1), intensity2 = ms_get(img, x2, y2);
    if (intensity1 > intensity2) kp->descriptor[i / 32] |= (1U << (i % 32));
  }
}

static void ms_sort_keypoints(struct ms_keypoint *kps, unsigned n) {
  for (unsigned i = 0; i < n - 1; i++) {
    for (unsigned j = 0; j < n - 1 - i; j++) {
      if (kps[j].response < kps[j + 1].response) {
        struct ms_keypoint temp = kps[j];
        kps[j] = kps[j + 1];
        kps[j + 1] = temp;
      }
    }
  }
}

ms_API unsigned ms_orb_extract(struct ms_image img, struct ms_keypoint *kps, unsigned nkps,
                               unsigned threshold, uint8_t *scoremap_buffer) {
  ms_assert(ms_valid(img) && kps && nkps > 0 && scoremap_buffer);
  struct ms_image scoremap = {img.w, img.h, scoremap_buffer};
  static struct ms_keypoint candidates[5000];
  unsigned n_fast = ms_fast(img, scoremap, candidates, ms_MIN(nkps * 4, 5000), threshold);
  if (n_fast > 1) ms_sort_keypoints(candidates, n_fast);
  unsigned n_orb = 0, radius = 15;
  for (unsigned i = 0; i < n_fast && n_orb < nkps; i++) {
    unsigned x = candidates[i].pt.x, y = candidates[i].pt.y;
    if (x >= radius && y >= radius && x < img.w - radius && y < img.h - radius) {
      kps[n_orb] = candidates[i];
      kps[n_orb].angle = ms_compute_orientation(img, x, y, radius);
      ms_brief_descriptor(img, &kps[n_orb]);
      n_orb++;
    }
  }
  return n_orb;
}

static inline unsigned ms_hamming_distance(const uint32_t desc1[8], const uint32_t desc2[8]) {
  unsigned dist = 0;
  for (int i = 0; i < 8; i++) {
    uint32_t eor = desc1[i] ^ desc2[i];
    while (eor) dist += eor & 1, eor >>= 1;
  }
  return dist;
}

ms_API unsigned ms_match_orb(const struct ms_keypoint *kps1, unsigned n1,
                             const struct ms_keypoint *kps2, unsigned n2, struct ms_match *matches,
                             unsigned max_matches, float max_distance) {
  ms_assert(kps1 && kps2 && matches);
  unsigned n = 0;
  for (unsigned i = 0; i < n1 && n < max_matches; i++) {
    float best_dist = max_distance + 1, second_best = max_distance + 1;
    unsigned best_idx = 0;
    for (unsigned j = 0; j < n2; j++) {
      float d = ms_hamming_distance(kps1[i].descriptor, kps2[j].descriptor);
      if (d < best_dist)
        second_best = best_dist, best_dist = d, best_idx = j;
      else if (d < second_best)
        second_best = d;
    }
    if (best_dist <= max_distance && best_dist < 0.8f * second_best)
      matches[n++] = (struct ms_match){i, best_idx, (unsigned)best_dist};
  }
  return n;
}

//
// Template matching
//

ms_API void ms_match_template(struct ms_image img, struct ms_image tmpl, struct ms_image result) {
  ms_assert(ms_valid(img) && ms_valid(tmpl) && ms_valid(result));
  ms_assert(img.w >= tmpl.w && img.h >= tmpl.h);
  ms_assert(result.w == img.w - tmpl.w + 1 && result.h == img.h - tmpl.h + 1);

  ms_for(result, rx, ry) {
    unsigned long long sum = 0;
    for (unsigned ty = 0; ty < tmpl.h; ty++) {
      for (unsigned tx = 0; tx < tmpl.w; tx++) {
        int diff = (int)ms_get(img, rx + tx, ry + ty) - (int)ms_get(tmpl, tx, ty);
        sum += (unsigned long long)(diff * diff);
      }
    }
    // Normalize to 0-255: lower values = better match
    unsigned long long max_diff = (unsigned long long)tmpl.w * tmpl.h * 255ULL * 255ULL;
    unsigned score = (unsigned)(sum * 255ULL / max_diff);
    ms_set(result, rx, ry, (uint8_t)(255 - ms_MIN(score, 255)));
  }
}

ms_API struct ms_point ms_find_best_match(struct ms_image result) {
  ms_assert(ms_valid(result));
  struct ms_point best = {0, 0};
  uint8_t best_score = 0;
  ms_for(result, x, y) {
    uint8_t score = ms_get(result, x, y);
    if (score > best_score) {
      best_score = score;
      best.x = x;
      best.y = y;
    }
  }
  return best;
}

//
// Integral image
//

ms_API void ms_integral(struct ms_image src, unsigned *ii) {
  ms_assert(ms_valid(src) && ii);
  unsigned row = 0;
  ms_for(src, x, y) {
    if (x == 0) row = 0;
    row += ms_get(src, x, y);
    ii[y * src.w + x] = row + (y ? ii[(y - 1) * src.w + x] : 0);
  }
}

static inline uint32_t ms_integral_sum(const unsigned *ii, unsigned iw, unsigned x, unsigned y,
                                       unsigned w, unsigned h) {
  ms_assert(ii && iw > 0 && x + w <= iw);
  unsigned x2 = x + w - 1, y2 = y + h - 1;
  unsigned A = (x > 0 && y > 0) ? ii[(y - 1) * iw + (x - 1)] : 0;
  unsigned B = (y > 0) ? ii[(y - 1) * iw + x2] : 0;
  unsigned C = (x > 0) ? ii[y2 * iw + (x - 1)] : 0;
  unsigned D = ii[y2 * iw + x2];
  return D + A - B - C;
}

//
// LBP cascade detection
//

static inline int ms_lbp_code(const unsigned *ii, unsigned iw, int x, int y, int fx, int fy, int fw,
                              int fh) {
  /* 3x3 LBP grid: TL TC TR / L C R / BL BC BR */
  unsigned tl = ms_integral_sum(ii, iw, x + fx, y + fy, fw, fh);
  unsigned tc = ms_integral_sum(ii, iw, x + fx + fw, y + fy, fw, fh);
  unsigned tr = ms_integral_sum(ii, iw, x + fx + 2 * fw, y + fy, fw, fh);
  unsigned l = ms_integral_sum(ii, iw, x + fx, y + fy + fh, fw, fh);
  unsigned c = ms_integral_sum(ii, iw, x + fx + fw, y + fy + fh, fw, fh);
  unsigned r = ms_integral_sum(ii, iw, x + fx + 2 * fw, y + fy + fh, fw, fh);
  unsigned bl = ms_integral_sum(ii, iw, x + fx, y + fy + 2 * fh, fw, fh);
  unsigned bc = ms_integral_sum(ii, iw, x + fx + fw, y + fy + 2 * fh, fw, fh);
  unsigned br = ms_integral_sum(ii, iw, x + fx + 2 * fw, y + fy + 2 * fh, fw, fh);
  return ((tl >= c) << 7) | ((tc >= c) << 6) | ((tr >= c) << 5) | ((r >= c) << 4) |
         ((br >= c) << 3) | ((bc >= c) << 2) | ((bl >= c) << 1) | ((l >= c) << 0);
}

static inline int ms_lbp_match(int code, const int32_t *subsets, int n) {
  int idx = code / 32, bit = code % 32;
  return (idx < n) && (subsets[idx] & (1 << bit));
}

ms_API unsigned ms_lbp_window(const struct ms_lbp_cascade *c, const unsigned *ii, unsigned iw,
                              unsigned ih, int x, int y, float scale) {
  int win_w = (int)(c->window_w * scale), win_h = (int)(c->window_h * scale);
  if (x + win_w > (int)iw || y + win_h > (int)ih) return 0;
  for (int si = 0; si < c->nstages; si++) {
    int start = c->stage_weak_start[si], n = c->stage_nweaks[si];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
      int wi = start + i, fi = c->weak_feature_idx[wi];
      int fx = (int)(c->features[fi * 4 + 0] * scale);
      int fy = (int)(c->features[fi * 4 + 1] * scale);
      int fw = (int)(c->features[fi * 4 + 2] * scale);
      int fh = (int)(c->features[fi * 4 + 3] * scale);
      if (fw < 1) fw = 1;
      if (fh < 1) fh = 1;
      int code = ms_lbp_code(ii, iw, x, y, fx, fy, fw, fh);
      int match =
          ms_lbp_match(code, &c->subsets[c->weak_subset_offset[wi]], c->weak_num_subsets[wi]);
      sum += match ? c->weak_left_val[wi] : c->weak_right_val[wi];
    }
    if (sum < c->stage_threshold[si]) return 0;
  }
  return 1;
}

ms_API unsigned ms_lbp_detect(const struct ms_lbp_cascade *c, const unsigned *ii, unsigned iw,
                              unsigned ih, struct ms_rect *rects, unsigned max_rects,
                              float scale_factor, float min_scale, float max_scale, int step) {
  unsigned n = 0;
  for (float scale = min_scale; scale <= max_scale && n < max_rects; scale *= scale_factor) {
    int win_w = (int)(c->window_w * scale), win_h = (int)(c->window_h * scale);
    if (win_w > (int)iw || win_h > (int)ih) break;
    for (int y = 0; y + win_h <= (int)ih && n < max_rects; y += step) {
      for (int x = 0; x + win_w <= (int)iw && n < max_rects; x += step) {
        if (ms_lbp_window(c, ii, iw, ih, x, y, scale)) {
          rects[n].x = x;
          rects[n].y = y;
          rects[n].w = win_w;
          rects[n].h = win_h;
          n++;
        }
      }
    }
  }
  return n;
}

#endif  // monoscule_H
