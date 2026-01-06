#include <stddef.h>

// A simple bump allocator for our WASM module since we don't have stdlib.
#define MEMORY_HEAP_SIZE (1024 * 1024 * 8)  // 8MB heap for a few buffers
static unsigned char memory_heap[MEMORY_HEAP_SIZE];
static size_t heap_ptr = 0;
void* ms_alloc(size_t size) {
  size_t aligned_size = (size + 7) & ~7;
  if (heap_ptr + aligned_size > MEMORY_HEAP_SIZE) return NULL;
  void* ptr = &memory_heap[heap_ptr];
  heap_ptr += aligned_size;
  return ptr;
}
void ms_free(void* ptr) { (void)ptr; /* No-op for bump allocator */ }
void ms_reset_allocator(void) { heap_ptr = 0; }

// Minimal standard library functions for WASM nostdlib build
void* memset(void* s, int c, size_t n) {
  unsigned char* p = (unsigned char*)s;
  for (size_t i = 0; i < n; i++) p[i] = (unsigned char)c;
  return s;
}

void* memcpy(void* dest, const void* src, size_t n) {
  unsigned char* d = (unsigned char*)dest;
  const unsigned char* s = (const unsigned char*)src;
  for (size_t i = 0; i < n; i++) d[i] = s[i];
  return dest;
}

#define ms_assert(cond)
#define ms_NO_STDLIB
#define ms_API
#include "../../monoscule.h"
#include "../nanomagick/frontalface.h"

#define NUM_BUFFERS 3
static struct ms_image images[NUM_BUFFERS];

// Functions to be exported to WASM
void ms_init_image(int idx, int w, int h) {
  if (idx < 0 || idx >= NUM_BUFFERS) return;
  // Simple bump allocation, so we just allocate once on first use.
  // ms_reset_allocator() must be called from JS if sizes change.
  if (images[idx].data == NULL) { images[idx].data = ms_alloc(w * h); }
  images[idx].w = w;
  images[idx].h = h;
}

uint8_t* ms_get_image_data(int idx) {
  if (idx < 0 || idx >= NUM_BUFFERS) return NULL;
  return images[idx].data;
}

void ms_copy_image(int dst_idx, int src_idx) {
  if (dst_idx < 0 || dst_idx >= NUM_BUFFERS || src_idx < 0 || src_idx >= NUM_BUFFERS) return;
  ms_copy(images[dst_idx], images[src_idx]);
}

void ms_blur_image(int dst_idx, int src_idx, int radius) {
  if (dst_idx < 0 || dst_idx >= NUM_BUFFERS || src_idx < 0 || src_idx >= NUM_BUFFERS) return;
  ms_blur(images[dst_idx], images[src_idx], radius);
}

uint8_t ms_otsu_threshold_image(int src_idx) {
  if (src_idx < 0 || src_idx >= NUM_BUFFERS) return 0;
  return ms_otsu_threshold(images[src_idx]);
}

void ms_threshold_image(int img_idx, uint8_t threshold) {
  if (img_idx < 0 || img_idx >= NUM_BUFFERS) return;
  ms_threshold(images[img_idx], threshold);
}

void ms_adaptive_threshold_image(int dst_idx, int src_idx, int block_size) {
  if (dst_idx < 0 || dst_idx >= NUM_BUFFERS || src_idx < 0 || src_idx >= NUM_BUFFERS) return;
  ms_adaptive_threshold(images[dst_idx], images[src_idx], block_size | 1, 2);
}

void ms_erode_image(int dst_idx, int src_idx) {
  if (dst_idx < 0 || dst_idx >= NUM_BUFFERS || src_idx < 0 || src_idx >= NUM_BUFFERS) return;
  ms_erode(images[dst_idx], images[src_idx]);
}

void ms_erode_image_iterations(int dst_idx, int src_idx, int iterations) {
  if (dst_idx < 0 || dst_idx >= NUM_BUFFERS || src_idx < 0 || src_idx >= NUM_BUFFERS) return;

  // First iteration
  ms_erode(images[dst_idx], images[src_idx]);

  // Additional iterations (ping-pong between buffers)
  int temp_idx = (dst_idx + 1) % NUM_BUFFERS;
  if (temp_idx == src_idx) temp_idx = (temp_idx + 1) % NUM_BUFFERS;

  for (int i = 1; i < iterations; i++) {
    if (i % 2 == 1) {
      ms_erode(images[temp_idx], images[dst_idx]);
    } else {
      ms_erode(images[dst_idx], images[temp_idx]);
    }
  }

  // Ensure result is in dst_idx
  if (iterations > 1 && iterations % 2 == 0) { ms_copy(images[dst_idx], images[temp_idx]); }
}

void ms_dilate_image(int dst_idx, int src_idx) {
  if (dst_idx < 0 || dst_idx >= NUM_BUFFERS || src_idx < 0 || src_idx >= NUM_BUFFERS) return;
  ms_dilate(images[dst_idx], images[src_idx]);
}

void ms_dilate_image_iterations(int dst_idx, int src_idx, int iterations) {
  if (dst_idx < 0 || dst_idx >= NUM_BUFFERS || src_idx < 0 || src_idx >= NUM_BUFFERS) return;

  // First iteration
  ms_dilate(images[dst_idx], images[src_idx]);

  // Additional iterations (ping-pong between buffers)
  int temp_idx = (dst_idx + 1) % NUM_BUFFERS;
  if (temp_idx == src_idx) temp_idx = (temp_idx + 1) % NUM_BUFFERS;

  for (int i = 1; i < iterations; i++) {
    if (i % 2 == 1) {
      ms_dilate(images[temp_idx], images[dst_idx]);
    } else {
      ms_dilate(images[dst_idx], images[temp_idx]);
    }
  }

  // Ensure result is in dst_idx
  if (iterations > 1 && iterations % 2 == 0) { ms_copy(images[dst_idx], images[temp_idx]); }
}

void ms_sobel_image(int dst_idx, int src_idx) {
  if (dst_idx < 0 || dst_idx >= NUM_BUFFERS || src_idx < 0 || src_idx >= NUM_BUFFERS) return;
  ms_sobel(images[dst_idx], images[src_idx]);
}

// Blob detection functions
static ms_label labels_buffer[640 * 480];  // Max image size buffer for labels
static struct ms_blob blobs_buffer[200];   // Buffer for blob storage

unsigned ms_detect_blobs(int src_idx, unsigned max_blobs) {
  if (src_idx < 0 || src_idx >= NUM_BUFFERS) return 0;
  if (max_blobs > 200) max_blobs = 200;  // Limit to buffer size

  unsigned size = images[src_idx].w * images[src_idx].h;
  for (unsigned i = 0; i < size; i++) labels_buffer[i] = 0;
  for (unsigned i = 0; i < max_blobs; i++) blobs_buffer[i] = (struct ms_blob){0};

  return ms_blobs(images[src_idx], labels_buffer, blobs_buffer, max_blobs);
}

struct ms_blob* ms_get_blob(unsigned idx) {
  if (idx >= 200) return NULL;
  return &blobs_buffer[idx];
}

ms_label* ms_get_labels_buffer(void) { return labels_buffer; }

// Blob corners buffer
static struct ms_point blob_corners_buffer[4];

void ms_get_blob_corners(unsigned blob_idx) {
  if (blob_idx >= 200) return;
  ms_blob_corners(images[0], labels_buffer, &blobs_buffer[blob_idx], blob_corners_buffer);
}

struct ms_point* ms_get_blob_corner(unsigned corner_idx) {
  if (corner_idx >= 4) return NULL;
  return &blob_corners_buffer[corner_idx];
}

// Contour tracing for largest blob
static struct ms_contour largest_blob_contour;
static uint8_t contour_visited_buffer[640 * 480];

void ms_trace_largest_blob_contour(int src_idx) {
  if (src_idx < 0 || src_idx >= NUM_BUFFERS) return;

  // Find largest blob
  unsigned largest_idx = 0;
  for (unsigned i = 1; i < 200; i++) {
    if (blobs_buffer[i].area > blobs_buffer[largest_idx].area) { largest_idx = i; }
  }

  if (blobs_buffer[largest_idx].area == 0) return;

  // Set up visited buffer
  struct ms_image visited = {images[src_idx].w, images[src_idx].h, contour_visited_buffer};
  for (unsigned i = 0; i < visited.w * visited.h; i++) { visited.data[i] = 0; }

  // Find a starting point on the blob boundary
  largest_blob_contour.start =
      (struct ms_point){blobs_buffer[largest_idx].box.x, blobs_buffer[largest_idx].box.y};

  // Find actual boundary point by scanning the blob's bounding box
  for (unsigned y = blobs_buffer[largest_idx].box.y;
       y < blobs_buffer[largest_idx].box.y + blobs_buffer[largest_idx].box.h; y++) {
    for (unsigned x = blobs_buffer[largest_idx].box.x;
         x < blobs_buffer[largest_idx].box.x + blobs_buffer[largest_idx].box.w; x++) {
      if (x < images[src_idx].w && y < images[src_idx].h &&
          images[src_idx].data[y * images[src_idx].w + x] > 128) {
        largest_blob_contour.start = (struct ms_point){x, y};
        goto found_start;
      }
    }
  }

found_start:
  ms_trace_contour(images[src_idx], visited, &largest_blob_contour);
}

struct ms_contour* ms_get_largest_blob_contour(void) { return &largest_blob_contour; }

// FAST keypoint detection
static struct ms_keypoint keypoints_buffer[500];
static uint8_t scoremap_buffer[640 * 480];

unsigned ms_detect_fast_keypoints(int src_idx, unsigned threshold, unsigned max_keypoints) {
  if (src_idx < 0 || src_idx >= NUM_BUFFERS) return 0;
  if (max_keypoints > 500) max_keypoints = 500;

  struct ms_image scoremap = {images[src_idx].w, images[src_idx].h, scoremap_buffer};
  return ms_fast(images[src_idx], scoremap, keypoints_buffer, max_keypoints, threshold);
}

struct ms_keypoint* ms_get_keypoint(unsigned idx) {
  if (idx >= 500) return NULL;
  return &keypoints_buffer[idx];
}

// ORB feature extraction
static struct ms_keypoint orb_keypoints_buffer[300];
static struct ms_keypoint template_keypoints_buffer[300];
static struct ms_match matches_buffer[200];

unsigned ms_extract_orb_features(int src_idx, unsigned threshold, unsigned max_keypoints) {
  if (src_idx < 0 || src_idx >= NUM_BUFFERS) return 0;
  if (max_keypoints > 300) max_keypoints = 300;

  return ms_orb_extract(images[src_idx], orb_keypoints_buffer, max_keypoints, threshold,
                        scoremap_buffer);
}

struct ms_keypoint* ms_get_orb_keypoint(unsigned idx) {
  if (idx >= 300) return NULL;
  return &orb_keypoints_buffer[idx];
}

void ms_store_template_keypoints(unsigned count) {
  if (count > 300) count = 300;
  for (unsigned i = 0; i < count; i++) { template_keypoints_buffer[i] = orb_keypoints_buffer[i]; }
}

unsigned ms_match_orb_features(unsigned template_count, unsigned scene_count, float max_distance) {
  if (template_count > 300) template_count = 300;
  if (scene_count > 300) scene_count = 300;

  return ms_match_orb(template_keypoints_buffer, template_count, orb_keypoints_buffer, scene_count,
                      matches_buffer, 200, max_distance);
}

struct ms_match* ms_get_match(unsigned idx) {
  if (idx >= 200) return NULL;
  return &matches_buffer[idx];
}

struct ms_keypoint* ms_get_template_keypoint(unsigned idx) {
  if (idx >= 300) return NULL;
  return &template_keypoints_buffer[idx];
}

// Contour detection for largest blob
static struct ms_contour contour_buffer;
static uint8_t visited_buffer[640 * 480];

int ms_detect_largest_blob_contour(int src_idx, unsigned max_blobs) {
  if (src_idx < 0 || src_idx >= NUM_BUFFERS) return 0;

  // First detect blobs
  unsigned num_blobs = ms_detect_blobs(src_idx, max_blobs);
  if (num_blobs == 0) return 0;

  // Find largest blob
  unsigned largest_idx = 0;
  unsigned largest_area = blobs_buffer[0].area;
  for (unsigned i = 1; i < num_blobs; i++) {
    if (blobs_buffer[i].area > largest_area) {
      largest_area = blobs_buffer[i].area;
      largest_idx = i;
    }
  }

  if (largest_area < 100) return 0;  // Skip very small blobs

  // Initialize visited map
  unsigned size = images[src_idx].w * images[src_idx].h;
  for (unsigned i = 0; i < size; i++) visited_buffer[i] = 0;

  struct ms_image visited = {images[src_idx].w, images[src_idx].h, visited_buffer};

  // Find a starting point on the blob boundary
  struct ms_blob* blob = &blobs_buffer[largest_idx];
  contour_buffer.start.x = blob->box.x;
  contour_buffer.start.y = blob->box.y;

  // Find first boundary pixel
  int found = 0;
  for (unsigned y = blob->box.y; y < blob->box.y + blob->box.h && !found; y++) {
    for (unsigned x = blob->box.x; x < blob->box.x + blob->box.w && !found; x++) {
      if (labels_buffer[y * images[src_idx].w + x] == blob->label) {
        contour_buffer.start.x = x;
        contour_buffer.start.y = y;
        found = 1;
      }
    }
  }

  if (!found) return 0;

  // Trace contour
  ms_trace_contour(images[src_idx], visited, &contour_buffer);

  return contour_buffer.length > 0 ? 1 : 0;
}

struct ms_contour* ms_get_contour(void) { return &contour_buffer; }

// LBP Face detection
static uint32_t integral_buffer[640 * 480];
static struct ms_rect faces_buffer[100];

unsigned ms_detect_faces(int src_idx, int min_neighbors) {
  if (src_idx < 0 || src_idx >= NUM_BUFFERS) return 0;
  if (images[src_idx].w * images[src_idx].h > 640 * 480) return 0;

  ms_integral(images[src_idx], integral_buffer);
  return ms_lbp_detect(&frontalface, integral_buffer, images[src_idx].w, images[src_idx].h,
                       faces_buffer, 100, 1.2f, 1.0f, 4.0f, min_neighbors);
}

struct ms_rect* ms_get_face(unsigned idx) {
  if (idx >= 100) return NULL;
  return &faces_buffer[idx];
}