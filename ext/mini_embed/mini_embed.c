#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctype.h>
#include "ruby.h"

#define HASH_SIZE  131071
#define MAX_DIMS   4
#define GGUF_ALIGN 32

enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
};

/* ------------------------------------------------------------------------- */
static int safe_advance(uint8_t **p, uint8_t *end, size_t sz) {
    if (*p + sz > end) return 0;
    *p += sz;
    return 1;
}

static uint32_t rd32(uint8_t **p, uint8_t *end) {
    uint32_t v = 0;
    if (!safe_advance(p, end, 4)) return 0;
    memcpy(&v, *p - 4, 4);
    return v;
}

static uint64_t rd64(uint8_t **p, uint8_t *end) {
    uint64_t v = 0;
    if (!safe_advance(p, end, 8)) return 0;
    memcpy(&v, *p - 8, 8);
    return v;
}

static char *rdstr(uint8_t **p, uint8_t *end) {
    if (*p + 8 > end) return NULL;
    uint64_t len;
    memcpy(&len, *p, 8);
    *p += 8;
    if (len == 0 || len > (1 << 20)) return NULL;
    if (*p + len > end) return NULL;
    char *s = malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, *p, len);
    s[len] = '\0';
    *p += len;
    return s;
}

static void align_to_32(uint8_t **p, uint8_t *end, uint8_t *base) {
    size_t off = *p - base;
    size_t aligned = (off + GGUF_ALIGN - 1) & ~(GGUF_ALIGN - 1);
    if (base + aligned <= end)
        *p = base + aligned;
}

/* ------------------------------------------------------------------------- */
typedef struct HashNode {
    char *key;
    int   id;
    struct HashNode *next;
} HashNode;

typedef struct {
    int        vocab_size;
    int        dim;
    char     **tokens;
    float     *float_data;
    void      *tensor_data;
    int        tensor_type;
    void      *mapped;
    size_t     mapped_size;
    HashNode **table;
} EmbedModel;

typedef struct {
    EmbedModel *model;
} ruby_embedder;

static unsigned long hash(const char *s) {
    unsigned long h = 5381;
    int c;
    while ((c = *s++)) h = ((h << 5) + h) + c;
    return h % HASH_SIZE;
}

static void hset(EmbedModel *m, char *k, int id) {
    unsigned long h = hash(k);
    HashNode *n = malloc(sizeof(*n));
    n->key = k;
    n->id = id;
    n->next = m->table[h];
    m->table[h] = n;
}

static int hget(EmbedModel *m, const char *k) {
    HashNode *n = m->table[hash(k)];
    while (n) {
        if (strcmp(n->key, k) == 0) return n->id;
        n = n->next;
    }
    return -1;
}

/* ------------------------------------------------------------------------- */
static void *map_file(const char *path, size_t *size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    *size = st.st_size;
    void *data = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) return NULL;
    return data;
}

/* ------------------------------------------------------------------------- */
static float fp16_to_fp32(uint16_t h) {
    const uint16_t sign = (h >> 15) & 1;
    const uint16_t exp  = (h >> 10) & 0x1F;
    const uint16_t mant = h & 0x3FF;
    float val;
    if (exp == 0) {
        val = (mant / 1024.0f) * 6.103515625e-5f;
    } else if (exp == 31) {
        return 0.0f;
    } else {
        val = (1.0f + mant / 1024.0f) * (1 << (exp - 15));
    }
    return sign ? -val : val;
}

/* ------------------------------------------------------------------------- */
/* Block dequantization */

static void dequantize_row_q4_0(const void *vx, float *y, int k) {
    const int nb = k / 32;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const float d = ((const float*)(x + i*34))[0];
        const uint8_t *q = x + i*34 + 4;
        for (int j = 0; j < 32; j++) {
            const int v = (q[j/2] >> (4*(j%2))) & 0x0F;
            y[i*32 + j] = (v - 8.0f) * d;
        }
    }
}

static void dequantize_row_q4_1(const void *vx, float *y, int k) {
    const int nb = k / 32;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const float d = ((const float*)(x + i*36))[0];
        const float m = ((const float*)(x + i*36))[1];
        const uint8_t *q = x + i*36 + 8;
        for (int j = 0; j < 32; j++) {
            const int v = (q[j/2] >> (4*(j%2))) & 0x0F;
            y[i*32 + j] = v * d + m;
        }
    }
}

static void dequantize_row_q5_0(const void *vx, float *y, int k) {
    const int nb = k / 32;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const float d = ((const float*)(x + i*40))[0];
        const uint8_t *qh = x + i*40 + 4;
        const uint8_t *ql = x + i*40 + 8;
        uint32_t qh32;
        memcpy(&qh32, qh, 4);
        for (int j = 0; j < 32; j++) {
            const uint8_t vh = (qh32 >> j) & 1;
            const int v = ((ql[j/2] >> (4*(j%2))) & 0x0F) | (vh << 4);
            y[i*32 + j] = (v - 16.0f) * d;
        }
    }
}

static void dequantize_row_q5_1(const void *vx, float *y, int k) {
    const int nb = k / 32;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const float d = ((const float*)(x + i*44))[0];
        const float m = ((const float*)(x + i*44))[1];
        const uint8_t *qh = x + i*44 + 8;
        const uint8_t *ql = x + i*44 + 12;
        uint32_t qh32;
        memcpy(&qh32, qh, 4);
        for (int j = 0; j < 32; j++) {
            const uint8_t vh = (qh32 >> j) & 1;
            const int v = ((ql[j/2] >> (4*(j%2))) & 0x0F) | (vh << 4);
            y[i*32 + j] = v * d + m;
        }
    }
}

static void dequantize_row_q8_0(const void *vx, float *y, int k) {
    const int nb = k / 32;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const float d = ((const float*)(x + i*36))[0];
        const int8_t *q = (const int8_t*)(x + i*36 + 4);
        for (int j = 0; j < 32; j++) {
            y[i*32 + j] = (float)q[j] * d;
        }
    }
}

static void dequantize_row_q8_1(const void *vx, float *y, int k) {
    const int nb = k / 32;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const float d = ((const float*)(x + i*40))[0];
        const float s = ((const float*)(x + i*40))[1];
        const int8_t *q = (const int8_t*)(x + i*40 + 8);
        for (int j = 0; j < 32; j++) {
            y[i*32 + j] = (float)q[j] * d + s;
        }
    }
}

/* K-quants */
static void dequantize_row_q2_K(const void *vx, float *y, int k) {
    const int nb = k / 256;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const float d = ((const float*)(x + i*336))[0];
        const float m = ((const float*)(x + i*336))[1];
        const uint8_t *q = x + i*336 + 8;
        const uint8_t *scales = q + 64;
        for (int j = 0; j < 256; j += 32) {
            const uint8_t ls = scales[j/32] & 0xF;
            const uint8_t ms = scales[j/32] >> 4;
            for (int l = 0; l < 32; l++) {
                const int v = (q[(j+l)/4] >> (2*((j+l)%4))) & 0x03;
                const float dl = d * (ls - 32);
                const float ml = m * (ms - 32);
                y[i*256 + j + l] = v * dl + ml;
            }
        }
    }
}

static void dequantize_row_q3_K(const void *vx, float *y, int k) {
    const int nb = k / 256;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const float d = ((const float*)(x + i*352))[0];
        const uint8_t *q = x + i*352 + 4;
        const uint8_t *scales = q + 256;
        const uint8_t *h = scales + 32;
        for (int j = 0; j < 256; j += 64) {
            const uint8_t ls1 = scales[j/64] & 0x1F;
            const uint8_t ls2 = (scales[j/64] >> 4) | ((scales[j/64+1] & 0x0F) << 4);
            const uint8_t ms = scales[j/64+1] >> 4;
            for (int l = 0; l < 64; l++) {
                int v = (q[(j+l)/2] >> (4*((j+l)%2))) & 0x0F;
                const int bit = (h[(j+l)/8] >> ((j+l)%8)) & 1;
                v |= bit << 4;
                const float dl = d * (ls1 - 32);
                const float ml = (l < 32) ? (ls2 - 32) * d : (ms - 32) * d;
                y[i*256 + j + l] = v * dl + ml;
            }
        }
    }
}

static void dequantize_row_q4_K(const void *vx, float *y, int k) {
    const int nb = k / 256;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const float d = ((const float*)(x + i*416))[0];
        const float m = ((const float*)(x + i*416))[1];
        const uint8_t *q = x + i*416 + 8;
        const uint8_t *scales = q + 128;
        for (int j = 0; j < 256; j += 32) {
            const uint8_t ls = scales[j/32] & 0x3F;
            const uint8_t ms = scales[j/32] >> 6;
            for (int l = 0; l < 32; l++) {
                const int v = (q[(j+l)/2] >> (4*((j+l)%2))) & 0x0F;
                const float dl = d * (ls - 32);
                const float ml = m * (ms - 2);
                y[i*256 + j + l] = v * dl + ml;
            }
        }
    }
}

static void dequantize_row_q5_K(const void *vx, float *y, int k) {
    const int nb = k / 256;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const float d = ((const float*)(x + i*448))[0];
        const float m = ((const float*)(x + i*448))[1];
        const uint8_t *q = x + i*448 + 8;
        const uint8_t *qh = q + 128;
        const uint8_t *scales = qh + 32;
        for (int j = 0; j < 256; j += 32) {
            const uint8_t ls = scales[j/32] & 0x3F;
            const uint8_t ms = scales[j/32] >> 6;
            for (int l = 0; l < 32; l++) {
                int v = (q[(j+l)/2] >> (4*((j+l)%2))) & 0x0F;
                const int bit = (qh[(j+l)/8] >> ((j+l)%8)) & 1;
                v |= bit << 4;
                const float dl = d * (ls - 32);
                const float ml = m * (ms - 2);
                y[i*256 + j + l] = v * dl + ml;
            }
        }
    }
}

static void dequantize_row_q6_K(const void *vx, float *y, int k) {
    const int nb = k / 256;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const float d = ((const float*)(x + i*480))[0];
        const uint8_t *q = x + i*480 + 4;
        const uint8_t *qh = q + 256;
        const uint8_t *scales = qh + 64;
        for (int j = 0; j < 256; j += 64) {
            const uint8_t ls = scales[j/64];
            for (int l = 0; l < 64; l++) {
                int v = (q[(j+l)/2] >> (4*((j+l)%2))) & 0x0F;
                const int bit = (qh[(j+l)/4] >> (2*((j+l)%4))) & 0x03;
                v |= bit << 4;
                y[i*256 + j + l] = v * d * (ls - 32);
            }
        }
    }
}

static void dequantize_row_q8_K(const void *vx, float *y, int k) {
    const int nb = k / 256;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const float d = ((const float*)(x + i*544))[0];
        const int8_t *q = (const int8_t*)(x + i*544 + 4);
        const uint8_t *scales = (const uint8_t*)(q + 256);
        for (int j = 0; j < 256; j += 32) {
            const uint8_t ls = scales[j/32];
            for (int l = 0; l < 32; l++) {
                y[i*256 + j + l] = (float)q[j+l] * d * ls;
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
static float* dequantize_tensor(const void *data, int type, int n_rows, int n_cols) {
    if (type == GGML_TYPE_F32) {
        float *out = malloc(n_rows * n_cols * sizeof(float));
        if (!out) return NULL;
        memcpy(out, data, n_rows * n_cols * sizeof(float));
        return out;
    }
    if (type == GGML_TYPE_F16) {
        float *out = malloc(n_rows * n_cols * sizeof(float));
        if (!out) return NULL;
        const uint16_t *in = data;
        for (int i = 0; i < n_rows * n_cols; i++) {
            out[i] = fp16_to_fp32(in[i]);
        }
        return out;
    }

    float *out = malloc(n_rows * n_cols * sizeof(float));
    if (!out) return NULL;
    const uint8_t *in = data;
    size_t row_bytes = 0;
    void (*dequant_func)(const void*, float*, int) = NULL;

    switch (type) {
        case GGML_TYPE_Q4_0: dequant_func = dequantize_row_q4_0; row_bytes = (n_cols / 32) * 34; break;
        case GGML_TYPE_Q4_1: dequant_func = dequantize_row_q4_1; row_bytes = (n_cols / 32) * 36; break;
        case GGML_TYPE_Q5_0: dequant_func = dequantize_row_q5_0; row_bytes = (n_cols / 32) * 40; break;
        case GGML_TYPE_Q5_1: dequant_func = dequantize_row_q5_1; row_bytes = (n_cols / 32) * 44; break;
        case GGML_TYPE_Q8_0: dequant_func = dequantize_row_q8_0; row_bytes = (n_cols / 32) * 36; break;
        case GGML_TYPE_Q8_1: dequant_func = dequantize_row_q8_1; row_bytes = (n_cols / 32) * 40; break;
        case GGML_TYPE_Q2_K: dequant_func = dequantize_row_q2_K; row_bytes = (n_cols / 256) * 336; break;
        case GGML_TYPE_Q3_K: dequant_func = dequantize_row_q3_K; row_bytes = (n_cols / 256) * 352; break;
        case GGML_TYPE_Q4_K: dequant_func = dequantize_row_q4_K; row_bytes = (n_cols / 256) * 416; break;
        case GGML_TYPE_Q5_K: dequant_func = dequantize_row_q5_K; row_bytes = (n_cols / 256) * 448; break;
        case GGML_TYPE_Q6_K: dequant_func = dequantize_row_q6_K; row_bytes = (n_cols / 256) * 480; break;
        case GGML_TYPE_Q8_K: dequant_func = dequantize_row_q8_K; row_bytes = (n_cols / 256) * 544; break;
        default:
            free(out);
            return NULL;
    }

    for (int r = 0; r < n_rows; r++) {
        dequant_func(in + r * row_bytes, out + r * n_cols, n_cols);
    }
    return out;
}

/* ------------------------------------------------------------------------- */
static int skip_value(uint8_t **p, uint8_t *end, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: return safe_advance(p, end, 1);
        case 2: case 3:         return safe_advance(p, end, 2);
        case 4: case 5: case 6: return safe_advance(p, end, 4);
        case 8: {
            uint64_t len = rd64(p, end);
            return safe_advance(p, end, len);
        }
        case 9: {
            uint32_t subtype = rd32(p, end);
            uint64_t n = rd64(p, end);
            for (uint64_t i = 0; i < n; i++)
                if (!skip_value(p, end, subtype)) return 0;
            return 1;
        }
        default: return 0;
    }
}

/* ------------------------------------------------------------------------- */
static void free_model_contents(EmbedModel *m) {
    if (!m) return;
    if (m->tokens) {
        for (int i = 0; i < m->vocab_size; i++) free(m->tokens[i]);
        free(m->tokens);
    }
    if (m->table) {
        for (int i = 0; i < HASH_SIZE; i++) {
            HashNode *n = m->table[i];
            while (n) {
                HashNode *next = n->next;
                free(n);
                n = next;
            }
        }
        free(m->table);
    }
    if (m->float_data) free(m->float_data);
    if (m->mapped) munmap(m->mapped, m->mapped_size);
    free(m);
}

/* ------------------------------------------------------------------------- */
static int is_printable_string(const char *s, size_t len) {
    for (size_t i = 0; i < len; i++)
        if (!isprint((unsigned char)s[i])) return 0;
    return 1;
}

/* Fallback: find the start of tensor info by scanning for a valid string */
static uint8_t *find_tensor_info_start(uint8_t *cur, uint8_t *end) {
    uint8_t *scan = cur;
    while (scan + 8 < end) {
        uint64_t len;
        memcpy(&len, scan, 8);
        if (len > 0 && len < 256 && scan + 8 + len <= end) {
            if (is_printable_string((char*)scan + 8, len)) {
                return scan;
            }
        }
        scan++;
    }
    return NULL;
}

/* ------------------------------------------------------------------------- */
static EmbedModel *embed_load_gguf(const char *path) {
    size_t sz;
    uint8_t *base = map_file(path, &sz);
    if (!base) return NULL;
    uint8_t *cur = base;
    uint8_t *end = base + sz;

    if (memcmp(cur, "GGUF", 4) != 0) { munmap(base, sz); return NULL; }
    cur += 4;
    uint32_t version = rd32(&cur, end);
    (void)version;
    uint64_t n_tensors = rd64(&cur, end);
    uint64_t n_kv = rd64(&cur, end);

    EmbedModel *m = calloc(1, sizeof(*m));
    if (!m) { munmap(base, sz); return NULL; }
    m->mapped = base;
    m->mapped_size = sz;
    m->table = calloc(HASH_SIZE, sizeof(HashNode*));
    if (!m->table) { free_model_contents(m); return NULL; }

    /* ---------- Metadata ---------- */
    int vocab_found = 0;
    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = rdstr(&cur, end);
        if (!key) { free_model_contents(m); return NULL; }
        uint32_t type = rd32(&cur, end);

        if ((strcmp(key, "tokenizer.ggml.tokens") == 0 ||
             strcmp(key, "tokenizer.ggml.token_list") == 0) && type == 9) {
            uint32_t subtype = rd32(&cur, end);
            uint64_t n = rd64(&cur, end);
            if (subtype != 8) { free(key); free_model_contents(m); return NULL; }
            m->tokens = malloc(sizeof(char*) * n);
            if (!m->tokens) { free(key); free_model_contents(m); return NULL; }
            m->vocab_size = (int)n;
            for (uint64_t j = 0; j < n; j++) {
                char *tok = rdstr(&cur, end);
                if (!tok) tok = strdup("");
                m->tokens[j] = tok;
                hset(m, tok, (int)j);
            }
            vocab_found = 1;
        } else {
            if (!skip_value(&cur, end, type)) {
                free(key); free_model_contents(m); return NULL;
            }
        }
        free(key);
    }

    if (!vocab_found) { free_model_contents(m); return NULL; }

    uint8_t *after_kv = cur;
    align_to_32(&cur, end, base);
    uint8_t *tensor_start = cur;

    /* ---------- Tensor info ---------- */
    int embd_found = 0;
    void *raw_tensor_data = NULL;
    int tensor_type = -1;
    uint64_t dim0 = 0, dim1 = 0;
    int need_transpose = 0;

    for (int attempt = 0; attempt < 2; attempt++) {
        cur = tensor_start;
        for (uint64_t i = 0; i < n_tensors; i++) {
            char *name = rdstr(&cur, end);
            if (!name) break;
            uint32_t n_dims = rd32(&cur, end);
            uint64_t dims[MAX_DIMS] = {0};
            for (uint32_t d = 0; d < n_dims && d < MAX_DIMS; d++)
                dims[d] = rd64(&cur, end);
            uint32_t type   = rd32(&cur, end);
            uint64_t offset = rd64(&cur, end);

            int is_token_embd = (strcmp(name, "token_embd.weight") == 0 ||
                                 strcmp(name, "embeddings.word_embeddings.weight") == 0 ||
                                 strcmp(name, "model.embed_tokens.weight") == 0);

            if (!is_token_embd && n_dims == 2 && m->vocab_size > 0) {
                if ((uint64_t)m->vocab_size == dims[0] && strstr(name, "embd") != NULL)
                    is_token_embd = 1;
                else if ((uint64_t)m->vocab_size == dims[1] && strstr(name, "embd") != NULL)
                    is_token_embd = 1;
            }

            if (!embd_found && is_token_embd) {
                if (n_dims < 2 || dims[1] == 0) { free(name); free_model_contents(m); return NULL; }
                dim0 = dims[0];
                dim1 = dims[1];
                if (dim0 == (uint64_t)m->vocab_size) {
                    m->dim = (int)dim1;
                    need_transpose = 0;
                } else if (dim1 == (uint64_t)m->vocab_size) {
                    m->dim = (int)dim0;
                    need_transpose = 1;
                } else {
                    m->dim = (dim0 < dim1) ? (int)dim0 : (int)dim1;
                    need_transpose = (dim0 > dim1) ? 1 : 0;
                }
                raw_tensor_data = base + offset;
                tensor_type = type;
                embd_found = 1;
            }
            free(name);
        }
        if (embd_found) break;
        if (attempt == 0) {
            tensor_start = find_tensor_info_start(after_kv, end);
            if (!tensor_start) break;
        }
    }

    if (!embd_found || m->dim == 0) {
        free_model_contents(m);
        return NULL;
    }

    /* Dequantize */
    if (tensor_type == GGML_TYPE_F32 && !need_transpose) {
        m->float_data = NULL;
        m->tensor_data = raw_tensor_data;
    } else {
        int n_rows = need_transpose ? (int)dim1 : (int)dim0;
        int n_cols = need_transpose ? (int)dim0 : (int)dim1;
        m->float_data = dequantize_tensor(raw_tensor_data, tensor_type, n_rows, n_cols);
        if (!m->float_data) {
            free_model_contents(m);
            return NULL;
        }
        m->tensor_data = m->float_data;
    }
    m->tensor_type = tensor_type;

    return m;
}

/* ------------------------------------------------------------------------- */
static void embed_text(EmbedModel *m, const char *txt, float *out) {
    memset(out, 0, sizeof(float) * m->dim);
    char *copy = strdup(txt);
    if (!copy) return;

    char *tok = strtok(copy, " ");
    int used = 0;
    const float *embd_matrix = m->tensor_data;

    while (tok) {
        int id = hget(m, tok);
        if (id >= 0 && id < m->vocab_size) {
            const float *vec = embd_matrix + id * m->dim;
            for (int i = 0; i < m->dim; i++) out[i] += vec[i];
            used++;
        }
        tok = strtok(NULL, " ");
    }

    if (used > 0) {
        float inv = 1.0f / used;
        for (int i = 0; i < m->dim; i++) out[i] *= inv;
    }
    free(copy);
}

/* ------------------------------------------------------------------------- */
static void rb_embedder_free(void *p) {
    ruby_embedder *e = p;
    if (!e) return;
    if (e->model) free_model_contents(e->model);
    free(e);
}

static size_t rb_embedder_memsize(const void *p) {
    return sizeof(ruby_embedder);
}

static const rb_data_type_t ruby_embedder_type = {
    "MiniEmbed",
    {NULL, rb_embedder_free, rb_embedder_memsize, NULL},
    NULL, NULL, RUBY_TYPED_FREE_IMMEDIATELY
};

static VALUE rb_embedder_alloc(VALUE klass) {
    ruby_embedder *e = calloc(1, sizeof(*e));
    return TypedData_Wrap_Struct(klass, &ruby_embedder_type, e);
}

static VALUE rb_embedder_initialize(VALUE self, VALUE opts) {
    ruby_embedder *e;
    TypedData_Get_Struct(self, ruby_embedder, &ruby_embedder_type, e);

    VALUE path = rb_hash_aref(opts, ID2SYM(rb_intern("model")));
    const char *cpath = StringValueCStr(path);
    e->model = embed_load_gguf(cpath);
    if (!e->model)
        rb_raise(rb_eRuntimeError, "failed to load GGUF model");
    return self;
}

static VALUE rb_embed(VALUE self, VALUE opts) {
    ruby_embedder *e;
    TypedData_Get_Struct(self, ruby_embedder, &ruby_embedder_type, e);

    VALUE text = rb_hash_aref(opts, ID2SYM(rb_intern("text")));
    const char *ctext = StringValueCStr(text);

    VALUE out = rb_str_new(NULL, e->model->dim * sizeof(float));
    embed_text(e->model, ctext, (float*)RSTRING_PTR(out));
    return out;
}

void Init_mini_embed(void) {
    VALUE c = rb_define_class("MiniEmbed", rb_cObject);
    rb_define_alloc_func(c, rb_embedder_alloc);
    rb_define_method(c, "initialize", rb_embedder_initialize, 1);
    rb_define_method(c, "embeddings", rb_embed, 1);
}