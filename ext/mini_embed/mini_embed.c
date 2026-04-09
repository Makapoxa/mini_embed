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
#include <limits.h>
#include "ruby.h"

#define HASH_SIZE       131071
#define MAX_DIMS        4
#define GGUF_ALIGN      32
#define MAX_MERGES      100000
#define MERGE_HASH_SIZE 65537
#define QK8_0           32
#define QK_K            256
#define K_SCALE_SIZE    12
#define MAX_DIM         16384

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

enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_NONE = 0,
    LLAMA_VOCAB_TYPE_SPM  = 1,
    LLAMA_VOCAB_TYPE_BPE  = 2,
    LLAMA_VOCAB_TYPE_WPM  = 3,
};

enum normalize_type {
    NORM_NONE = 0,
    NORM_L2   = 1,
};

/* ------------------------------------------------------------------------- */
// Unicode helper functions
static int unicode_len_utf8(char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static int unicode_is_letter(uint32_t cp) {
    return (cp >= 0x41 && cp <= 0x5A) || (cp >= 0x61 && cp <= 0x7A) ||
           (cp >= 0xC0 && cp <= 0xD6) || (cp >= 0xD8 && cp <= 0xF6) ||
           (cp >= 0xF8 && cp <= 0x2FF) || (cp >= 0x370 && cp <= 0x37D) ||
           (cp >= 0x37F && cp <= 0x1FFF) || (cp >= 0x200C && cp <= 0x200D) ||
           (cp >= 0x2070 && cp <= 0x218F) || (cp >= 0x2C00 && cp <= 0x2FEF) ||
           (cp >= 0x3001 && cp <= 0xD7FF) || (cp >= 0xF900 && cp <= 0xFDCF) ||
           (cp >= 0xFDF0 && cp <= 0xFFFD);
}

static int unicode_is_number(uint32_t cp) {
    return (cp >= 0x30 && cp <= 0x39) || (cp >= 0x660 && cp <= 0x669) ||
           (cp >= 0x6F0 && cp <= 0x6F9) || (cp >= 0x7C0 && cp <= 0x7C9) ||
           (cp >= 0x966 && cp <= 0x96F);
}

static uint32_t unicode_cpt_from_utf8(const char *s, size_t *len) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { *len = 1; return c; }
    if ((c & 0xE0) == 0xC0) { *len = 2; return ((c & 0x1F) << 6) | (s[1] & 0x3F); }
    if ((c & 0xF0) == 0xE0) { *len = 3; return ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); }
    if ((c & 0xF8) == 0xF0) { *len = 4; return ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); }
    *len = 1;
    return c;
}

/* ------------------------------------------------------------------------- */
// Pre-tokenizer (GPT-2/Llama style, replaces broken regex)
#define CHAR_CLASS_SPACE   0
#define CHAR_CLASS_LETTER  1
#define CHAR_CLASS_NUMBER  2
#define CHAR_CLASS_NEWLINE 3
#define CHAR_CLASS_OTHER   4

static int get_char_class(uint32_t cp) {
    if (unicode_is_letter(cp)) return CHAR_CLASS_LETTER;
    if (unicode_is_number(cp)) return CHAR_CLASS_NUMBER;
    if (cp == '\n' || cp == '\r') return CHAR_CLASS_NEWLINE;
    if (cp == ' ' || cp == '\t') return CHAR_CLASS_SPACE;
    return CHAR_CLASS_OTHER;
}

static int is_contraction(const char *text, size_t pos, size_t text_len) {
    if (pos >= text_len) return 0;
    unsigned char c = (unsigned char)text[pos];
    if (c != '\'' && c != 0xE2) return 0;
    if (c == 0xE2 && pos + 2 < text_len && text[pos+1] == 0x80 && (text[pos+2] == 0x99 || text[pos+2] == 0x98)) {
        if (pos + 3 >= text_len) return 0;
        char next = tolower((unsigned char)text[pos + 3]);
        return next == 's' || next == 't' || next == 'r' || next == 'v' ||
               next == 'm' || next == 'l' || next == 'd';
    }
    if (c == '\'' && pos + 1 < text_len) {
        char next = tolower((unsigned char)text[pos + 1]);
        return next == 's' || next == 't' || next == 'r' || next == 'v' ||
               next == 'm' || next == 'l' || next == 'd';
    }
    return 0;
}

static size_t contraction_len(const char *text, size_t pos) {
    unsigned char c = (unsigned char)text[pos];
    if (c == '\'') return 2;
    return 4;
}

static char** pre_tokenize(const char *text, int *num_words) {
    char **words = NULL;
    int word_count = 0, word_capacity = 0;
    size_t text_len = strlen(text);

    if (text_len == 0) {
        *num_words = 0;
        return NULL;
    }

    #define ADD_WORD(ptr, len) do { \
        char *w = malloc((len) + 1); \
        if (!w) goto error; \
        memcpy(w, ptr, len); \
        w[len] = '\0'; \
        if (word_count >= word_capacity) { \
            word_capacity = word_capacity ? word_capacity * 2 : 16; \
            char **nw = realloc(words, word_capacity * sizeof(char*)); \
            if (!nw) { free(w); goto error; } \
            words = nw; \
        } \
        words[word_count++] = w; \
    } while(0)

    size_t i = 0;
    while (i < text_len) {
        size_t char_len;
        uint32_t cp = unicode_cpt_from_utf8(text + i, &char_len);
        int cls = get_char_class(cp);

        if (cls == CHAR_CLASS_NEWLINE) {
            ADD_WORD(text + i, char_len);
            i += char_len;
            continue;
        }

        if (cls == CHAR_CLASS_SPACE) {
            size_t space_start = i;
            while (i < text_len) {
                size_t cl;
                uint32_t c = unicode_cpt_from_utf8(text + i, &cl);
                int cc = get_char_class(c);
                if (cc != CHAR_CLASS_SPACE) break;
                i += cl;
            }
            if (i >= text_len) break;
            size_t space_len = i - space_start;
            ADD_WORD(text + space_start, space_len);
            continue;
        }

        size_t start = i;
        i += char_len;

        while (i < text_len) {
            size_t cl;
            uint32_t c = unicode_cpt_from_utf8(text + i, &cl);
            int ccls = get_char_class(c);

            if (is_contraction(text, i, text_len)) {
                size_t clen = contraction_len(text, i);
                i += clen;
                continue;
            }

            if (ccls != cls) break;
            if (cls == CHAR_CLASS_NUMBER) {
                int digits = 0;
                size_t check = start;
                while (check < i) {
                    size_t dl;
                    uint32_t dc = unicode_cpt_from_utf8(text + check, &dl);
                    if (get_char_class(dc) == CHAR_CLASS_NUMBER) digits++;
                    check += dl;
                }
                if (digits >= 3) break;
            }
            i += cl;
        }

        ADD_WORD(text + start, i - start);
    }

    #undef ADD_WORD
    *num_words = word_count;
    return words;

error:
    for (int j = 0; j < word_count; j++) free(words[j]);
    free(words);
    *num_words = 0;
    return NULL;
}

/* ------------------------------------------------------------------------- */
// BPE merge structures with hash table for O(1) lookup
typedef struct MergeHashNode {
    char *left;
    char *right;
    int rank;
    struct MergeHashNode *next;
} MergeHashNode;

typedef struct {
    MergeHashNode **table;
    int table_size;
    int num_merges;
} BPEMergeTable;

static uint64_t merge_hash(const char *left, const char *right) {
    uint64_t h = 0xcbf29ce484222325ULL;
    while (*left) { h ^= (uint64_t)(unsigned char)*left++; h *= 0x100000001b3ULL; }
    h ^= (uint64_t)' ';
    h *= 0x100000001b3ULL;
    while (*right) { h ^= (uint64_t)(unsigned char)*right++; h *= 0x100000001b3ULL; }
    return h;
}

static void bpe_merge_table_init(BPEMergeTable *table) {
    table->table_size = MERGE_HASH_SIZE;
    table->table = calloc(MERGE_HASH_SIZE, sizeof(MergeHashNode*));
    table->num_merges = 0;
}

static void bpe_merge_table_add(BPEMergeTable *table, const char *left, const char *right, int rank) {
    uint64_t h = merge_hash(left, right) % table->table_size;
    MergeHashNode *n = malloc(sizeof(MergeHashNode));
    if (!n) return;
    n->left = strdup(left);
    n->right = strdup(right);
    n->rank = rank;
    n->next = table->table[h];
    table->table[h] = n;
    table->num_merges++;
}

static void bpe_merge_table_free(BPEMergeTable *table) {
    if (!table->table) return;
    for (int i = 0; i < table->table_size; i++) {
        MergeHashNode *n = table->table[i];
        while (n) {
            MergeHashNode *next = n->next;
            free(n->left);
            free(n->right);
            free(n);
            n = next;
        }
    }
    free(table->table);
    table->table = NULL;
}

static int bpe_merge_rank(const BPEMergeTable *table, const char *left, const char *right) {
    uint64_t h = merge_hash(left, right) % table->table_size;
    MergeHashNode *n = table->table[h];
    while (n) {
        if (strcmp(n->left, left) == 0 && strcmp(n->right, right) == 0)
            return n->rank;
        n = n->next;
    }
    return -1;
}

/* ------------------------------------------------------------------------- */
// BPE tokenization (correct iterative algorithm)
typedef struct {
    const char *text;
    int start, end;
    int prev, next;
    int used;
} BPESymbol;

static int text_to_id(void *vocab_data, const char *text);

static void bpe_tokenize_word(const BPEMergeTable *merges, const char *word,
                               void *vocab_data, int *token_ids, int *num_tokens) {
    int word_len = strlen(word);
    if (word_len == 0) return;

    int num_symbols = 0;
    BPESymbol *symbols = malloc(word_len * sizeof(BPESymbol));
    if (!symbols) return;

    int offset = 0;
    while (offset < word_len) {
        int char_len = unicode_len_utf8(word[offset]);
        if (offset + char_len > word_len) char_len = word_len - offset;
        symbols[num_symbols].text = word;
        symbols[num_symbols].start = offset;
        symbols[num_symbols].end = offset + char_len;
        symbols[num_symbols].prev = num_symbols - 1;
        symbols[num_symbols].next = num_symbols + 1;
        symbols[num_symbols].used = 1;
        offset += char_len;
        num_symbols++;
    }

    if (num_symbols > 0) symbols[num_symbols - 1].next = -1;

    if (num_symbols <= 1) {
        int id = text_to_id(vocab_data, word);
        if (id != -1) token_ids[(*num_tokens)++] = id;
        free(symbols);
        return;
    }

    while (1) {
        int best_rank = INT_MAX;
        int best_idx = -1;

        int idx = 0;
        while (idx != -1) {
            int next = symbols[idx].next;
            if (next != -1 && symbols[idx].used && symbols[next].used) {
                int left_len = symbols[idx].end - symbols[idx].start;
                int right_len = symbols[next].end - symbols[next].start;
                char *left_str = malloc(left_len + 1);
                char *right_str = malloc(right_len + 1);
                if (left_str && right_str) {
                    memcpy(left_str, word + symbols[idx].start, left_len);
                    left_str[left_len] = '\0';
                    memcpy(right_str, word + symbols[next].start, right_len);
                    right_str[right_len] = '\0';
                    int rank = bpe_merge_rank(merges, left_str, right_str);
                    if (rank != -1 && rank < best_rank) {
                        best_rank = rank;
                        best_idx = idx;
                    }
                }
                free(left_str);
                free(right_str);
            }
            idx = symbols[idx].next;
        }

        if (best_idx == -1) break;

        int right_idx = symbols[best_idx].next;
        symbols[best_idx].end = symbols[right_idx].end;
        symbols[best_idx].next = symbols[right_idx].next;
        symbols[right_idx].used = 0;

        if (symbols[right_idx].next != -1) {
            symbols[symbols[right_idx].next].prev = best_idx;
        }
    }

    for (int i = 0; i < num_symbols; i++) {
        if (symbols[i].used) {
            int len = symbols[i].end - symbols[i].start;
            char *substr = malloc(len + 1);
            if (substr) {
                memcpy(substr, word + symbols[i].start, len);
                substr[len] = '\0';
                int id = text_to_id(vocab_data, substr);
                if (id != -1) token_ids[(*num_tokens)++] = id;
                free(substr);
            }
        }
    }
    free(symbols);
}

/* ------------------------------------------------------------------------- */
// GGUF parsing
static int safe_advance(uint8_t **p, uint8_t *end, size_t sz) {
    if (*p + sz > end) return 0;
    *p += sz;
    return 1;
}

static uint32_t rd32(uint8_t **p, uint8_t *end) {
    uint32_t v;
    if (!safe_advance(p, end, 4)) return 0;
    memcpy(&v, *p - 4, 4);
    return v;
}

static uint64_t rd64(uint8_t **p, uint8_t *end) {
    uint64_t v;
    if (!safe_advance(p, end, 8)) return 0;
    memcpy(&v, *p - 8, 8);
    return v;
}

static char *rdstr(uint8_t **p, uint8_t *end) {
    if (*p + 8 > end) return NULL;
    uint64_t len;
    memcpy(&len, *p, 8);
    *p += 8;
    if (len == 0 || len > (1<<20)) return NULL;
    if (*p + len > end) return NULL;
    char *s = malloc(len+1);
    if (!s) return NULL;
    memcpy(s, *p, len);
    s[len] = '\0';
    *p += len;
    return s;
}

static void align_to_32(uint8_t **p, uint8_t *end, uint8_t *base) {
    size_t off = *p - base;
    size_t aligned = (off + GGUF_ALIGN - 1) & ~(GGUF_ALIGN - 1);
    if (base + aligned <= end) *p = base + aligned;
}

/* ------------------------------------------------------------------------- */
// Hash table for vocabulary
typedef struct HashNode {
    char *key;
    int id;
    struct HashNode *next;
} HashNode;

typedef struct {
    int vocab_size;
    int dim;
    char **tokens;
    void *mapped;
    size_t mapped_size;
    HashNode **table;
    BPEMergeTable merges;
    int unknown_token_id;
    int bos_token_id;
    int eos_token_id;
    int vocab_type;
    char space_marker[8];
    int space_marker_len;
    const void *raw_tensor_data;
    int tensor_type;
    size_t row_bytes;
    int need_transpose;
    uint64_t raw_dim0, raw_dim1;
    int normalize;
} EmbedModel;

typedef struct {
    EmbedModel *model;
} ruby_embedder;

static uint64_t vocab_hash(const char *s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    while (*s) {
        h ^= (uint64_t)(unsigned char)*s++;
        h *= 0x100000001b3ULL;
    }
    return h % HASH_SIZE;
}

static void hset(EmbedModel *m, char *k, int id) {
    uint64_t h = vocab_hash(k);
    HashNode *n = malloc(sizeof(*n));
    if (!n) return;
    n->key = k;
    n->id = id;
    n->next = m->table[h];
    m->table[h] = n;
}

static int hget(EmbedModel *m, const char *k) {
    if (!k || !m->table) return -1;
    HashNode *n = m->table[vocab_hash(k)];
    while (n) {
        if (strcmp(n->key, k) == 0) return n->id;
        n = n->next;
    }
    return -1;
}

static int text_to_id(void *vocab_data, const char *text) {
    return hget((EmbedModel*)vocab_data, text);
}

/* ------------------------------------------------------------------------- */
// File mapping
static void *map_file(const char *path, size_t *size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    *size = st.st_size;
    if (*size == 0) { close(fd); return NULL; }
    void *data = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    return data == MAP_FAILED ? NULL : data;
}

/* ------------------------------------------------------------------------- */
// FP16 conversion (corrected)
static float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            uint32_t e = 0;
            uint32_t m = mant;
            while (!(m & 0x400)) { m <<= 1; e++; }
            f = (sign << 31) | ((127 - 15 - e + 1) << 23) | ((m & 0x3FF) << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

/* ------------------------------------------------------------------------- */
// Block dequantization functions (correct sizes)
static void dequantize_row_q4_0(const void *vx, float *y, int k) {
    const int nb = k / QK8_0;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = x + i * 18;
        uint16_t d16;
        memcpy(&d16, block, 2);
        const float d = fp16_to_fp32(d16);
        const uint8_t *q = block + 2;
        for (int j = 0; j < 32; j++) {
            const int v = (q[j/2] >> (4*(j%2))) & 0x0F;
            y[i*32 + j] = (v - 8.0f) * d;
        }
    }
}

static void dequantize_row_q4_1(const void *vx, float *y, int k) {
    const int nb = k / QK8_0;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = x + i * 20;
        uint16_t d16, m16;
        memcpy(&d16, block, 2);
        memcpy(&m16, block + 2, 2);
        const float d = fp16_to_fp32(d16);
        const float m = fp16_to_fp32(m16);
        const uint8_t *q = block + 4;
        for (int j = 0; j < 32; j++) {
            const int v = (q[j/2] >> (4*(j%2))) & 0x0F;
            y[i*32 + j] = v * d + m;
        }
    }
}

static void dequantize_row_q5_0(const void *vx, float *y, int k) {
    const int nb = k / QK8_0;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = x + i * 22;
        uint16_t d16;
        memcpy(&d16, block, 2);
        const float d = fp16_to_fp32(d16);
        uint32_t qh32;
        memcpy(&qh32, block + 2, 4);
        const uint8_t *ql = block + 6;
        for (int j = 0; j < 32; j++) {
            const uint8_t vh = (qh32 >> j) & 1;
            const int v = ((ql[j/2] >> (4*(j%2))) & 0x0F) | (vh << 4);
            y[i*32 + j] = (v - 16.0f) * d;
        }
    }
}

static void dequantize_row_q5_1(const void *vx, float *y, int k) {
    const int nb = k / QK8_0;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = x + i * 24;
        uint16_t d16, m16;
        memcpy(&d16, block, 2);
        memcpy(&m16, block + 2, 2);
        const float d = fp16_to_fp32(d16);
        const float m = fp16_to_fp32(m16);
        uint32_t qh32;
        memcpy(&qh32, block + 4, 4);
        const uint8_t *ql = block + 8;
        for (int j = 0; j < 32; j++) {
            const uint8_t vh = (qh32 >> j) & 1;
            const int v = ((ql[j/2] >> (4*(j%2))) & 0x0F) | (vh << 4);
            y[i*32 + j] = v * d + m;
        }
    }
}

static void dequantize_row_q8_0(const void *vx, float *y, int k) {
    const int nb = k / QK8_0;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = x + i * 34;
        float d;
        memcpy(&d, block, 4);
        const int8_t *q = (const int8_t*)(block + 4);
        for (int j = 0; j < 32; j++) {
            y[i*32 + j] = (float)q[j] * d;
        }
    }
}

static void dequantize_row_q8_1(const void *vx, float *y, int k) {
    const int nb = k / QK8_0;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = x + i * 40;
        float d, s;
        memcpy(&d, block, 4);
        memcpy(&s, block + 4, 4);
        const int8_t *q = (const int8_t*)(block + 8);
        for (int j = 0; j < 32; j++) {
            y[i*32 + j] = (float)q[j] * d + s;
        }
    }
}

// K-quant scale helpers
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-3] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-1] >> 6) << 4);
    }
}

static void dequantize_row_q2_K(const void *vx, float *y, int k) {
    const int nb = k / QK_K;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = x + i * 84;
        uint16_t d16, dmin16;
        memcpy(&d16, block, 2);
        memcpy(&dmin16, block + 2, 2);
        const float d = fp16_to_fp32(d16);
        const float min = fp16_to_fp32(dmin16);
        const uint8_t *scales = block + 4;
        const uint8_t *q = block + 20;
        for (int j = 0; j < QK_K; j += 64) {
            const float dl = d * (scales[j/64] & 0xF);
            const float ml = min * (scales[j/64] >> 4);
            for (int l = 0; l < 64; l++) {
                const int v = (q[(j+l)/4] >> (2*((j+l)%4))) & 0x03;
                y[i*QK_K + j + l] = v * dl + ml;
            }
        }
    }
}

static void dequantize_row_q3_K(const void *vx, float *y, int k) {
    const int nb = k / QK_K;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = x + i * 110;
        uint16_t d16;
        memcpy(&d16, block, 2);
        const float d = fp16_to_fp32(d16);
        const uint8_t *hmask = block + 2;
        const uint8_t *q = block + 34;
        const uint8_t *scales = block + 98;
        for (int j = 0; j < QK_K; j += 64) {
            const uint8_t ls1 = scales[j/64] & 0x1F;
            const uint8_t ls2 = (scales[j/64] >> 5) | ((scales[j/64 + 1] & 0x7) << 3);
            const uint8_t ls3 = ((scales[j/64 + 1] >> 3) & 0x1F);
            const uint8_t ls4 = (scales[j/64 + 1] >> 8);
            for (int l = 0; l < 64; l++) {
                int v = (q[(j+l)/2] >> (4*((j+l)%2))) & 0x0F;
                const int bit = (hmask[(j+l)/8] >> ((j+l)%8)) & 1;
                v |= bit << 4;
                float ls;
                if (l < 16) ls = ls1;
                else if (l < 32) ls = ls2;
                else if (l < 48) ls = ls3;
                else ls = ls4;
                y[i*QK_K + j + l] = (v - 32.0f) * d * ls;
            }
        }
    }
}

static void dequantize_row_q4_K(const void *vx, float *y, int k) {
    const int nb = k / QK_K;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = x + i * 144;
        uint16_t d16, dmin16;
        memcpy(&d16, block, 2);
        memcpy(&dmin16, block + 2, 2);
        const float d = fp16_to_fp32(d16);
        const float min = fp16_to_fp32(dmin16);
        const uint8_t *scales = block + 4;
        const uint8_t *q = block + 16;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is, scales, &sc, &m);
            float d1 = d * sc;
            float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc;
            float m2 = min * m;
            for (int l = 0; l < 32; l++) {
                y[i*QK_K + j + l] = d1 * (q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; l++) {
                y[i*QK_K + j + 32 + l] = d2 * (q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

static void dequantize_row_q5_K(const void *vx, float *y, int k) {
    const int nb = k / QK_K;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = x + i * 176;
        uint16_t d16, dmin16;
        memcpy(&d16, block, 2);
        memcpy(&dmin16, block + 2, 2);
        const float d = fp16_to_fp32(d16);
        const float min = fp16_to_fp32(dmin16);
        const uint8_t *scales = block + 4;
        const uint8_t *qh = block + 16;
        const uint8_t *ql = block + 48;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is, scales, &sc, &m);
            float d1 = d * sc;
            float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc;
            float m2 = min * m;
            for (int l = 0; l < 32; l++) {
                int vh = (qh[j/64 * 4 + l/8] >> (l%8)) & 1;
                int v = (ql[l] & 0xF) | (vh << 4);
                y[i*QK_K + j + l] = d1 * v - m1;
            }
            for (int l = 0; l < 32; l++) {
                int vh = (qh[j/64 * 4 + 4 + l/8] >> (l%8)) & 1;
                int v = (ql[l] >> 4) | (vh << 4);
                y[i*QK_K + j + 32 + l] = d2 * v - m2;
            }
            ql += 32;
            is += 2;
        }
    }
}

static void dequantize_row_q6_K(const void *vx, float *y, int k) {
    const int nb = k / QK_K;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = x + i * 210;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t *scales = (const int8_t*)(block + 192);
        uint16_t d16;
        memcpy(&d16, block + 208, 2);
        const float d = fp16_to_fp32(d16);
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; l++) {
                int v = (ql[j/2 + l] & 0xF) | (((qh[j/4 + l/2] >> ((l%2)*4)) & 0xF) << 4);
                y[i*QK_K + j + l] = v * d * scales[j/128 * 8 + l/4];
            }
            for (int l = 0; l < 32; l++) {
                int v = (ql[j/2 + 32 + l] >> 4) | (((qh[j/4 + 16 + l/2] >> ((l%2)*4)) & 0xF) << 4);
                y[i*QK_K + j + 32 + l] = v * d * scales[j/128 * 8 + 8 + l/4];
            }
            for (int l = 0; l < 32; l++) {
                int v = (ql[j/2 + 64 + l] & 0xF) | (((qh[j/4 + 32 + l/2] >> ((l%2)*4)) & 0xF) << 4);
                y[i*QK_K + j + 64 + l] = v * d * scales[j/128 * 8 + 4 + l/4];
            }
            for (int l = 0; l < 32; l++) {
                int v = (ql[j/2 + 96 + l] >> 4) | (((qh[j/4 + 48 + l/2] >> ((l%2)*4)) & 0xF) << 4);
                y[i*QK_K + j + 96 + l] = v * d * scales[j/128 * 8 + 12 + l/4];
            }
        }
    }
}

static void dequantize_row_q8_K(const void *vx, float *y, int k) {
    const int nb = k / QK_K;
    const uint8_t *x = vx;
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = x + i * 292;
        float d;
        memcpy(&d, block, 4);
        const int8_t *q = (const int8_t*)(block + 4);
        for (int j = 0; j < QK_K; j++) {
            y[i*QK_K + j] = (float)q[j] * d;
        }
    }
}

// Lazy single-row dequantization
static void dequantize_row_lazy(const EmbedModel *m, int row, float *out) {
    if (!m->raw_tensor_data || row < 0 || row >= m->vocab_size) {
        memset(out, 0, sizeof(float) * m->dim);
        return;
    }

    const uint8_t *raw;
    int effective_cols;

    if (m->need_transpose) {
        int src_row_size;
        switch (m->tensor_type) {
            case GGML_TYPE_F32: src_row_size = m->raw_dim1 * sizeof(float); break;
            case GGML_TYPE_F16: src_row_size = m->raw_dim1 * sizeof(uint16_t); break;
            default: {
                size_t rb = 0;
                int nc = (int)m->raw_dim1;
                switch (m->tensor_type) {
                    case GGML_TYPE_Q4_0: rb = (nc / 32) * 18; break;
                    case GGML_TYPE_Q4_1: rb = (nc / 32) * 20; break;
                    case GGML_TYPE_Q5_0: rb = (nc / 32) * 22; break;
                    case GGML_TYPE_Q5_1: rb = (nc / 32) * 24; break;
                    case GGML_TYPE_Q8_0: rb = (nc / 32) * 34; break;
                    case GGML_TYPE_Q8_1: rb = (nc / 32) * 40; break;
                    case GGML_TYPE_Q2_K: rb = (nc / 256) * 84; break;
                    case GGML_TYPE_Q3_K: rb = (nc / 256) * 110; break;
                    case GGML_TYPE_Q4_K: rb = (nc / 256) * 144; break;
                    case GGML_TYPE_Q5_K: rb = (nc / 256) * 176; break;
                    case GGML_TYPE_Q6_K: rb = (nc / 256) * 210; break;
                    case GGML_TYPE_Q8_K: rb = (nc / 256) * 292; break;
                    default: src_row_size = 0; return;
                }
                src_row_size = (int)rb;
            }
        }
        float *temp_row = malloc(m->raw_dim1 * sizeof(float));
        if (!temp_row) return;
        for (int col = 0; col < m->dim; col++) {
            const uint8_t *src_row = (const uint8_t*)m->raw_tensor_data + col * src_row_size;
            if (m->tensor_type == GGML_TYPE_F32) {
                float val;
                memcpy(&val, src_row + row * sizeof(float), sizeof(float));
                out[col] = val;
            } else if (m->tensor_type == GGML_TYPE_F16) {
                uint16_t val;
                memcpy(&val, src_row + row * sizeof(uint16_t), sizeof(uint16_t));
                out[col] = fp16_to_fp32(val);
            } else {
                memset(out, 0, sizeof(float) * m->dim);
                free(temp_row);
                return;
            }
        }
        free(temp_row);
        return;
    }

    raw = (const uint8_t*)m->raw_tensor_data + row * m->row_bytes;
    effective_cols = m->dim;

    switch (m->tensor_type) {
        case GGML_TYPE_F32:
            memcpy(out, raw, effective_cols * sizeof(float));
            break;
        case GGML_TYPE_F16:
            for (int j = 0; j < effective_cols; j++) {
                uint16_t h;
                memcpy(&h, raw + j * sizeof(uint16_t), sizeof(uint16_t));
                out[j] = fp16_to_fp32(h);
            }
            break;
        case GGML_TYPE_Q4_0: dequantize_row_q4_0(raw, out, effective_cols); break;
        case GGML_TYPE_Q4_1: dequantize_row_q4_1(raw, out, effective_cols); break;
        case GGML_TYPE_Q5_0: dequantize_row_q5_0(raw, out, effective_cols); break;
        case GGML_TYPE_Q5_1: dequantize_row_q5_1(raw, out, effective_cols); break;
        case GGML_TYPE_Q8_0: dequantize_row_q8_0(raw, out, effective_cols); break;
        case GGML_TYPE_Q8_1: dequantize_row_q8_1(raw, out, effective_cols); break;
        case GGML_TYPE_Q2_K: dequantize_row_q2_K(raw, out, effective_cols); break;
        case GGML_TYPE_Q3_K: dequantize_row_q3_K(raw, out, effective_cols); break;
        case GGML_TYPE_Q4_K: dequantize_row_q4_K(raw, out, effective_cols); break;
        case GGML_TYPE_Q5_K: dequantize_row_q5_K(raw, out, effective_cols); break;
        case GGML_TYPE_Q6_K: dequantize_row_q6_K(raw, out, effective_cols); break;
        case GGML_TYPE_Q8_K: dequantize_row_q8_K(raw, out, effective_cols); break;
        default:
            memset(out, 0, sizeof(float) * effective_cols);
    }

    for (int j = 0; j < effective_cols; j++) {
        if (isnan(out[j]) || isinf(out[j]) || fabsf(out[j]) > 1e10f) {
            out[j] = 0.0f;
        }
    }
}

static size_t get_row_bytes(int type, int n_cols) {
    switch (type) {
        case GGML_TYPE_F32: return n_cols * sizeof(float);
        case GGML_TYPE_F16: return n_cols * sizeof(uint16_t);
        case GGML_TYPE_Q4_0: return (n_cols / 32) * 18;
        case GGML_TYPE_Q4_1: return (n_cols / 32) * 20;
        case GGML_TYPE_Q5_0: return (n_cols / 32) * 22;
        case GGML_TYPE_Q5_1: return (n_cols / 32) * 24;
        case GGML_TYPE_Q8_0: return (n_cols / 32) * 34;
        case GGML_TYPE_Q8_1: return (n_cols / 32) * 40;
        case GGML_TYPE_Q2_K: return (n_cols / 256) * 84;
        case GGML_TYPE_Q3_K: return (n_cols / 256) * 110;
        case GGML_TYPE_Q4_K: return (n_cols / 256) * 144;
        case GGML_TYPE_Q5_K: return (n_cols / 256) * 176;
        case GGML_TYPE_Q6_K: return (n_cols / 256) * 210;
        case GGML_TYPE_Q8_K: return (n_cols / 256) * 292;
        default: return 0;
    }
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
            for (uint64_t i = 0; i < n && i < 1000000; i++)
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
    if (m->mapped) munmap(m->mapped, m->mapped_size);
    bpe_merge_table_free(&m->merges);
    free(m);
}

/* ------------------------------------------------------------------------- */
static int is_printable_string(const char *s, size_t len) {
    for (size_t i = 0; i < len; i++) if (!isprint((unsigned char)s[i])) return 0;
    return 1;
}

static uint8_t *find_tensor_info_start(uint8_t *cur, uint8_t *end) {
    uint8_t *scan = cur;
    while (scan + 8 < end) {
        uint64_t len;
        memcpy(&len, scan, 8);
        if (len > 0 && len < 256 && scan + 8 + len <= end && is_printable_string((char*)scan+8, len))
            return scan;
        scan++;
    }
    return NULL;
}

/* ------------------------------------------------------------------------- */
static void detect_space_marker(EmbedModel *m) {
    int marker_count[4] = {0};
    const char *markers[] = {"▁", "Ġ", "ĉ", " "};
    int marker_lens[] = {3, 2, 2, 1};

    for (int i = 0; i < m->vocab_size && i < 5000; i++) {
        for (int j = 0; j < 3; j++) {
            if (strncmp(m->tokens[i], markers[j], marker_lens[j]) == 0) {
                marker_count[j]++;
            }
        }
        if (m->tokens[i][0] == ' ' && strlen(m->tokens[i]) > 1) {
            marker_count[3]++;
        }
    }

    int best = 0;
    for (int i = 1; i < 4; i++) {
        if (marker_count[i] > marker_count[best]) best = i;
    }

    if (marker_count[best] > 10) {
        strcpy(m->space_marker, markers[best]);
        m->space_marker_len = marker_lens[best];
    }
}

static void parse_merge(const char *merge_str, char **left, char **right) {
    const char *space = strchr(merge_str, ' ');
    if (space) {
        int left_len = space - merge_str;
        *left = malloc(left_len + 1);
        memcpy(*left, merge_str, left_len);
        (*left)[left_len] = '\0';
        *right = strdup(space + 1);
    } else {
        *left = strdup(merge_str);
        *right = strdup("");
    }
}

/* ------------------------------------------------------------------------- */
static EmbedModel *embed_load_gguf(const char *path) {
    size_t sz;
    uint8_t *base = map_file(path, &sz);
    if (!base) return NULL;
    uint8_t *cur = base, *end = base + sz;
    if (sz < 4 || memcmp(cur, "GGUF", 4) != 0) { munmap(base, sz); return NULL; }
    cur += 4;
    uint32_t version = rd32(&cur, end);
    (void)version;
    uint64_t n_tensors = rd64(&cur, end);
    uint64_t n_kv = rd64(&cur, end);

    if (n_kv > 1000000 || n_tensors > 1000000) { munmap(base, sz); return NULL; }

    EmbedModel *m = calloc(1, sizeof(*m));
    if (!m) { munmap(base, sz); return NULL; }
    m->mapped = base;
    m->mapped_size = sz;
    m->table = calloc(HASH_SIZE, sizeof(HashNode*));
    if (!m->table) { free_model_contents(m); return NULL; }
    bpe_merge_table_init(&m->merges);
    m->unknown_token_id = -1;
    m->bos_token_id = -1;
    m->eos_token_id = -1;
    m->vocab_type = LLAMA_VOCAB_TYPE_NONE;
    m->normalize = NORM_NONE;

    int vocab_found = 0;
    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = rdstr(&cur, end);
        if (!key) { free_model_contents(m); return NULL; }
        uint32_t type = rd32(&cur, end);

        if ((strcmp(key, "tokenizer.ggml.tokens") == 0 || strcmp(key, "tokenizer.ggml.token_list") == 0) && type == 9) {
            uint32_t subtype = rd32(&cur, end);
            uint64_t n = rd64(&cur, end);
            if (subtype != 8 || n > 1000000) { free(key); free_model_contents(m); return NULL; }
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
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && type == 9) {
            uint32_t subtype = rd32(&cur, end);
            uint64_t n = rd64(&cur, end);
            if (subtype == 8) {
                for (uint64_t j = 0; j < n && j < MAX_MERGES; j++) {
                    char *merge_str = rdstr(&cur, end);
                    if (merge_str) {
                        char *left, *right;
                        parse_merge(merge_str, &left, &right);
                        bpe_merge_table_add(&m->merges, left, right, (int)j);
                        free(left);
                        free(right);
                        free(merge_str);
                    } else {
                        break;
                    }
                }
                if (n > MAX_MERGES) {
                    for (uint64_t j = MAX_MERGES; j < n; j++) {
                        char *merge_str = rdstr(&cur, end);
                        free(merge_str);
                    }
                }
            } else {
                if (!skip_value(&cur, end, type)) { free(key); free_model_contents(m); return NULL; }
            }
        } else if (strcmp(key, "tokenizer.ggml.model") == 0 && type == 8) {
            char *model_type = rdstr(&cur, end);
            if (model_type) {
                if (strcmp(model_type, "gpt2") == 0 || strcmp(model_type, "llama") == 0 ||
                    strcmp(model_type, "phi") == 0 || strcmp(model_type, "qwen") == 0)
                    m->vocab_type = LLAMA_VOCAB_TYPE_BPE;
                else if (strcmp(model_type, "bert") == 0)
                    m->vocab_type = LLAMA_VOCAB_TYPE_WPM;
                else if (strcmp(model_type, "spm") == 0)
                    m->vocab_type = LLAMA_VOCAB_TYPE_SPM;
                free(model_type);
            }
        } else if (strcmp(key, "tokenizer.ggml.pre") == 0 && type == 8) {
            char *pre = rdstr(&cur, end);
            free(pre);
        } else if (strcmp(key, "tokenizer.ggml.unknown_token_id") == 0 && type == 6) {
            m->unknown_token_id = (int)rd32(&cur, end);
        } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0 && type == 6) {
            m->bos_token_id = (int)rd32(&cur, end);
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0 && type == 6) {
            m->eos_token_id = (int)rd32(&cur, end);
        } else if (strcmp(key, "general.alignment") == 0 && type == 6) {
            rd32(&cur, end);
        } else {
            if (!skip_value(&cur, end, type)) { free(key); free_model_contents(m); return NULL; }
        }
        free(key);
    }

    if (!vocab_found) { free_model_contents(m); return NULL; }
    detect_space_marker(m);

    uint8_t *after_kv = cur;
    align_to_32(&cur, end, base);
    uint8_t *tensor_start = cur;
    int embd_found = 0;

    for (int attempt = 0; attempt < 2; attempt++) {
        cur = tensor_start;
        for (uint64_t i = 0; i < n_tensors; i++) {
            char *name = rdstr(&cur, end);
            if (!name) break;
            uint32_t n_dims = rd32(&cur, end);
            uint64_t dims[MAX_DIMS] = {0};
            for (uint32_t d = 0; d < n_dims && d < MAX_DIMS; d++) dims[d] = rd64(&cur, end);
            uint32_t type   = rd32(&cur, end);
            uint64_t offset = rd64(&cur, end);

            int is_token_embd = (strcmp(name, "token_embd.weight") == 0 ||
                                 strcmp(name, "embeddings.word_embeddings.weight") == 0 ||
                                 strcmp(name, "model.embed_tokens.weight") == 0);

            if (!is_token_embd && n_dims == 2 && m->vocab_size > 0) {
                if ((uint64_t)m->vocab_size == dims[0] && strstr(name, "embd")) is_token_embd = 1;
                else if ((uint64_t)m->vocab_size == dims[1] && strstr(name, "embd")) is_token_embd = 1;
            }

            if (!embd_found && is_token_embd) {
                if (n_dims < 2 || dims[1] == 0) { 
                    free(name); free_model_contents(m); return NULL; 
                }
                
                uint64_t ne0 = dims[0];
                uint64_t ne1 = dims[1];

                int need_transpose = 0;
                int dim;
                
                if (ne1 == (uint64_t)m->vocab_size) {
                    dim = (int)ne0;
                    need_transpose = 0;
                } else if (ne0 == (uint64_t)m->vocab_size) {
                    dim = (int)ne1;
                    need_transpose = 1;
                } else {
                    dim = (ne0 < ne1) ? (int)ne0 : (int)ne1;
                    need_transpose = (ne0 > ne1) ? 1 : 0;
                }

                if (dim <= 0 || dim > MAX_DIM) { 
                    free(name); free_model_contents(m); return NULL; 
                }

                size_t row_bytes = get_row_bytes(type, (int)(need_transpose ? ne1 : ne0));
                size_t total_size = (size_t)(need_transpose ? ne1 : ne0) * row_bytes;
                
                if (offset >= sz || offset + total_size > sz) {
                    free(name);
                    free_model_contents(m);
                    return NULL;
                }

                m->dim = dim;
                m->raw_dim0 = ne0;
                m->raw_dim1 = ne1;
                m->need_transpose = need_transpose;
                m->raw_tensor_data = base + offset;
                m->tensor_type = type;
                m->row_bytes = row_bytes;
                embd_found = 1;
                free(name);
                break;
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
        free_model_contents(m); return NULL; 
    }

    return m;
}

/* ------------------------------------------------------------------------- */
// L2 normalization
static void normalize_l2(float *vec, int dim) {
    float sum = 0;
    for (int i = 0; i < dim; i++) sum += vec[i] * vec[i];
    float norm = sqrtf(sum);
    if (norm > 1e-8f) {
        float inv = 1.0f / norm;
        for (int i = 0; i < dim; i++) vec[i] *= inv;
    }
}

/* ------------------------------------------------------------------------- */
static void embed_text(EmbedModel *m, const char *txt, float *out) {
    memset(out, 0, sizeof(float) * m->dim);
    if (!txt || !*txt) return;

    int num_words = 0;
    char **words = pre_tokenize(txt, &num_words);

    if (!words || num_words == 0) {
        if (words) free(words);
        return;
    }

    int *token_ids = malloc(m->vocab_size * sizeof(int));
    if (!token_ids) {
        for (int i = 0; i < num_words; i++) free(words[i]);
        free(words);
        return;
    }

    int used = 0;
    float *temp_vec = malloc(m->dim * sizeof(float));

    for (int i = 0; i < num_words; i++) {
        char *word = words[i];
        int id = hget(m, word);

        if (id == -1 && m->space_marker_len > 0) {
            size_t with_marker_len = m->space_marker_len + strlen(word);
            char *with_marker = malloc(with_marker_len + 1);
            if (with_marker) {
                memcpy(with_marker, m->space_marker, m->space_marker_len);
                strcpy(with_marker + m->space_marker_len, word);
                id = hget(m, with_marker);
                free(with_marker);
            }
        }

        if (id != -1 && id >= 0 && id < m->vocab_size) {
            dequantize_row_lazy(m, id, temp_vec);
            for (int j = 0; j < m->dim; j++) out[j] += temp_vec[j];
            used++;
        } else {
            int num_tokens = 0;
            bpe_tokenize_word(&m->merges, word, m, token_ids, &num_tokens);
            for (int k = 0; k < num_tokens; k++) {
                int tid = token_ids[k];
                if (tid >= 0 && tid < m->vocab_size) {
                    dequantize_row_lazy(m, tid, temp_vec);
                    for (int j = 0; j < m->dim; j++) out[j] += temp_vec[j];
                    used++;
                } else if (m->unknown_token_id != -1 && m->unknown_token_id < m->vocab_size) {
                    dequantize_row_lazy(m, m->unknown_token_id, temp_vec);
                    for (int j = 0; j < m->dim; j++) out[j] += temp_vec[j];
                    used++;
                }
            }
        }
        free(word);
    }

    free(words);
    free(token_ids);
    free(temp_vec);

    if (used > 0) {
        float inv = 1.0f / used;
        for (int i = 0; i < m->dim; i++) out[i] *= inv;
    }

    for (int i = 0; i < m->dim; i++) {
        if (isnan(out[i]) || isinf(out[i])) {
            out[i] = 0.0f;
        }
    }

    if (m->normalize == NORM_L2) {
        normalize_l2(out, m->dim);
    }
}

/* ------------------------------------------------------------------------- */
// Ruby bindings
static void rb_embedder_free(void *p) {
    ruby_embedder *e = p;
    if (e) { if (e->model) free_model_contents(e->model); free(e); }
}

static size_t rb_embedder_memsize(const void *p) {
    const ruby_embedder *e = p;
    size_t sz = sizeof(ruby_embedder);
    if (e && e->model) {
        sz += e->model->vocab_size * sizeof(char*);
        sz += e->model->mapped_size;
        sz += HASH_SIZE * sizeof(HashNode*);
    }
    return sz;
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

    Check_Type(opts, T_HASH);
    VALUE path = rb_hash_aref(opts, ID2SYM(rb_intern("model")));
    if (NIL_P(path)) rb_raise(rb_eArgError, "missing required key: model");
    const char *cpath = StringValueCStr(path);

    VALUE normalize = rb_hash_aref(opts, ID2SYM(rb_intern("normalize")));
    int norm_type = NORM_NONE;
    if (!NIL_P(normalize)) {
        if (SYMBOL_P(normalize)) {
            ID sym_id = SYM2ID(normalize);
            if (sym_id == rb_intern("l2") || sym_id == rb_intern("L2")) {
                norm_type = NORM_L2;
            }
        } else if (TYPE(normalize) == T_STRING) {
            const char *norm_str = StringValueCStr(normalize);
            if (strcasecmp(norm_str, "l2") == 0) {
                norm_type = NORM_L2;
            }
        }
    }

    e->model = embed_load_gguf(cpath);
    if (!e->model) rb_raise(rb_eRuntimeError, "failed to load GGUF model: %s", cpath);

    e->model->normalize = norm_type;
    return self;
}

static VALUE rb_embed(VALUE self, VALUE opts) {
    ruby_embedder *e;
    TypedData_Get_Struct(self, ruby_embedder, &ruby_embedder_type, e);

    Check_Type(opts, T_HASH);
    VALUE text = rb_hash_aref(opts, ID2SYM(rb_intern("text")));
    if (NIL_P(text)) rb_raise(rb_eArgError, "missing required key: text");
    const char *ctext = StringValueCStr(text);

    VALUE out = rb_str_new(NULL, e->model->dim * sizeof(float));
    embed_text(e->model, ctext, (float*)RSTRING_PTR(out));
    return out;
}

void Init_mini_embed(void) {
    VALUE c = rb_define_class("MiniEmbed", rb_cObject);
    rb_define_alloc_func(c, rb_embedder_alloc);
    rb_define_method(c, "initialize", rb_embedder_initialize, 1);
    rb_define_method(c, "embed", rb_embed, 1);
}