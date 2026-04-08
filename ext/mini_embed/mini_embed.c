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
#define MAX_MERGES 10000
#define MAX_REGEX 256

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
// Simple regex pattern matcher (simplified)
typedef struct {
    char *pattern;
    int pattern_len;
} RegexPattern;

static int match_regex(const char *text, const RegexPattern *patterns, int num_patterns) {
    for (int i = 0; i < num_patterns; i++) {
        const char *p = patterns[i].pattern;
        if (strstr(p, "\\p{L}")) {
            size_t len;
            uint32_t cp = unicode_cpt_from_utf8(text, &len);
            if (unicode_is_letter(cp)) return 1;
        } else if (strstr(p, "\\p{N}")) {
            size_t len;
            uint32_t cp = unicode_cpt_from_utf8(text, &len);
            if (unicode_is_number(cp)) return 1;
        } else if (p[0] == '\\' && p[1] == 's') {
            if (isspace(text[0])) return 1;
        } else if (p[0] == '\\' && p[1] == 'r') {
            if (text[0] == '\r') return 1;
        } else if (p[0] == '\\' && p[1] == 'n') {
            if (text[0] == '\n') return 1;
        } else if (p[0] == '.' && p[1] == '*') {
            return 1;
        } else if (isalnum(p[0]) || ispunct(p[0])) {
            if (text[0] == p[0]) return 1;
        }
    }
    return 0;
}

static char** unicode_regex_split(const char *text, const RegexPattern *patterns, int num_patterns, int *num_words) {
    char **words = NULL;
    int word_count = 0, word_capacity = 0;
    size_t text_len = strlen(text), pos = 0;

    while (pos < text_len) {
        size_t start = pos;
        while (start < text_len && !match_regex(text + start, patterns, num_patterns)) start++;
        if (start >= text_len) break;
        size_t end = start;
        while (end < text_len && match_regex(text + end, patterns, num_patterns)) end++;
        if (end > start) {
            size_t word_len = end - start;
            char *word = malloc(word_len + 1);
            if (!word) { while (--word_count >= 0) free(words[word_count]); free(words); *num_words = 0; return NULL; }
            memcpy(word, text + start, word_len);
            word[word_len] = '\0';
            if (word_count >= word_capacity) {
                word_capacity = word_capacity ? word_capacity * 2 : 16;
                words = realloc(words, word_capacity * sizeof(char*));
                if (!words) { free(word); while (--word_count >= 0) free(words[word_count]); *num_words = 0; return NULL; }
            }
            words[word_count++] = word;
        }
        pos = end;
    }
    *num_words = word_count;
    return words;
}

/* ------------------------------------------------------------------------- */
// BPE merge structures
typedef struct {
    char *left;
    char *right;
    char *merged;
    int rank;
} BPEMerge;

typedef struct {
    BPEMerge *merges;
    int num_merges;
    int capacity;
} BPEMergeTable;

static void bpe_merge_table_init(BPEMergeTable *table) {
    memset(table, 0, sizeof(*table));
}

static void bpe_merge_table_add(BPEMergeTable *table, const char *left, const char *right, const char *merged, int rank) {
    if (table->num_merges >= table->capacity) {
        table->capacity = table->capacity ? table->capacity * 2 : 100;
        table->merges = realloc(table->merges, table->capacity * sizeof(BPEMerge));
    }
    BPEMerge *m = &table->merges[table->num_merges++];
    m->left = strdup(left);
    m->right = strdup(right);
    m->merged = strdup(merged);
    m->rank = rank;
}

static void bpe_merge_table_free(BPEMergeTable *table) {
    for (int i = 0; i < table->num_merges; i++) {
        free(table->merges[i].left);
        free(table->merges[i].right);
        free(table->merges[i].merged);
    }
    free(table->merges);
    table->merges = NULL;
    table->num_merges = 0;
}

static int bpe_merge_rank(const BPEMergeTable *table, const char *left, const char *right) {
    for (int i = 0; i < table->num_merges; i++) {
        if (strcmp(table->merges[i].left, left) == 0 && strcmp(table->merges[i].right, right) == 0)
            return table->merges[i].rank;
    }
    return -1;
}

/* ------------------------------------------------------------------------- */
// BPE tokenization
typedef struct {
    char *text;
    int start, end;
    int prev, next;
    int used;
} BPESymbol;

static void bpe_tokenize_word(const BPEMergeTable *merges, const char *word, int (*text_to_id)(void*, const char*), void *vocab_data, int *token_ids, int *num_tokens) {
    int word_len = strlen(word);
    int num_symbols = 0;
    BPESymbol *symbols = malloc(word_len * sizeof(BPESymbol));
    int offset = 0;
    while (offset < word_len) {
        int char_len = unicode_len_utf8(word[offset]);
        symbols[num_symbols].text = (char*)word + offset;
        symbols[num_symbols].start = offset;
        symbols[num_symbols].end = offset + char_len;
        symbols[num_symbols].prev = num_symbols - 1;
        symbols[num_symbols].next = num_symbols + 1;
        symbols[num_symbols].used = 1;
        offset += char_len;
        num_symbols++;
    }

    if (num_symbols <= 1) {
        int id = text_to_id(vocab_data, word);
        if (id != -1) token_ids[(*num_tokens)++] = id;
        free(symbols);
        return;
    }

    typedef struct { int left, right, rank; } Bigram;
    Bigram *bigrams = malloc(num_symbols * num_symbols * sizeof(Bigram));
    int num_bigrams = 0;
    for (int i = 0; i < num_symbols - 1; i++) {
        if (symbols[i].used && symbols[i+1].used) {
            char *left_str = malloc(symbols[i].end - symbols[i].start + 1);
            char *right_str = malloc(symbols[i+1].end - symbols[i+1].start + 1);
            memcpy(left_str, symbols[i].text, symbols[i].end - symbols[i].start);
            memcpy(right_str, symbols[i+1].text, symbols[i+1].end - symbols[i+1].start);
            left_str[symbols[i].end - symbols[i].start] = '\0';
            right_str[symbols[i+1].end - symbols[i+1].start] = '\0';
            int rank = bpe_merge_rank(merges, left_str, right_str);
            if (rank != -1) {
                bigrams[num_bigrams].left = i;
                bigrams[num_bigrams].right = i+1;
                bigrams[num_bigrams].rank = rank;
                num_bigrams++;
            }
            free(left_str); free(right_str);
        }
    }
    for (int i = 0; i < num_bigrams - 1; i++)
        for (int j = i+1; j < num_bigrams; j++)
            if (bigrams[i].rank > bigrams[j].rank) {
                Bigram tmp = bigrams[i];
                bigrams[i] = bigrams[j];
                bigrams[j] = tmp;
            }

    int *merged = calloc(num_symbols, sizeof(int));
    for (int i = 0; i < num_bigrams; i++) {
        int left = bigrams[i].left, right = bigrams[i].right;
        if (merged[left] || merged[right]) continue;
        symbols[left].end = symbols[right].end;
        symbols[left].next = symbols[right].next;
        merged[right] = 1;
        if (symbols[right].next < num_symbols) symbols[symbols[right].next].prev = left;
    }

    for (int i = 0; i < num_symbols; i++) {
        if (!merged[i] && symbols[i].used) {
            char *substr = malloc(symbols[i].end - symbols[i].start + 1);
            memcpy(substr, word + symbols[i].start, symbols[i].end - symbols[i].start);
            substr[symbols[i].end - symbols[i].start] = '\0';
            int id = text_to_id(vocab_data, substr);
            if (id != -1) token_ids[(*num_tokens)++] = id;
            free(substr);
        }
    }
    free(bigrams); free(merged); free(symbols);
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
    float *float_data;
    void *tensor_data;
    int tensor_type;
    void *mapped;
    size_t mapped_size;
    HashNode **table;
    BPEMergeTable merges;
    RegexPattern *pre_patterns;
    int num_pre_patterns;
    int unknown_token_id;
    int bos_token_id;
    int eos_token_id;
    int vocab_type;
    char space_marker[8];
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
    void *data = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    return data == MAP_FAILED ? NULL : data;
}

/* ------------------------------------------------------------------------- */
// FP16 conversion
static float fp16_to_fp32(uint16_t h) {
    uint16_t sign = (h >> 15) & 1;
    uint16_t exp  = (h >> 10) & 0x1F;
    uint16_t mant = h & 0x3FF;
    if (exp == 0) return (mant / 1024.0f) * 6.103515625e-5f * (sign ? -1.0f : 1.0f);
    if (exp == 31) return 0.0f;
    return (1.0f + mant / 1024.0f) * (1 << (exp - 15)) * (sign ? -1.0f : 1.0f);
}

/* ------------------------------------------------------------------------- */
// Block dequantization functions
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

    // Sanitize the tensor: replace NaNs, Infs, and astronomically large values with zero
    int total = n_rows * n_cols;
    for (int i = 0; i < total; i++) {
        if (isnan(out[i]) || isinf(out[i]) || fabs(out[i]) > 1e10f) {
            out[i] = 0.0f;
        }
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
    bpe_merge_table_free(&m->merges);
    if (m->pre_patterns) {
        for (int i = 0; i < m->num_pre_patterns; i++) free(m->pre_patterns[i].pattern);
        free(m->pre_patterns);
    }
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
    const char *candidates[] = {"▁", "Ġ", " "};
    for (int i = 0; i < 3; i++) {
        const char *marker = candidates[i];
        int marker_len = strlen(marker);
        for (int j = 0; j < m->vocab_size; j++) {
            if (strncmp(m->tokens[j], marker, marker_len) == 0) {
                strcpy(m->space_marker, marker);
                return;
            }
        }
    }
    m->space_marker[0] = '\0';
}

static void setup_default_pre_patterns(EmbedModel *m) {
    const char *default_patterns[] = {
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
        "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+",
        "\\p{N}{1,3}",
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*",
        "\\s*[\\r\\n]+",
        "\\s+(?!\\S)",
        "\\s+"
    };
    m->num_pre_patterns = sizeof(default_patterns)/sizeof(default_patterns[0]);
    m->pre_patterns = malloc(m->num_pre_patterns * sizeof(RegexPattern));
    for (int i = 0; i < m->num_pre_patterns; i++) {
        m->pre_patterns[i].pattern = strdup(default_patterns[i]);
        m->pre_patterns[i].pattern_len = strlen(default_patterns[i]);
    }
}

static void parse_merge(const char *merge_str, char **left, char **right) {
    const char *space = strchr(merge_str, ' ');
    if (space) {
        int left_len = space - merge_str;
        *left = malloc(left_len+1);
        memcpy(*left, merge_str, left_len);
        (*left)[left_len] = '\0';
        *right = strdup(space+1);
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
    if (memcmp(cur, "GGUF", 4) != 0) { munmap(base, sz); return NULL; }
    cur += 4;
    uint32_t version = rd32(&cur, end);
    uint64_t n_tensors = rd64(&cur, end);
    uint64_t n_kv = rd64(&cur, end);

    EmbedModel *m = calloc(1, sizeof(*m));
    if (!m) { munmap(base, sz); return NULL; }
    m->mapped = base;
    m->mapped_size = sz;
    m->table = calloc(HASH_SIZE, sizeof(HashNode*));
    if (!m->table) { free_model_contents(m); return NULL; }
    bpe_merge_table_init(&m->merges);
    setup_default_pre_patterns(m);
    m->unknown_token_id = -1;
    m->bos_token_id = -1;
    m->eos_token_id = -1;
    m->vocab_type = LLAMA_VOCAB_TYPE_NONE;
    m->space_marker[0] = '\0';

    int vocab_found = 0;
    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = rdstr(&cur, end);
        if (!key) { free_model_contents(m); return NULL; }
        uint32_t type = rd32(&cur, end);
        if ((strcmp(key, "tokenizer.ggml.tokens") == 0 || strcmp(key, "tokenizer.ggml.token_list") == 0) && type == 9) {
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
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && type == 9) {
            uint32_t subtype = rd32(&cur, end);
            uint64_t n = rd64(&cur, end);
            if (subtype == 8) {
                for (uint64_t j = 0; j < n && j < MAX_MERGES; j++) {
                    char *merge_str = rdstr(&cur, end);
                    if (merge_str) {
                        char *left, *right;
                        parse_merge(merge_str, &left, &right);
                        bpe_merge_table_add(&m->merges, left, right, merge_str, (int)j);
                        free(left); free(right);
                        free(merge_str);
                    }
                }
            } else {
                if (!skip_value(&cur, end, type)) { free(key); free_model_contents(m); return NULL; }
            }
        } else if (strcmp(key, "tokenizer.ggml.model") == 0 && type == 8) {
            char *model_type = rdstr(&cur, end);
            if (model_type) {
                if (strcmp(model_type, "gpt2") == 0 || strcmp(model_type, "llama") == 0) m->vocab_type = LLAMA_VOCAB_TYPE_BPE;
                else if (strcmp(model_type, "bert") == 0) m->vocab_type = LLAMA_VOCAB_TYPE_WPM;
                free(model_type);
            }
        } else if (strcmp(key, "tokenizer.ggml.pre") == 0 && type == 8) {
            char *pre = rdstr(&cur, end);
            if (pre) free(pre);
        } else if (strcmp(key, "tokenizer.ggml.unknown_token_id") == 0 && type == 6) {
            m->unknown_token_id = rd32(&cur, end);
        } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0 && type == 6) {
            m->bos_token_id = rd32(&cur, end);
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0 && type == 6) {
            m->eos_token_id = rd32(&cur, end);
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
                if (n_dims < 2 || dims[1] == 0) { free(name); free_model_contents(m); return NULL; }
                dim0 = dims[0]; dim1 = dims[1];
                if (dim0 == (uint64_t)m->vocab_size) { m->dim = (int)dim1; need_transpose = 0; }
                else if (dim1 == (uint64_t)m->vocab_size) { m->dim = (int)dim0; need_transpose = 1; }
                else { m->dim = (dim0 < dim1) ? (int)dim0 : (int)dim1; need_transpose = (dim0 > dim1) ? 1 : 0; }
                raw_tensor_data = base + offset;
                tensor_type = type;
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
    if (!embd_found || m->dim == 0) { free_model_contents(m); return NULL; }

    if (tensor_type == GGML_TYPE_F32 && !need_transpose) {
        m->float_data = NULL;
        m->tensor_data = raw_tensor_data;
    } else {
        int n_rows = need_transpose ? (int)dim1 : (int)dim0;
        int n_cols = need_transpose ? (int)dim0 : (int)dim1;
        m->float_data = dequantize_tensor(raw_tensor_data, tensor_type, n_rows, n_cols);
        if (!m->float_data) { free_model_contents(m); return NULL; }
        m->tensor_data = m->float_data;
    }
    m->tensor_type = tensor_type;

    return m;
}

/* ------------------------------------------------------------------------- */
static void embed_text(EmbedModel *m, const char *txt, float *out) {
    memset(out, 0, sizeof(float) * m->dim);
    int num_words = 0;
    char **words = unicode_regex_split(txt, m->pre_patterns, m->num_pre_patterns, &num_words);
    if (!words || num_words == 0) {
        // Fallback to simple space split
        char *copy = strdup(txt);
        if (copy) {
            char *tok = strtok(copy, " \t\n\r");
            int used = 0;
            const float *embd = (float*)m->tensor_data;
            while (tok) {
                int id = hget(m, tok);
                if (id >= 0 && id < m->vocab_size) {
                    const float *vec = embd + id * m->dim;
                    for (int i = 0; i < m->dim; i++) out[i] += vec[i];
                    used++;
                }
                tok = strtok(NULL, " \t\n\r");
            }
            if (used) { float inv = 1.0f / used; for (int i = 0; i < m->dim; i++) out[i] *= inv; }
            free(copy);
        }
        if (words) free(words);
        return;
    }

    int *token_ids = malloc(m->vocab_size * sizeof(int));
    int used = 0;
    const float *embd = (float*)m->tensor_data;
    for (int i = 0; i < num_words; i++) {
        char *word = words[i];
        int id = hget(m, word);
        if (id == -1 && m->space_marker[0]) {
            char *with_marker = malloc(strlen(m->space_marker) + strlen(word) + 1);
            strcpy(with_marker, m->space_marker);
            strcat(with_marker, word);
            id = hget(m, with_marker);
            free(with_marker);
        }
        if (id != -1) {
            const float *vec = embd + id * m->dim;
            for (int j = 0; j < m->dim; j++) out[j] += vec[j];
            used++;
        } else {
            int num_tokens = 0;
            bpe_tokenize_word(&m->merges, word, text_to_id, m, token_ids, &num_tokens);
            for (int k = 0; k < num_tokens; k++) {
                int tid = token_ids[k];
                if (tid >= 0 && tid < m->vocab_size) {
                    const float *vec = embd + tid * m->dim;
                    for (int j = 0; j < m->dim; j++) out[j] += vec[j];
                    used++;
                } else if (m->unknown_token_id != -1 && m->unknown_token_id < m->vocab_size) {
                    const float *vec = embd + m->unknown_token_id * m->dim;
                    for (int j = 0; j < m->dim; j++) out[j] += vec[j];
                    used++;
                }
            }
        }
        free(word);
    }
    free(words);
    free(token_ids);
    if (used > 0) {
        float inv = 1.0f / used;
        for (int i = 0; i < m->dim; i++) out[i] *= inv;
    }
    for (int i = 0; i < m->dim; i++) {
        if (isnan(out[i]) || isinf(out[i])) {
            out[i] = 0.0f;
        }
    }
}

/* ------------------------------------------------------------------------- */
// Ruby bindings
static void rb_embedder_free(void *p) {
    ruby_embedder *e = p;
    if (e) { if (e->model) free_model_contents(e->model); free(e); }
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
    if (!e->model) rb_raise(rb_eRuntimeError, "failed to load GGUF model");
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
    rb_define_method(c, "embed", rb_embed, 1);
}