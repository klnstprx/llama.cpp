
// NOTE: This is modified from clip.cpp only for LLaVA,
// so there might be still unnecessary artifacts hanging around
// I'll gradually clean and extend it
// Note: Even when using identical normalized image inputs (see normalize_image_u8_to_f32()) we have a significant difference in resulting embeddings compared to pytorch
#include "clip.h"

#include "clip-impl.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#define STB_IMAGE_IMPLEMENTATION
#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "stb_image.h"

// Helper to ensure a tensor is in the canonical shape [rows, cols]; if not, transpose and copy
static ggml_tensor * assign_canonical_or_transposed_tensor(
    ggml_context * ctx_target_metadata,  // This is ctx_clip.ctx_data.get()
    ggml_tensor * tensor_from_gguf, int64_t n_rows, int64_t n_cols) {
    if (!tensor_from_gguf) {
        return nullptr;
    }
    if (tensor_from_gguf->ne[0] == n_rows && tensor_from_gguf->ne[1] == n_cols) {
        // Already canonical, just return it (it's a view into ctx_meta)
        // We need to duplicate its metadata into ctx_target_metadata
        // No, get_tensor() already does ggml_dup_tensor into ctx_target_metadata.
        // So, if shape matches, tensor_from_gguf IS ALREADY in ctx_target_metadata.
        fprintf(stderr, "[ASSIGN_CANONICAL_DEBUG] Tensor '%s' already canonical. ne: [%lld, %lld]\n",
                tensor_from_gguf->name, (long long) tensor_from_gguf->ne[0], (long long) tensor_from_gguf->ne[1]);
        return tensor_from_gguf;
    }
    if (tensor_from_gguf->ne[0] == n_cols && tensor_from_gguf->ne[1] == n_rows) {
        // Needs transpose and copy
        fprintf(
            stderr,
            "[ASSIGN_CANONICAL_DEBUG] Tensor '%s' needs transpose. current ne: [%lld, %lld], target ne: [%lld, %lld]\n",
            tensor_from_gguf->name, (long long) tensor_from_gguf->ne[0], (long long) tensor_from_gguf->ne[1],
            (long long) n_rows, (long long) n_cols);

        const size_t buf_size_pad = ggml_nbytes_pad(tensor_from_gguf);  // Data for the transposed tensor

        const size_t unpadded_ggml_tensor_struct_size_assign = sizeof(struct ggml_tensor);
        const size_t size_of_ggml_object_struct_assign =
            ggml_tensor_overhead() - unpadded_ggml_tensor_struct_size_assign;
        const size_t padded_ggml_tensor_struct_size_assign =
            GGML_PAD(unpadded_ggml_tensor_struct_size_assign, GGML_MEM_ALIGN);
        const size_t actual_metadata_footprint_for_one_tensor_assign =  // Renamed for clarity
            size_of_ggml_object_struct_assign + padded_ggml_tensor_struct_size_assign;

        struct ggml_init_params params_tmp_transpose = {
            // Renamed
            /*.mem_size*/ buf_size_pad +
                actual_metadata_footprint_for_one_tensor_assign,  // Space for 1 tensor's metadata + its data
            /*.mem_buffer*/ nullptr,
            /*.no_alloc*/ false,  // Data WILL be allocated in tmp_ctx for the transpose operation
        };
        std::unique_ptr<ggml_context, decltype(&ggml_free)> tmp_ctx(ggml_init(params_tmp_transpose), &ggml_free);
        if (!tmp_ctx) {
            fprintf(stderr, "[ASSIGN_CANONICAL_ERROR] Failed to create tmp_ctx for transpose of '%s'\n",
                    tensor_from_gguf->name);
            return nullptr;  // Or throw
        }

        // ggml_dup_tensor into tmp_ctx to get a modifiable copy if tensor_from_gguf is from ctx_meta (read-only data)
        // However, ggml_transpose will create a new tensor for its output anyway.
        // What's important is that tensor_from_gguf->data is valid.
        ggml_tensor * transposed_in_tmp_ctx = ggml_transpose(tmp_ctx.get(), tensor_from_gguf);
        if (!transposed_in_tmp_ctx) {
            fprintf(stderr, "[ASSIGN_CANONICAL_ERROR] ggml_transpose failed in tmp_ctx for '%s'\n",
                    tensor_from_gguf->name);
            return nullptr;
        }
        ggml_set_name(transposed_in_tmp_ctx, (std::string(tensor_from_gguf->name) + " (transposed_tmp)").c_str());
        fprintf(stderr, "[ASSIGN_CANONICAL_DEBUG] Transposed '%s' in tmp_ctx. New name: '%s', ne: [%lld, %lld]\n",
                tensor_from_gguf->name, transposed_in_tmp_ctx->name, (long long) transposed_in_tmp_ctx->ne[0],
                (long long) transposed_in_tmp_ctx->ne[1]);

        // Now, copy this transposed tensor (both metadata and data) into the target context
        ggml_tensor * contig_in_target_ctx = ggml_cont(ctx_target_metadata, transposed_in_tmp_ctx);
        if (!contig_in_target_ctx) {
            fprintf(stderr, "[ASSIGN_CANONICAL_ERROR] ggml_cont failed for '%s' into target ctx %p\n",
                    tensor_from_gguf->name, (void *) ctx_target_metadata);
            return nullptr;
        }
        ggml_set_name(contig_in_target_ctx, (std::string(tensor_from_gguf->name) + " (transposed_cont)").c_str());
        fprintf(stderr,
                "[ASSIGN_CANONICAL_DEBUG] ggml_cont for '%s' into target ctx %p. Result tensor name: '%s', ne: [%lld, "
                "%lld]\n",
                tensor_from_gguf->name, (void *) ctx_target_metadata, contig_in_target_ctx->name,
                (long long) contig_in_target_ctx->ne[0], (long long) contig_in_target_ctx->ne[1]);

        return contig_in_target_ctx;
    }
    fprintf(stderr,
            "[ASSIGN_CANONICAL_ERROR] Tensor '%s' has unexpected shape for canonicalization. current ne: [%lld, %lld], "
            "target: [%lld, %lld]\n",
            tensor_from_gguf->name, (long long) tensor_from_gguf->ne[0], (long long) tensor_from_gguf->ne[1],
            (long long) n_rows, (long long) n_cols);
    GGML_ASSERT(false && "assign_canonical_or_transposed_tensor: Unexpected shape.");
    // return nullptr;
}

struct clip_logger_state g_logger_state = { GGML_LOG_LEVEL_CONT, clip_log_callback_default, NULL };

enum ffn_op_type {
    FFN_GELU,
    FFN_SILU,
    FFN_GELU_QUICK,
};

enum norm_type {
    NORM_TYPE_NORMAL,
    NORM_TYPE_RMS,
};

//#define CLIP_DEBUG_FUNCTIONS

#ifdef CLIP_DEBUG_FUNCTIONS
static void clip_image_write_image_to_ppm(const clip_image_u8 & img, const std::string & filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    // PPM header: P6 format, width, height, and max color value
    file << "P6\n" << img.nx << " " << img.ny << "\n255\n";

    // Write pixel data
    for (size_t i = 0; i < img.buf.size(); i += 3) {
        // PPM expects binary data in RGB format, which matches our image buffer
        file.write(reinterpret_cast<const char *>(&img.buf[i]), 3);
    }

    file.close();
}

static void clip_image_save_to_bmp(const clip_image_u8 & img, const std::string & filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    int fileSize      = 54 + 3 * img.nx * img.ny;  // File header + info header + pixel data
    int bytesPerPixel = 3;
    int widthInBytes  = img.nx * bytesPerPixel;
    int paddingAmount = (4 - (widthInBytes % 4)) % 4;
    int stride        = widthInBytes + paddingAmount;

    // Bitmap file header
    unsigned char fileHeader[14] = {
        'B', 'M',        // Signature
        0,   0,   0, 0,  // Image file size in bytes
        0,   0,   0, 0,  // Reserved
        54,  0,   0, 0   // Start of pixel array
    };

    // Total file size
    fileSize      = 54 + (stride * img.ny);
    fileHeader[2] = (unsigned char) (fileSize);
    fileHeader[3] = (unsigned char) (fileSize >> 8);
    fileHeader[4] = (unsigned char) (fileSize >> 16);
    fileHeader[5] = (unsigned char) (fileSize >> 24);

    // Bitmap information header (BITMAPINFOHEADER)
    unsigned char infoHeader[40] = {
        40, 0, 0, 0,  // Size of this header (40 bytes)
        0,  0, 0, 0,  // Image width
        0,  0, 0, 0,  // Image height
        1,  0,        // Number of color planes
        24, 0,        // Bits per pixel
        0,  0, 0, 0,  // No compression
        0,  0, 0, 0,  // Image size (can be 0 for no compression)
        0,  0, 0, 0,  // X pixels per meter (not specified)
        0,  0, 0, 0,  // Y pixels per meter (not specified)
        0,  0, 0, 0,  // Total colors (color table not used)
        0,  0, 0, 0   // Important colors (all are important)
    };

    // Width and height in the information header
    infoHeader[4]  = (unsigned char) (img.nx);
    infoHeader[5]  = (unsigned char) (img.nx >> 8);
    infoHeader[6]  = (unsigned char) (img.nx >> 16);
    infoHeader[7]  = (unsigned char) (img.nx >> 24);
    infoHeader[8]  = (unsigned char) (img.ny);
    infoHeader[9]  = (unsigned char) (img.ny >> 8);
    infoHeader[10] = (unsigned char) (img.ny >> 16);
    infoHeader[11] = (unsigned char) (img.ny >> 24);

    // Write file headers
    file.write(reinterpret_cast<char *>(fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<char *>(infoHeader), sizeof(infoHeader));

    // Pixel data
    std::vector<unsigned char> padding(3, 0);  // Max padding size to be added to each row
    for (int y = img.ny - 1; y >= 0; --y) {    // BMP files are stored bottom-to-top
        for (int x = 0; x < img.nx; ++x) {
            // Each pixel
            size_t        pixelIndex = (y * img.nx + x) * 3;
            unsigned char pixel[3]   = { img.buf[pixelIndex + 2],  // BMP stores pixels in BGR format
                                         img.buf[pixelIndex + 1], img.buf[pixelIndex] };
            file.write(reinterpret_cast<char *>(pixel), 3);
        }
        // Write padding for the row
        file.write(reinterpret_cast<char *>(padding.data()), paddingAmount);
    }

    file.close();
}

// debug function to convert f32 to u8
static void clip_image_convert_f32_to_u8(const clip_image_f32 & src, clip_image_u8 & dst) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(3 * src.nx * src.ny);
    for (size_t i = 0; i < src.buf.size(); ++i) {
        dst.buf[i] = static_cast<uint8_t>(std::min(std::max(int(src.buf[i] * 255.0f), 0), 255));
    }
}
#endif

//
// clip layers
//

enum patch_merge_type {
    PATCH_MERGE_FLAT,
    PATCH_MERGE_SPATIAL_UNPAD,
};

struct clip_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t n_embd;
    int32_t n_ff;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer           = 0;
    int32_t proj_scale_factor = 0;  // idefics3

    // for models using dynamic image size, we need to have a smaller image size to warmup
    // otherwise, user will get OOM everytime they load the model
    int32_t warmup_image_size = 0;

    ffn_op_type ffn_op = FFN_GELU;

    patch_merge_type mm_patch_merge_type = PATCH_MERGE_FLAT;

    float eps        = 1e-6;
    float rope_theta = 0.0;

    std::vector<int32_t>        image_grid_pinpoints;
    int32_t                     image_crop_resolution;
    std::unordered_set<int32_t> vision_feature_layer;
    int32_t                     attn_window_size   = 0;
    int32_t                     n_wa_pattern       = 0;
    int32_t                     spatial_merge_size = 0;

    clip_hparams() :
        image_size(0),
        patch_size(0),
        n_embd(0),
        n_ff(0),
        projection_dim(0),
        n_head(0),
        n_layer(-99), /* sentinel */
        proj_scale_factor(0),
        warmup_image_size(0),
        ffn_op(FFN_GELU),
        mm_patch_merge_type(PATCH_MERGE_FLAT),
        eps(1e-6f),
        rope_theta(0.0f),
        image_crop_resolution(0),
        attn_window_size(0),
        n_wa_pattern(0),
        spatial_merge_size(0) {}
};

struct clip_layer {
    // attention
    ggml_tensor * k_w = nullptr;
    ggml_tensor * k_b = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;

    ggml_tensor * o_w = nullptr;
    ggml_tensor * o_b = nullptr;

    ggml_tensor * k_norm = nullptr;
    ggml_tensor * q_norm = nullptr;

    // layernorm 1
    ggml_tensor * ln_1_w = nullptr;
    ggml_tensor * ln_1_b = nullptr;

    ggml_tensor * ff_up_w   = nullptr;
    ggml_tensor * ff_up_b   = nullptr;
    ggml_tensor * ff_gate_w = nullptr;
    ggml_tensor * ff_gate_b = nullptr;
    ggml_tensor * ff_down_w = nullptr;
    ggml_tensor * ff_down_b = nullptr;

    // layernorm 2
    ggml_tensor * ln_2_w = nullptr;
    ggml_tensor * ln_2_b = nullptr;

    // layer scale (no bias)
    ggml_tensor * ls_1_w = nullptr;
    ggml_tensor * ls_2_w = nullptr;
};

// ------------------- Text transformer layer -------------------
struct clip_text_layer {
    // attention weights
    ggml_tensor * k_w = nullptr;
    ggml_tensor * k_b = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;

    ggml_tensor * o_w = nullptr;
    ggml_tensor * o_b = nullptr;

    // layer norm 1
    ggml_tensor * ln_1_w = nullptr;
    ggml_tensor * ln_1_b = nullptr;

    // ffn weights
    ggml_tensor * ff_up_w   = nullptr;
    ggml_tensor * ff_up_b   = nullptr;
    ggml_tensor * ff_gate_w = nullptr;
    ggml_tensor * ff_gate_b = nullptr;
    ggml_tensor * ff_down_w = nullptr;
    ggml_tensor * ff_down_b = nullptr;

    // layer norm 2
    ggml_tensor * ln_2_w = nullptr;
    ggml_tensor * ln_2_b = nullptr;
};

struct clip_vision_model {
    struct clip_hparams hparams;

    // embeddings
    ggml_tensor * class_embedding    = nullptr;
    ggml_tensor * patch_embeddings_0 = nullptr;
    ggml_tensor * patch_embeddings_1 =
        nullptr;  // second Conv2D kernel when we decouple Conv3D along temproal dimension (Qwen2VL)
    ggml_tensor * patch_bias          = nullptr;
    ggml_tensor * position_embeddings = nullptr;

    ggml_tensor * pre_ln_w = nullptr;
    ggml_tensor * pre_ln_b = nullptr;

    std::vector<clip_layer> layers;

    ggml_tensor * post_ln_w  = nullptr;
    ggml_tensor * post_ln_b  = nullptr;
    // optional projection head (for vanilla CLIP/SigLIP)
    ggml_tensor * projection = nullptr;

    // LLaVA projection
    ggml_tensor * mm_input_norm_w = nullptr;
    ggml_tensor * mm_0_w          = nullptr;
    ggml_tensor * mm_0_b          = nullptr;
    ggml_tensor * mm_2_w          = nullptr;
    ggml_tensor * mm_2_b          = nullptr;

    ggml_tensor * image_newline = nullptr;

    // Yi type models with mlp+normalization projection
    ggml_tensor * mm_1_w = nullptr;  // Yi type models have 0, 1, 3, 4
    ggml_tensor * mm_1_b = nullptr;
    ggml_tensor * mm_3_w = nullptr;
    ggml_tensor * mm_3_b = nullptr;
    ggml_tensor * mm_4_w = nullptr;
    ggml_tensor * mm_4_b = nullptr;

    // GLMV-Edge projection
    ggml_tensor * mm_model_adapter_conv_w = nullptr;
    ggml_tensor * mm_model_adapter_conv_b = nullptr;
    ggml_tensor * mm_glm_tok_boi          = nullptr;
    ggml_tensor * mm_glm_tok_eoi          = nullptr;

    // MobileVLM projection
    ggml_tensor * mm_model_mlp_1_w               = nullptr;
    ggml_tensor * mm_model_mlp_1_b               = nullptr;
    ggml_tensor * mm_model_mlp_3_w               = nullptr;
    ggml_tensor * mm_model_mlp_3_b               = nullptr;
    ggml_tensor * mm_model_block_1_block_0_0_w   = nullptr;
    ggml_tensor * mm_model_block_1_block_0_1_w   = nullptr;
    ggml_tensor * mm_model_block_1_block_0_1_b   = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc1_w = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc1_b = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc2_w = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc2_b = nullptr;
    ggml_tensor * mm_model_block_1_block_2_0_w   = nullptr;
    ggml_tensor * mm_model_block_1_block_2_1_w   = nullptr;
    ggml_tensor * mm_model_block_1_block_2_1_b   = nullptr;
    ggml_tensor * mm_model_block_2_block_0_0_w   = nullptr;
    ggml_tensor * mm_model_block_2_block_0_1_w   = nullptr;
    ggml_tensor * mm_model_block_2_block_0_1_b   = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc1_w = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc1_b = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc2_w = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc2_b = nullptr;
    ggml_tensor * mm_model_block_2_block_2_0_w   = nullptr;
    ggml_tensor * mm_model_block_2_block_2_1_w   = nullptr;
    ggml_tensor * mm_model_block_2_block_2_1_b   = nullptr;

    // MobileVLM_V2 projection
    ggml_tensor * mm_model_mlp_0_w = nullptr;
    ggml_tensor * mm_model_mlp_0_b = nullptr;
    ggml_tensor * mm_model_mlp_2_w = nullptr;
    ggml_tensor * mm_model_mlp_2_b = nullptr;
    ggml_tensor * mm_model_peg_0_w = nullptr;
    ggml_tensor * mm_model_peg_0_b = nullptr;

    // MINICPMV projection
    ggml_tensor * mm_model_pos_embed_k = nullptr;
    ggml_tensor * mm_model_query       = nullptr;
    ggml_tensor * mm_model_proj        = nullptr;
    ggml_tensor * mm_model_kv_proj     = nullptr;
    ggml_tensor * mm_model_attn_q_w    = nullptr;
    ggml_tensor * mm_model_attn_q_b    = nullptr;
    ggml_tensor * mm_model_attn_k_w    = nullptr;
    ggml_tensor * mm_model_attn_k_b    = nullptr;
    ggml_tensor * mm_model_attn_v_w    = nullptr;
    ggml_tensor * mm_model_attn_v_b    = nullptr;
    ggml_tensor * mm_model_attn_o_w    = nullptr;
    ggml_tensor * mm_model_attn_o_b    = nullptr;
    ggml_tensor * mm_model_ln_q_w      = nullptr;
    ggml_tensor * mm_model_ln_q_b      = nullptr;
    ggml_tensor * mm_model_ln_kv_w     = nullptr;
    ggml_tensor * mm_model_ln_kv_b     = nullptr;
    ggml_tensor * mm_model_ln_post_w   = nullptr;
    ggml_tensor * mm_model_ln_post_b   = nullptr;

    // gemma3
    ggml_tensor * mm_input_proj_w    = nullptr;
    ggml_tensor * mm_soft_emb_norm_w = nullptr;

    // pixtral
    ggml_tensor * token_embd_img_break = nullptr;
    ggml_tensor * mm_patch_merger_w    = nullptr;
};

struct clip_text_model {
    clip_hparams hparams;

    ggml_tensor * token_embeddings    = nullptr;
    ggml_tensor * position_embeddings = nullptr;

    std::vector<clip_text_layer> layers;

    ggml_tensor * post_ln_w = nullptr;
    ggml_tensor * post_ln_b = nullptr;

    ggml_tensor * projection = nullptr;

    int32_t n_embd     = 0;
    int32_t n_embd_out = 0;
};

struct clip_ctx {
    bool has_llava_projector = false;
    int  minicpmv_version    = 0;

    struct clip_vision_model vision_model;

    struct clip_text_model text_model;

    projector_type proj_type = PROJECTOR_TYPE_MLP;

    float image_mean[3];
    float image_std[3];

  public:
    gguf_context_ptr ctx_gguf;
    ggml_context_ptr ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    ggml_backend_t          backend;
    ggml_backend_t          backend_cpu;
    ggml_backend_buffer_ptr buf;

    int                    max_nodes = 8192;
    ggml_backend_sched_ptr sched;

    clip_image_size  load_image_size;
    // persistent context and graph for text transformer
    ggml_context_ptr text_ctx0_ptr;
    ggml_cgraph *    text_gf                    = nullptr;
    int              cached_text_graph_n_tokens = -1;  // Add this

    clip_ctx(clip_context_params & ctx_params) {
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backend = ctx_params.use_gpu ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr) : nullptr;

        if (backend) {
            LOG_INF("%s: CLIP using %s backend\n", __func__, ggml_backend_name(backend));
            backend_ptrs.push_back(backend);
            backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
        } else {
            backend = backend_cpu;
            LOG_INF("%s: CLIP using CPU backend\n", __func__);
        }

        backend_ptrs.push_back(backend_cpu);
        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

        sched.reset(
            ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), 8192, false, true));
    }

    ~clip_ctx() {
        ggml_backend_free(backend);
        if (backend != backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }
};

struct clip_graph {
    clip_ctx *                ctx;
    const clip_vision_model & model;
    const clip_hparams &      hparams;

    // we only support single image per batch
    const clip_image_f32 & img;

    const int   patch_size;
    const int   n_patches_x;
    const int   n_patches_y;
    const int   n_patches;
    const int   n_embd;
    const int   n_head;
    const int   d_head;
    const int   n_layer;
    const float eps;
    const float kq_scale;

    ggml_context_ptr ctx0_ptr;
    ggml_context *   ctx0;
    ggml_cgraph *    gf;

    // static void log_add_details(const char * message_prefix, const char * operation_detail, int layer_idx,
    //                             ggml_tensor * a, ggml_tensor * b) {
    //     fprintf(stderr,
    //             "[VISION_GRAPH_ADD_DEBUG] %s - %s (Layer/Ctx %d):\n"
    //             "  Operand A (e.g., cur): name='%s', type=%s(%d), ne=[%lld,%lld,%lld,%lld], nb=[%zu,%zu,%zu,%zu], "
    //             "data=%p, op=%s\n"
    //             "  Operand B (e.g., bias/residual): name='%s', type=%s(%d), ne=[%lld,%lld,%lld,%lld], "
    //             "nb=[%zu,%zu,%zu,%zu], data=%p, op=%s\n",
    //             message_prefix, operation_detail, layer_idx, a->name, ggml_type_name(a->type), a->type,
    //             (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3], a->nb[0],
    //             a->nb[1], a->nb[2], a->nb[3], a->data, ggml_op_name(a->op), b->name, ggml_type_name(b->type), b->type,
    //             (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3], b->nb[0],
    //             b->nb[1], b->nb[2], b->nb[3], b->data, ggml_op_name(b->op));
    // }

    clip_graph(clip_ctx * ctx, const clip_image_f32 & img) :
        ctx(ctx),
        model(ctx->vision_model),
        hparams(model.hparams),
        img(img),
        patch_size(hparams.patch_size),
        n_patches_x(img.nx / patch_size),
        n_patches_y(img.ny / patch_size),
        n_patches(n_patches_x * n_patches_y),
        n_embd(hparams.n_embd),
        n_head(hparams.n_head),
        d_head(n_embd / n_head),
        n_layer(hparams.n_layer),
        eps(hparams.eps),
        kq_scale(1.0f / sqrtf((float) d_head)) {
        // sanity: ensure vision model layers loaded
        if ((int) model.layers.size() != n_layer) {
            throw std::runtime_error("vision_model: expected " + std::to_string(n_layer) + " layers, found " +
                                     std::to_string(model.layers.size()));
        }
        for (int il = 0; il < n_layer; ++il) {
            auto & layer = model.layers[il];
            if (!layer.q_w || !layer.k_w || !layer.v_w || !layer.o_w || !layer.ff_up_w || !layer.ff_down_w) {
                throw std::runtime_error("vision_model: missing tensor in layer " + std::to_string(il));
            }
        }
        struct ggml_init_params params = {
            /*.mem_size   =*/ctx->buf_compute_meta.size(),
            /*.mem_buffer =*/ctx->buf_compute_meta.data(),
            /*.no_alloc   =*/true,
        };
        ctx0_ptr.reset(ggml_init(params));
        ctx0 = ctx0_ptr.get();
        gf   = ggml_new_graph(ctx0);
    }

    ggml_cgraph * build_siglip() {
        ggml_tensor * inp = build_inp();
        ggml_tensor * cur =
            build_vit(inp, n_patches, NORM_TYPE_NORMAL, hparams.ffn_op, model.position_embeddings, nullptr);

        if (ctx->proj_type == PROJECTOR_TYPE_GEMMA3) {
            const int batch_size = 1;
            GGML_ASSERT(n_patches_x == n_patches_y);
            const int patches_per_image = n_patches_x;
            const int kernel_size       = hparams.proj_scale_factor;

            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
            cur = ggml_reshape_4d(ctx0, cur, patches_per_image, patches_per_image, n_embd, batch_size);

            // doing a pool2d to reduce the number of output tokens
            cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
            cur = ggml_reshape_3d(ctx0, cur, cur->ne[0] * cur->ne[0], n_embd, batch_size);
            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            // apply norm before projection
            cur = ggml_rms_norm(ctx0, cur, eps);
            cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);

            // apply projection
            cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, model.mm_input_proj_w)), cur);

        } else if (ctx->proj_type == PROJECTOR_TYPE_IDEFICS3) {
            // https://github.com/huggingface/transformers/blob/0a950e0bbe1ed58d5401a6b547af19f15f0c195e/src/transformers/models/idefics3/modeling_idefics3.py#L578

            const int scale_factor = model.hparams.proj_scale_factor;
            const int n_embd       = cur->ne[0];
            const int seq          = cur->ne[1];
            const int bsz          = 1;  // batch size, always 1 for now since we don't support batching
            const int height       = std::sqrt(seq);
            const int width        = std::sqrt(seq);
            GGML_ASSERT(scale_factor != 0);
            cur = ggml_reshape_4d(ctx0, cur, n_embd * scale_factor, width / scale_factor, height, bsz);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_reshape_4d(ctx0, ggml_cont(ctx0, cur), n_embd * scale_factor * scale_factor,
                                  height / scale_factor, width / scale_factor, bsz);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, cur), n_embd * scale_factor * scale_factor,
                                  seq / (scale_factor * scale_factor), bsz);
            cur = ggml_mul_mat(ctx0, model.projection, cur);
        } else if (ctx->proj_type == PROJECTOR_TYPE_NONE) {
            // Standard SigLIP / CLIP: mean-pool the sequence dimension.
            // cur shape: [n_embd, seq]
            cur = ggml_mean(ctx0, cur);  // result shape: [n_embd, 1]

            // Optionally normalize (keep same behaviour as transformers implementation)
            cur = ggml_rms_norm(ctx0, cur, eps);

            // If the model provides a projection tensor use it, otherwise skip.
            if (model.projection) {
                cur = ggml_mul_mat(ctx0, model.projection, cur);
            }
        } else {
            GGML_ABORT("SigLIP: Unsupported projector type");
        }

        // build the graph
        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    ggml_cgraph * build_pixtral() {
        const int n_merge = hparams.spatial_merge_size;

        // 2D input positions
        ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
        ggml_set_name(pos_h, "pos_h");
        ggml_set_input(pos_h);

        ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
        ggml_set_name(pos_w, "pos_w");
        ggml_set_input(pos_w);

        auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
            return build_rope_2d(ctx0, cur, pos_h, pos_w, hparams.rope_theta);
        };

        ggml_tensor * inp = build_inp();
        ggml_tensor * cur = build_vit(inp, n_patches, NORM_TYPE_RMS, hparams.ffn_op,
                                      nullptr,  // no learned pos embd
                                      add_pos);

        // mistral small 3.1 patch merger
        // ref: https://github.com/huggingface/transformers/blob/7a3e208892c06a5e278144eaf38c8599a42f53e7/src/transformers/models/mistral3/modeling_mistral3.py#L67
        if (model.mm_patch_merger_w) {
            GGML_ASSERT(hparams.spatial_merge_size > 0);

            cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, eps), model.mm_input_norm_w);

            // reshape image tokens to 2D grid
            cur = ggml_reshape_3d(ctx0, cur, n_embd, n_patches_x, n_patches_y);
            cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);  // [x, y, n_embd]
            cur = ggml_cont(ctx0, cur);

            // torch.nn.functional.unfold is just an im2col under the hood
            // we just need a dummy kernel to make it work
            ggml_tensor * kernel = ggml_view_3d(ctx0, cur, n_merge, n_merge, cur->ne[2], 0, 0, 0);
            cur                  = ggml_im2col(ctx0, kernel, cur, n_merge, n_merge, 0, 0, 1, 1, true, inp->type);

            // project to n_embd
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
            cur = ggml_mul_mat(ctx0, model.mm_patch_merger_w, cur);
        }

        // LlavaMultiModalProjector (always using GELU activation)
        {
            cur = ggml_mul_mat(ctx0, model.mm_1_w, cur);
            if (model.mm_1_b) {
                cur = ggml_add(ctx0, cur, model.mm_1_b);
            }

            cur = ggml_gelu(ctx0, cur);
            cur = ggml_mul_mat(ctx0, model.mm_2_w, cur);
            if (model.mm_2_b) {
                cur = ggml_add(ctx0, cur, model.mm_2_b);
            }
        }

        // arrangement of the [IMG_BREAK] token
        {
            // not efficient, but works
            // the trick is to view the embeddings as a 3D tensor with shape [n_embd, n_patches_per_row, n_rows]
            // and then concatenate the [IMG_BREAK] token to the end of each row, aka n_patches_per_row dimension
            // after the concatenation, we have a tensor with shape [n_embd, n_patches_per_row + 1, n_rows]

            const int p_y             = n_merge > 0 ? n_patches_y / n_merge : n_patches_y;
            const int p_x             = n_merge > 0 ? n_patches_x / n_merge : n_patches_x;
            const int p_total         = p_x * p_y;
            const int n_embd_text     = cur->ne[0];
            const int n_tokens_output = p_total + p_y - 1;  // one [IMG_BREAK] per row, except the last row

            ggml_tensor * tmp = ggml_reshape_3d(ctx0, cur, n_embd_text, p_x, p_y);
            ggml_tensor * tok = ggml_new_tensor_3d(ctx0, tmp->type, n_embd_text, 1, p_y);
            tok               = ggml_scale(ctx0, tok, 0.0);  // clear the tensor
            tok               = ggml_add(ctx0, tok, model.token_embd_img_break);
            tmp               = ggml_concat(ctx0, tmp, tok, 1);
            cur = ggml_view_2d(ctx0, tmp, n_embd_text, n_tokens_output, ggml_row_size(tmp->type, n_embd_text), 0);
        }

        // build the graph
        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    // Qwen2VL and Qwen2.5VL use M-RoPE
    ggml_cgraph * build_qwen2vl() {
        GGML_ASSERT(model.patch_bias == nullptr);
        GGML_ASSERT(model.class_embedding == nullptr);

        const int  batch_size       = 1;
        const bool use_window_attn  = hparams.n_wa_pattern > 0;
        const int  n_wa_pattern     = hparams.n_wa_pattern;
        const int  n_pos            = n_patches;
        const int  num_position_ids = n_pos * 4;  // m-rope requires 4 dim per position

        norm_type norm_t = ctx->proj_type == PROJECTOR_TYPE_QWEN25VL ? NORM_TYPE_RMS      // qwen 2.5 vl
                                                                       :
                                                                       NORM_TYPE_NORMAL;  // qwen 2 vl

        int mrope_sections[4] = { d_head / 4, d_head / 4, d_head / 4, d_head / 4 };

        ggml_tensor * inp_raw = build_inp_raw();
        ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

        GGML_ASSERT(img.nx % (patch_size * 2) == 0);
        GGML_ASSERT(img.ny % (patch_size * 2) == 0);

        // second conv dimension
        {
            auto inp_1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
            inp        = ggml_add(ctx0, inp, inp_1);

            inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 2, 0, 3));  // [w, h, c, b] -> [c, w, h, b]
            inp = ggml_reshape_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
            inp = ggml_reshape_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
            inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 0, 2, 1, 3));
            inp = ggml_reshape_3d(ctx0, inp, n_embd, n_patches_x * n_patches_y, batch_size);
        }

        ggml_tensor * inpL           = inp;
        ggml_tensor * window_mask    = nullptr;
        ggml_tensor * window_idx     = nullptr;
        ggml_tensor * inv_window_idx = nullptr;

        ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        // pre-layernorm
        if (model.pre_ln_w) {
            inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
        }

        if (use_window_attn) {
            // handle window attention inputs
            inv_window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
            ggml_set_name(inv_window_idx, "inv_window_idx");
            ggml_set_input(inv_window_idx);
            // mask for window attention
            window_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos);
            ggml_set_name(window_mask, "window_mask");
            ggml_set_input(window_mask);

            // inpL shape: [n_embd, n_patches_x * n_patches_y, batch_size]
            GGML_ASSERT(batch_size == 1);
            inpL = ggml_reshape_2d(ctx0, inpL, n_embd * 4, n_patches_x * n_patches_y * batch_size / 4);
            inpL = ggml_get_rows(ctx0, inpL, inv_window_idx);
            inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_patches_x * n_patches_y, batch_size);
        }

        // loop over layers
        for (int il = 0; il < n_layer; il++) {
            auto &     layer     = model.layers[il];
            const bool full_attn = use_window_attn ? (il + 1) % n_wa_pattern == 0 : true;

            ggml_tensor * cur = inpL;  // inpL = residual, cur = hidden_states

            // layernorm1
            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
            cb(cur, "ln1", il);

            // self-attention
            {
                ggml_tensor * Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.q_w, cur), layer.q_b);
                ggml_tensor * Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.k_w, cur), layer.k_b);
                ggml_tensor * Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.v_w, cur), layer.v_b);

                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
                Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
                Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                // apply M-RoPE
                Qcur = ggml_rope_multi(ctx0, Qcur, positions, nullptr, d_head / 2, mrope_sections,
                                       GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
                Kcur = ggml_rope_multi(ctx0, Kcur, positions, nullptr, d_head / 2, mrope_sections,
                                       GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

                cb(Qcur, "Qcur_rope", il);
                cb(Kcur, "Kcur_rope", il);

                ggml_tensor * attn_mask = full_attn ? nullptr : window_mask;

                cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, attn_mask, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            // re-add the layer input, e.g., residual
            cur = ggml_add(ctx0, cur, inpL);

            inpL = cur;  // inpL = residual, cur = hidden_states

            cb(cur, "ffn_inp", il);

            // layernorm2
            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
            cb(cur, "ffn_inp_normed", il);

            // ffn
            cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w,
                            layer.ff_down_b, hparams.ffn_op, il);

            cb(cur, "ffn_out", il);

            // residual 2
            cur = ggml_add(ctx0, inpL, cur);
            cb(cur, "layer_out", il);

            inpL = cur;
        }

        // post-layernorm
        if (model.post_ln_w) {
            inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer);
        }

        // multimodal projection
        ggml_tensor * embeddings = inpL;
        embeddings               = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);

        embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
        embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);

        // GELU activation
        embeddings = ggml_gelu(ctx0, embeddings);

        // Second linear layer
        embeddings = ggml_mul_mat(ctx0, model.mm_1_w, embeddings);
        embeddings = ggml_add(ctx0, embeddings, model.mm_1_b);

        if (use_window_attn) {
            window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
            ggml_set_name(window_idx, "window_idx");
            ggml_set_input(window_idx);

            // embeddings shape: [n_embd, n_patches_x * n_patches_y, batch_size]
            GGML_ASSERT(batch_size == 1);
            embeddings = ggml_reshape_2d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4);
            embeddings = ggml_get_rows(ctx0, embeddings, window_idx);
            embeddings =
                ggml_reshape_3d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4, batch_size);
        }

        // build the graph
        ggml_build_forward_expand(gf, embeddings);

        return gf;
    }

    ggml_cgraph * build_minicpmv() {
        const int batch_size = 1;

        GGML_ASSERT(model.class_embedding == nullptr);
        const int n_pos = n_patches;

        // position embeddings for the projector (not for ViT)
        int           n_output_dim = clip_n_mmproj_embd(ctx);
        // 3D pos_embed input: [n_output_dim, n_pos, batch_size]
        ggml_tensor * pos_embed    = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_output_dim, n_pos, batch_size);
        ggml_set_name(pos_embed, "pos_embed");
        ggml_set_input(pos_embed);

        // for selecting learned pos embd, used by ViT
        struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        ggml_tensor * learned_pos_embd = ggml_get_rows(ctx0, model.position_embeddings, positions);

        ggml_tensor * inp = build_inp();
        ggml_tensor * embeddings =
            build_vit(inp, n_patches, NORM_TYPE_NORMAL, hparams.ffn_op, learned_pos_embd, nullptr);

        // resampler projector (it is just another transformer)

        ggml_tensor * q = model.mm_model_query;
        ggml_tensor * v = ggml_mul_mat(ctx0, model.mm_model_kv_proj, embeddings);

        // norm
        q = build_norm(q, model.mm_model_ln_q_w, model.mm_model_ln_q_b, NORM_TYPE_NORMAL, eps, -1);
        v = build_norm(v, model.mm_model_ln_kv_w, model.mm_model_ln_kv_b, NORM_TYPE_NORMAL, eps, -1);

        // k = v + pos_embed
        ggml_tensor * k = ggml_add(ctx0, v, pos_embed);

        // attention
        {
            int       n_embd    = clip_n_mmproj_embd(ctx);
            const int d_head    = 128;
            int       n_head    = n_embd / d_head;
            int       num_query = 96;
            if (ctx->minicpmv_version == 2) {
                num_query = 96;
            } else if (ctx->minicpmv_version == 3) {
                num_query = 64;
            } else if (ctx->minicpmv_version == 4) {
                num_query = 64;
            }

            ggml_tensor * Q = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_q_w, q), model.mm_model_attn_q_b);
            ggml_tensor * K = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_k_w, k), model.mm_model_attn_k_b);
            ggml_tensor * V = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_v_w, v), model.mm_model_attn_v_b);

            Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, num_query);
            K = ggml_reshape_3d(ctx0, K, d_head, n_head, n_pos);
            V = ggml_reshape_3d(ctx0, V, d_head, n_head, n_pos);

            cb(Q, "resampler_Q", -1);
            cb(K, "resampler_K", -1);
            cb(V, "resampler_V", -1);

            embeddings = build_attn(model.mm_model_attn_o_w, model.mm_model_attn_o_b, Q, K, V, nullptr, kq_scale, -1);
            cb(embeddings, "resampler_attn_out", -1);
        }
        // layernorm
        embeddings =
            build_norm(embeddings, model.mm_model_ln_post_w, model.mm_model_ln_post_b, NORM_TYPE_NORMAL, eps, -1);

        // projection
        embeddings = ggml_mul_mat(ctx0, model.mm_model_proj, embeddings);

        // build the graph
        ggml_build_forward_expand(gf, embeddings);

        return gf;
    }

    ggml_cgraph * build_internvl() {
        GGML_ASSERT(model.class_embedding != nullptr);
        GGML_ASSERT(model.position_embeddings != nullptr);

        const int     n_pos = n_patches + 1;
        ggml_tensor * inp   = build_inp();

        // add CLS token
        inp = ggml_concat(ctx0, inp, model.class_embedding, 1);

        // The larger models use a different ViT, which uses RMS norm instead of layer norm
        // ref: https://github.com/ggml-org/llama.cpp/pull/13443#issuecomment-2869786188
        norm_type norm_t = (hparams.n_embd == 3200 && hparams.n_layer == 45) ?
                               NORM_TYPE_RMS      // 6B ViT (Used by InternVL 2.5/3 - 26B, 38B, 78B)
                               :
                               NORM_TYPE_NORMAL;  // 300M ViT (Used by all smaller InternVL models)

        ggml_tensor * cur = build_vit(inp, n_pos, norm_t, hparams.ffn_op, model.position_embeddings, nullptr);

        // remove CLS token
        cur = ggml_view_2d(ctx0, cur, n_embd, n_patches, ggml_row_size(cur->type, n_embd), 0);

        // pixel shuffle
        {
            const int scale_factor = model.hparams.proj_scale_factor;
            const int bsz          = 1;  // batch size, always 1 for now since we don't support batching
            const int height       = n_patches_y;
            const int width        = n_patches_x;
            GGML_ASSERT(scale_factor > 0);
            cur = ggml_reshape_4d(ctx0, cur, n_embd * scale_factor, height / scale_factor, width, bsz);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_reshape_4d(ctx0, ggml_cont(ctx0, cur), n_embd * scale_factor * scale_factor,
                                  height / scale_factor, width / scale_factor, bsz);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            // flatten to 2D
            cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, cur), n_embd * scale_factor * scale_factor,
                                  cur->ne[1] * cur->ne[2]);
        }

        // projector (always using GELU activation)
        {
            // projector LayerNorm uses pytorch's default eps = 1e-5
            // ref: https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct/blob/a34d3e4e129a5856abfd6aa6de79776484caa14e/modeling_internvl_chat.py#L79
            cur = build_norm(cur, model.mm_0_w, model.mm_0_b, NORM_TYPE_NORMAL, 1e-5, -1);
            cur = ggml_mul_mat(ctx0, model.mm_1_w, cur);
            cur = ggml_add(ctx0, cur, model.mm_1_b);
            cur = ggml_gelu(ctx0, cur);
            cur = ggml_mul_mat(ctx0, model.mm_3_w, cur);
            cur = ggml_add(ctx0, cur, model.mm_3_b);
        }

        // build the graph
        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    // this graph is used by llava, granite and glm
    // due to having embedding_stack (used by granite), we cannot reuse build_vit
    ggml_cgraph * build_llava() {
        const int batch_size = 1;
        const int n_pos      = n_patches + (model.class_embedding ? 1 : 0);

        GGML_ASSERT(n_patches_x == n_patches_y && "only square images supported");

        // Calculate the deepest feature layer based on hparams and projector type
        int max_feature_layer = n_layer;
        {
            // Get the index of the second to last layer; this is the default for models that have a llava projector
            int il_last               = hparams.n_layer - 1;
            int deepest_feature_layer = -1;

            if (ctx->proj_type == PROJECTOR_TYPE_MINICPMV || ctx->proj_type == PROJECTOR_TYPE_GLM_EDGE) {
                il_last += 1;
            }

            // If we set explicit vision feature layers, only go up to the deepest one
            // NOTE: only used by granite-vision models for now
            for (const auto & feature_layer : hparams.vision_feature_layer) {
                if (feature_layer > deepest_feature_layer) {
                    deepest_feature_layer = feature_layer;
                }
            }
            max_feature_layer = deepest_feature_layer < 0 ? il_last : deepest_feature_layer;
        }

        ggml_tensor * inp = build_inp();

        // concat class_embeddings and patch_embeddings
        if (model.class_embedding) {
            inp = ggml_concat(ctx0, inp, model.class_embedding, 1);
        }

        ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        inp = ggml_add(ctx0, inp, ggml_get_rows(ctx0, model.position_embeddings, positions));

        ggml_tensor * inpL = inp;

        // pre-layernorm
        if (model.pre_ln_w) {
            inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, NORM_TYPE_NORMAL, eps, -1);
            cb(inpL, "pre_ln", -1);
        }

        std::vector<ggml_tensor *> embedding_stack;
        const auto &               vision_feature_layer = hparams.vision_feature_layer;

        // loop over layers
        for (int il = 0; il < max_feature_layer; il++) {
            auto &        layer = model.layers[il];
            ggml_tensor * cur   = inpL;  // inpL = residual, cur = hidden_states

            // If this is an embedding feature layer, save the output.
            // NOTE: 0 index here refers to the input to the encoder.
            if (vision_feature_layer.find(il) != vision_feature_layer.end()) {
                embedding_stack.push_back(cur);
            }

            // layernorm1
            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
            cb(cur, "layer_inp_normed", il);

            // self-attention
            {
                auto add_bias_cast = [&](ggml_tensor * tensor, ggml_tensor * bias) {
                    if (!bias) {
                        return tensor;
                    }
                    ggml_tensor * b = bias;
                    if (b->type != GGML_TYPE_F32) {
                        b = ggml_cast(ctx0, b, GGML_TYPE_F32);
                    }
                    return ggml_add(ctx0, tensor, b);
                };

                ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
                Qcur               = add_bias_cast(Qcur, layer.q_b);

                ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
                Kcur               = add_bias_cast(Kcur, layer.k_b);

                ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
                Vcur               = add_bias_cast(Vcur, layer.v_b);

                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
                Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
                Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            // re-add the layer input, e.g., residual
            cur = ggml_add(ctx0, cur, inpL);

            inpL = cur;  // inpL = residual, cur = hidden_states

            cb(cur, "ffn_inp", il);

            // layernorm2
            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
            cb(cur, "ffn_inp_normed", il);

            // ffn
            cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w,
                            layer.ff_down_b, hparams.ffn_op, il);

            cb(cur, "ffn_out", il);

            // residual 2
            cur = ggml_add(ctx0, inpL, cur);
            cb(cur, "layer_out", il);

            inpL = cur;
        }

        // post-layernorm
        if (model.post_ln_w) {
            inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);
        }

        ggml_tensor * embeddings = inpL;

        // process vision feature layers (used by granite)
        {
            // final layer is a vision feature layer
            if (vision_feature_layer.find(max_feature_layer) != vision_feature_layer.end()) {
                embedding_stack.push_back(inpL);
            }

            // If feature layers are explicitly set, stack them (if we have multiple)
            if (!embedding_stack.empty()) {
                embeddings = embedding_stack[0];
                for (size_t i = 1; i < embedding_stack.size(); i++) {
                    embeddings = ggml_concat(ctx0, embeddings, embedding_stack[i], 0);
                }
            }
        }

        // llava projector (also used by granite)
        if (ctx->has_llava_projector) {
            embeddings = ggml_reshape_2d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1]);

            ggml_tensor * patches = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
            ggml_set_name(patches, "patches");
            ggml_set_input(patches);

            // shape [1, 576, 1024]
            // ne is whcn, ne = [1024, 576, 1, 1]
            embeddings = ggml_get_rows(ctx0, embeddings, patches);

            // print_tensor_info(embeddings, "embeddings");

            // llava projector
            if (ctx->proj_type == PROJECTOR_TYPE_MLP) {
                embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
                embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);

                embeddings = ggml_gelu(ctx0, embeddings);
                if (model.mm_2_w) {
                    embeddings = ggml_mul_mat(ctx0, model.mm_2_w, embeddings);
                    embeddings = ggml_add(ctx0, embeddings, model.mm_2_b);
                }
            } else if (ctx->proj_type == PROJECTOR_TYPE_MLP_NORM) {
                embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
                embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);
                // ggml_tensor_printf(embeddings, "mm_0_w",0,true,false);
                // First LayerNorm
                embeddings = ggml_norm(ctx0, embeddings, eps);
                embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_1_w), model.mm_1_b);

                // GELU activation
                embeddings = ggml_gelu(ctx0, embeddings);

                // Second linear layer
                embeddings = ggml_mul_mat(ctx0, model.mm_3_w, embeddings);
                embeddings = ggml_add(ctx0, embeddings, model.mm_3_b);

                // Second LayerNorm
                embeddings = ggml_norm(ctx0, embeddings, eps);
                embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_4_w), model.mm_4_b);
            } else if (ctx->proj_type == PROJECTOR_TYPE_LDP) {
                // MobileVLM projector
                int           n_patch = 24;
                ggml_tensor * mlp_1   = ggml_mul_mat(ctx0, model.mm_model_mlp_1_w, embeddings);
                mlp_1                 = ggml_add(ctx0, mlp_1, model.mm_model_mlp_1_b);
                mlp_1                 = ggml_gelu(ctx0, mlp_1);
                ggml_tensor * mlp_3   = ggml_mul_mat(ctx0, model.mm_model_mlp_3_w, mlp_1);
                mlp_3                 = ggml_add(ctx0, mlp_3, model.mm_model_mlp_3_b);
                // mlp_3 shape = [1, 576, 2048], ne = [2048, 576, 1, 1]

                // block 1
                ggml_tensor * block_1 = nullptr;
                {
                    // transpose from [1, 576, 2048] --> [1, 2048, 576] --> [1, 2048, 24, 24]
                    mlp_3   = ggml_cont(ctx0, ggml_permute(ctx0, mlp_3, 1, 0, 2, 3));
                    mlp_3   = ggml_reshape_4d(ctx0, mlp_3, n_patch, n_patch, mlp_3->ne[1], mlp_3->ne[2]);
                    // stride = 1, padding = 1, bias is nullptr
                    block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_block_1_block_0_0_w, mlp_3, 1, 1, 1, 1, 1, 1);

                    // layer norm
                    // // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_0_1_w),
                                       model.mm_model_block_1_block_0_1_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));

                    // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                    // hardswish
                    ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

                    block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1],
                                           block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
                    // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                    // pointwise conv
                    block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0] * block_1->ne[1] * block_1->ne[2],
                                              block_1->ne[3]);
                    block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_1_fc1_w, block_1);
                    block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc1_b);
                    block_1 = ggml_relu(ctx0, block_1);
                    block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_1_fc2_w, block_1);
                    block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc2_b);
                    block_1 = ggml_hardsigmoid(ctx0, block_1);
                    // block_1_hw shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1], block_1 shape = [1, 2048], ne = [2048, 1, 1, 1]
                    block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
                    block_1 = ggml_mul(ctx0, block_1_hw, block_1);

                    int w = block_1->ne[0], h = block_1->ne[1];
                    block_1 = ggml_reshape_3d(ctx0, block_1, w * h, block_1->ne[2], block_1->ne[3]);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 0, 2, 3));

                    // block_1 shape = [1, 24*24, 2048], ne = [24*24, 2048, 1]
                    block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_2_0_w, block_1);
                    block_1 = ggml_reshape_4d(ctx0, block_1, block_1->ne[0], w, h, block_1->ne[3]);

                    // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_2_1_w),
                                       model.mm_model_block_1_block_2_1_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    // block1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                    // residual
                    block_1 = ggml_add(ctx0, mlp_3, block_1);
                }

                // block_2
                {
                    // stride = 2
                    block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_block_2_block_0_0_w, block_1, 2, 2, 1, 1, 1, 1);

                    // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
                    // layer norm
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_0_1_w),
                                       model.mm_model_block_2_block_0_1_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
                    // hardswish
                    ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

                    // not sure the parameters is right for globalAvgPooling
                    block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1],
                                           block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
                    // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                    // pointwise conv
                    block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0] * block_1->ne[1] * block_1->ne[2],
                                              block_1->ne[3]);
                    block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_1_fc1_w, block_1);
                    block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc1_b);
                    block_1 = ggml_relu(ctx0, block_1);
                    block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_1_fc2_w, block_1);
                    block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc2_b);
                    block_1 = ggml_hardsigmoid(ctx0, block_1);

                    // block_1_hw shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1], block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                    block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
                    block_1 = ggml_mul(ctx0, block_1_hw, block_1);

                    int w = block_1->ne[0], h = block_1->ne[1];
                    block_1 = ggml_reshape_3d(ctx0, block_1, w * h, block_1->ne[2], block_1->ne[3]);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 0, 2, 3));
                    // block_1 shape = [1, 24*24, 2048], ne = [24*24, 2048, 1]
                    block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_2_0_w, block_1);
                    block_1 = ggml_reshape_4d(ctx0, block_1, block_1->ne[0], w, h, block_1->ne[3]);

                    // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_2_1_w),
                                       model.mm_model_block_2_block_2_1_b);
                    block_1 =
                        ggml_reshape_3d(ctx0, block_1, block_1->ne[0], block_1->ne[1] * block_1->ne[2], block_1->ne[3]);
                    // block_1 shape = [1, 144, 2048], ne = [2048, 144, 1]
                }
                embeddings = block_1;
            } else if (ctx->proj_type == PROJECTOR_TYPE_LDPV2) {
                int           n_patch = 24;
                ggml_tensor * mlp_0   = ggml_mul_mat(ctx0, model.mm_model_mlp_0_w, embeddings);
                mlp_0                 = ggml_add(ctx0, mlp_0, model.mm_model_mlp_0_b);
                mlp_0                 = ggml_gelu(ctx0, mlp_0);
                ggml_tensor * mlp_2   = ggml_mul_mat(ctx0, model.mm_model_mlp_2_w, mlp_0);
                mlp_2                 = ggml_add(ctx0, mlp_2, model.mm_model_mlp_2_b);
                // mlp_2 ne = [2048, 576, 1, 1]
                // // AVG Pool Layer 2*2, strides = 2
                mlp_2                 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_2, 1, 0, 2, 3));
                // mlp_2 ne = [576, 2048, 1, 1]
                mlp_2                 = ggml_reshape_4d(ctx0, mlp_2, n_patch, n_patch, mlp_2->ne[1], mlp_2->ne[2]);
                // mlp_2 ne [24, 24, 2048, 1]
                mlp_2                 = ggml_pool_2d(ctx0, mlp_2, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
                // weight ne = [3, 3, 2048, 1]
                ggml_tensor * peg_0   = ggml_conv_2d_dw(ctx0, model.mm_model_peg_0_w, mlp_2, 1, 1, 1, 1, 1, 1);
                peg_0                 = ggml_cont(ctx0, ggml_permute(ctx0, peg_0, 1, 2, 0, 3));
                peg_0                 = ggml_add(ctx0, peg_0, model.mm_model_peg_0_b);
                mlp_2                 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_2, 1, 2, 0, 3));
                peg_0                 = ggml_add(ctx0, peg_0, mlp_2);
                peg_0      = ggml_reshape_3d(ctx0, peg_0, peg_0->ne[0], peg_0->ne[1] * peg_0->ne[2], peg_0->ne[3]);
                embeddings = peg_0;
            } else {
                GGML_ABORT("fatal error");
            }
        }

        // glm projector
        else if (ctx->proj_type == PROJECTOR_TYPE_GLM_EDGE) {
            size_t gridsz = (size_t) sqrt(embeddings->ne[1]);
            embeddings    = ggml_cont(ctx0, ggml_permute(ctx0, embeddings, 1, 0, 2, 3));
            embeddings    = ggml_reshape_3d(ctx0, embeddings, gridsz, gridsz, embeddings->ne[1]);
            embeddings    = ggml_conv_2d(ctx0, model.mm_model_adapter_conv_w, embeddings, 2, 2, 0, 0, 1, 1);
            embeddings =
                ggml_reshape_3d(ctx0, embeddings, embeddings->ne[0] * embeddings->ne[1], embeddings->ne[2], batch_size);
            embeddings = ggml_cont(ctx0, ggml_permute(ctx0, embeddings, 1, 0, 2, 3));
            embeddings = ggml_add(ctx0, embeddings, model.mm_model_adapter_conv_b);
            // GLU
            {
                embeddings = ggml_mul_mat(ctx0, model.mm_model_mlp_0_w, embeddings);
                embeddings = ggml_norm(ctx0, embeddings, eps);
                embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_model_ln_q_w), model.mm_model_ln_q_b);
                embeddings = ggml_gelu_inplace(ctx0, embeddings);
                ggml_tensor * x = embeddings;
                embeddings      = ggml_mul_mat(ctx0, model.mm_model_mlp_2_w, embeddings);
                x               = ggml_mul_mat(ctx0, model.mm_model_mlp_1_w, x);
                embeddings      = ggml_silu_inplace(ctx0, embeddings);
                embeddings      = ggml_mul(ctx0, embeddings, x);
                embeddings      = ggml_mul_mat(ctx0, model.mm_model_mlp_3_w, embeddings);
            }
            // arrangement of BOI/EOI token embeddings
            // note: these embeddings are not present in text model, hence we cannot process them as text tokens
            // see: https://huggingface.co/THUDM/glm-edge-v-2b/blob/main/siglip.py#L53
            {
                embeddings = ggml_concat(ctx0, model.mm_glm_tok_boi, embeddings, 1);  // BOI
                embeddings = ggml_concat(ctx0, embeddings, model.mm_glm_tok_eoi, 1);  // EOI
            }
        }

        else {
            GGML_ABORT("llava: unknown projector type");
        }

        // build the graph
        ggml_build_forward_expand(gf, embeddings);

        return gf;
    }

  private:
    //
    // utility functions
    //

    void cb(ggml_tensor * cur, const char * name, int il) const {
        // TODO: implement this
        GGML_UNUSED(cur);
        GGML_UNUSED(name);
        GGML_UNUSED(il);
    }

    // build vision transformer (ViT) cgraph
    // this function should cover most of the models
    // if your model has specific features, you should probably duplicate this function
    ggml_tensor * build_vit(ggml_tensor * inp, int64_t n_pos, norm_type norm_t, ffn_op_type ffn_t,
                            ggml_tensor *                                                   learned_pos_embd,
                            std::function<ggml_tensor *(ggml_tensor *, const clip_layer &)> add_pos) {
        if (learned_pos_embd) {
            ggml_tensor * pos = learned_pos_embd;
            if (pos->type != GGML_TYPE_F32) {
                pos = ggml_cast(ctx0, pos, GGML_TYPE_F32);
            }
            // log_add_details("build_vit", "learned_pos_embd add", -1, inp, pos);  // Logging
            inp = ggml_add(ctx0, inp, pos);
            cb(inp, "pos_embed", -1);
        }

        ggml_tensor * inpL = inp;

        // pre-layernorm
        if (model.pre_ln_w) {
            inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
            cb(inpL, "pre_ln", -1);
        }

        // loop over layers
        for (int il = 0; il < n_layer; il++) {
            auto &        layer = model.layers[il];
            ggml_tensor * cur   = inpL;  // inpL = residual, cur = hidden_states

            // layernorm1
            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
            cb(cur, "layer_inp_normed", il);

            // self-attention
            {
                ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
                if (layer.q_b) {
                    // log_add_details("build_vit", "Q_bias add", il, Qcur, layer.q_b);
                    Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                }

                ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
                if (layer.k_b) {
                    // log_add_details("build_vit", "K_bias add", il, Kcur, layer.k_b);
                    Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                }

                ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
                if (layer.v_b) {
                    // log_add_details("build_vit", "V_bias add", il, Vcur, layer.v_b);
                    Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                }

                if (layer.q_norm) {
                    Qcur = build_norm(Qcur, layer.q_norm, NULL, norm_t, eps, il);
                    cb(Qcur, "Qcur_norm", il);
                }

                if (layer.k_norm) {
                    Kcur = build_norm(Kcur, layer.k_norm, NULL, norm_t, eps, il);
                    cb(Kcur, "Kcur_norm", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
                Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
                Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                if (add_pos) {
                    Qcur = add_pos(Qcur, layer);
                    Kcur = add_pos(Kcur, layer);
                    cb(Qcur, "Qcur_pos", il);
                    cb(Kcur, "Kcur_pos", il);
                }

                cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            if (layer.ls_1_w) {
                cur = ggml_mul(ctx0, cur, layer.ls_1_w);
                cb(cur, "attn_out_scaled", il);
            }

            // re-add the layer input, e.g., residual
            // log_add_details("build_vit", "residual1 add", il, cur, inpL);  // Logging
            cur = ggml_add(ctx0, cur, inpL);

            inpL = cur;  // inpL = residual, cur = hidden_states

            cb(cur, "ffn_inp", il);

            // layernorm2
            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
            cb(cur, "ffn_inp_normed", il);

            // ffn
            cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w,
                            layer.ff_down_b, ffn_t, il);

            cb(cur, "ffn_out", il);

            if (layer.ls_2_w) {
                cur = ggml_mul(ctx0, cur, layer.ls_2_w);
                cb(cur, "ffn_out_scaled", il);
            }

            // residual 2
            // log_add_details("build_vit", "residual2 add", il, cur, inpL);  // Logging
            cur = ggml_add(ctx0, cur, inpL);
            cb(cur, "layer_out", il);
            inpL = cur;
        }

        // post-layernorm
        if (model.post_ln_w) {
            inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, -1);
        }
        return inpL;
    }

    // build the input after conv2d (inp_raw --> patches)
    // returns tensor with shape [n_embd, n_patches]
    ggml_tensor * build_inp() {
        ggml_tensor * inp_raw = build_inp_raw();
        ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp               = ggml_reshape_2d(ctx0, inp, n_patches, n_embd);
        inp               = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
        if (model.patch_bias) {
            ggml_tensor * patch_bias = model.patch_bias;
            if (patch_bias->type != GGML_TYPE_F32) {
                patch_bias = ggml_cast(ctx0, patch_bias, GGML_TYPE_F32);
            }
            // log_add_details("build_inp", "patch_bias add", -1, inp, patch_bias);  // Logging
            inp = ggml_add(ctx0, inp, patch_bias);
            cb(inp, "patch_bias", -1);
        }
        return inp;
    }

    ggml_tensor * build_inp_raw() {
        ggml_tensor * inp_raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, img.nx, img.ny, 3);
        ggml_set_name(inp_raw, "inp_raw");
        ggml_set_input(inp_raw);
        return inp_raw;
    }

    ggml_tensor * build_norm(ggml_tensor * cur, ggml_tensor * mw, ggml_tensor * mb, norm_type type, float norm_eps,
                             int il) const {
        cur = type == NORM_TYPE_RMS ? ggml_rms_norm(ctx0, cur, norm_eps) : ggml_norm(ctx0, cur, norm_eps);

        if (mw || mb) {
            cb(cur, "norm", il);
        }

        if (mw) {
            ggml_tensor * w = mw->type == GGML_TYPE_F32 ? mw : ggml_cast(ctx0, mw, GGML_TYPE_F32);
            cur             = ggml_mul(ctx0, cur, w);
            if (mb) {
                cb(cur, "norm_w", il);
            }
        }

        if (mb) {
            ggml_tensor * b = mb->type == GGML_TYPE_F32 ? mb : ggml_cast(ctx0, mb, GGML_TYPE_F32);
            // log_add_details("build_norm", "bias add", il, cur, b);
            cur             = ggml_add(ctx0, cur, b);
        }

        return cur;
    }

    ggml_tensor * build_ffn(ggml_tensor * cur, ggml_tensor * up, ggml_tensor * up_b, ggml_tensor * gate,
                            ggml_tensor * gate_b, ggml_tensor * down, ggml_tensor * down_b, ffn_op_type type_op,
                            int il) const {
        auto cast_bias_if_needed = [&](ggml_tensor * b) -> ggml_tensor * {
            if (!b) {
                return b;
            }
            return b->type == GGML_TYPE_F32 ? b : ggml_cast(ctx0, b, GGML_TYPE_F32);
        };

        ggml_tensor * tmp = nullptr;
        if (up) {
            tmp = ggml_mul_mat(ctx0, up, cur);
        } else {
            tmp = cur;
        }
        cb(tmp, "ffn_up", il);

        if (up_b) {
            ggml_tensor * up_b_casted = cast_bias_if_needed(up_b);
            // log_add_details("build_ffn", "up_bias add", il, tmp, up_b_casted);  // Logging
            tmp                       = ggml_add(ctx0, tmp, up_b_casted);
            cb(tmp, "ffn_up_b", il);
        }

        if (gate) {
            cur = ggml_mul_mat(ctx0, gate, cur);
            cb(cur, "ffn_gate", il);

            if (gate_b) {
                ggml_tensor * gate_b_casted = cast_bias_if_needed(gate_b);
                // log_add_details("build_ffn", "gate_bias add", il, cur, gate_b_casted);  // Logging
                cur                         = ggml_add(ctx0, cur, gate_b_casted);
                cb(cur, "ffn_gate_b", il);
            }
        } else {
            cur = tmp;
        }

        switch (type_op) {
            case FFN_SILU:
                {
                    cur = ggml_silu(ctx0, cur);
                    cb(cur, "ffn_silu", il);
                }
                break;
            case FFN_GELU:
                {
                    cur = ggml_gelu(ctx0, cur);
                    cb(cur, "ffn_gelu", il);
                }
                break;
            case FFN_GELU_QUICK:
                {
                    cur = ggml_gelu_quick(ctx0, cur);
                    cb(cur, "ffn_relu", il);
                }
                break;
        }

        // we only support parallel ffn for now
        if (gate) {
            cur = ggml_mul(ctx0, cur, tmp);
            cb(cur, "ffn_gate_par", il);
        }

        if (down) {
            cur = ggml_mul_mat(ctx0, down, cur);
        }

        if (down_b) {
            cb(cur, "ffn_down", il);
            ggml_tensor * down_b_casted = cast_bias_if_needed(down_b);
            // log_add_details("build_ffn", "down_bias add", il, cur, down_b_casted);  // Logging
            cur                         = ggml_add(ctx0, cur, down_b_casted);
        }

        return cur;
    }

    ggml_tensor * build_attn(ggml_tensor * wo, ggml_tensor * wo_b, ggml_tensor * q_cur, ggml_tensor * k_cur,
                             ggml_tensor * v_cur, ggml_tensor * kq_mask, float kq_scale, int il) const {
        // these nodes are added to the graph together so that they are not reordered
        // by doing so, the number of splits in the graph is reduced
        ggml_build_forward_expand(gf, q_cur);
        ggml_build_forward_expand(gf, k_cur);
        ggml_build_forward_expand(gf, v_cur);

        ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
        //cb(q, "q", il);

        ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
        //cb(k, "k", il);

        ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
        v               = ggml_cont(ctx0, v);
        //cb(k, "v", il);

        ggml_tensor * cur;

        // TODO @ngxson : support flash attention
        {
            const auto n_tokens = q->ne[1];
            const auto n_head   = q->ne[2];
            // const auto n_kv     = k->ne[1]; // for flash attention

            ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            // F32 may not needed for vision encoders?
            // ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

            kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

            ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
            cur               = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cur               = ggml_cont_2d(ctx0, cur, cur->ne[0] * n_head, n_tokens);
        }

        cb(cur, "kqv_out", il);

        if (wo) {
            cur = ggml_mul_mat(ctx0, wo, cur);
        }

        if (wo_b) {
            // log_add_details("build_attn", "output_bias add", il, cur, wo_b);  // Logging
            cur = ggml_add(ctx0, cur, wo_b);
        }

        return cur;
    }

    // implementation of the 2D RoPE without adding a new op in ggml
    // this is not efficient (use double the memory), but works on all backends
    // TODO: there was a more efficient which relies on ggml_view and ggml_rope_ext_inplace, but the rope inplace does not work well with non-contiguous tensors ; we should fix that and revert back to the original implementation in https://github.com/ggml-org/llama.cpp/pull/13065
    static ggml_tensor * build_rope_2d(ggml_context * ctx0, ggml_tensor * cur, ggml_tensor * pos_h, ggml_tensor * pos_w,
                                       const float freq_base) {
        const int64_t n_dim  = cur->ne[0];
        const int64_t n_head = cur->ne[1];
        const int64_t n_pos  = cur->ne[2];

        // for example, if we have cur tensor of shape (n_dim=8, n_head, n_pos)
        // we will have a list of 4 inv_freq: 1e-0, 1e-1, 1e-2, 1e-3
        // first half of cur will use 1e-0, 1e-2 (even)
        // second half of cur will use 1e-1, 1e-3 (odd)
        // the trick here is to rotate just half of n_dim, so inv_freq will automatically be even
        //  ^ don't ask me why, it's math! -2(2i) / n_dim == -2i / (n_dim/2)
        // then for the second half, we use freq_scale to shift the inv_freq
        //  ^ why? replace (2i) with (2i+1) in the above equation
        const float freq_scale_odd = std::pow(freq_base, (float) -2 / n_dim);

        // first half
        ggml_tensor * first;
        {
            first = ggml_view_3d(ctx0, cur, n_dim / 2, n_head, n_pos, ggml_row_size(cur->type, n_dim),
                                 ggml_row_size(cur->type, n_dim * n_head), 0);
            first = ggml_rope_ext(ctx0, first,
                                  pos_h,      // positions
                                  nullptr,    // freq factors
                                  n_dim / 2,  // n_dims
                                  0, 0, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }

        // second half
        ggml_tensor * second;
        {
            second = ggml_view_3d(ctx0, cur, n_dim / 2, n_head, n_pos, ggml_row_size(cur->type, n_dim),
                                  ggml_row_size(cur->type, n_dim * n_head), n_dim / 2 * ggml_element_size(cur));
            second = ggml_cont(ctx0, second);  // copy, because ggml_rope don't play well with non-contiguous tensors
            second = ggml_rope_ext(ctx0, second,
                                   pos_w,      // positions
                                   nullptr,    // freq factors
                                   n_dim / 2,  // n_dims
                                   0, 0, freq_base, freq_scale_odd, 0.0f, 1.0f, 0.0f, 0.0f);
        }

        cur = ggml_concat(ctx0, first, second, 0);
        return cur;
    }
};

static ggml_cgraph * clip_image_build_graph(clip_ctx * ctx, const clip_image_f32_batch & imgs) {
    GGML_ASSERT(imgs.entries.size() == 1 && "n_batch > 1 is not supported");
    clip_graph graph(ctx, *imgs.entries[0]);

    ggml_cgraph * res;

    switch (ctx->proj_type) {
        case PROJECTOR_TYPE_NONE:
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_IDEFICS3:
            {
                res = graph.build_siglip();
            }
            break;
        case PROJECTOR_TYPE_PIXTRAL:
            {
                res = graph.build_pixtral();
            }
            break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
            {
                res = graph.build_qwen2vl();
            }
            break;
        case PROJECTOR_TYPE_MINICPMV:
            {
                res = graph.build_minicpmv();
            }
            break;
        case PROJECTOR_TYPE_INTERNVL:
            {
                res = graph.build_internvl();
            }
            break;
        default:
            {
                res = graph.build_llava();
            }
            break;
    }
    return res;
}

struct clip_model_loader {
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;

    clip_ctx &  ctx_clip;
    std::string fname;

    size_t model_size = 0;  // in bytes

    // TODO @ngxson : we should not pass clip_ctx here, it should be clip_vision_model
    clip_model_loader(const char * fname, clip_ctx & ctx_clip) : ctx_clip(ctx_clip), fname(fname) {
        struct ggml_context * meta = nullptr;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_context_ptr(gguf_init_from_file(fname, params));
        if (!ctx_gguf.get()) {
            throw std::runtime_error(
                string_format("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, fname));
        }

        ctx_meta.reset(meta);

        const int n_tensors = gguf_get_n_tensors(ctx_gguf.get());

        // print gguf info
        {
            std::string name;
            get_string(KEY_NAME, name, false);
            std::string description;
            get_string(KEY_DESCRIPTION, description, false);
            LOG_INF("%s: model name:   %s\n", __func__, name.c_str());
            LOG_INF("%s: description:  %s\n", __func__, description.c_str());
            LOG_INF("%s: GGUF version: %d\n", __func__, gguf_get_version(ctx_gguf.get()));
            LOG_INF("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx_gguf.get()));
            LOG_INF("%s: n_tensors:    %d\n", __func__, n_tensors);
            LOG_INF("%s: n_kv:         %d\n", __func__, (int) gguf_get_n_kv(ctx_gguf.get()));
            LOG_INF("\n");
        }

        // tensors
        {
            for (int i = 0; i < n_tensors; ++i) {
                const char *   name        = gguf_get_tensor_name(ctx_gguf.get(), i);
                const size_t   offset      = gguf_get_tensor_offset(ctx_gguf.get(), i);
                enum ggml_type type        = gguf_get_tensor_type(ctx_gguf.get(), i);
                ggml_tensor *  cur         = ggml_get_tensor(meta, name);
                size_t         tensor_size = ggml_nbytes(cur);
                model_size += tensor_size;
                LOG_DBG("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64
                        ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                        __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1],
                        cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    void load_hparams() {
        // Vision model hparams - get a direct reference
        auto &      vision_hparams = ctx_clip.vision_model.hparams;
        std::string log_ffn_op_vision;  // For logging vision FFN type

        // Projector type - this seems global to the clip_ctx or determined by model architecture
        std::string proj_type_str;
        get_string(KEY_PROJ_TYPE, proj_type_str, false);  // Read into local string
        if (!proj_type_str.empty()) {
            ctx_clip.proj_type = clip_projector_type_from_string(proj_type_str);
        }
        // If proj_type_str is empty, ctx_clip.proj_type retains its default (e.g., PROJECTOR_TYPE_MLP or a more suitable default)
        if (ctx_clip.proj_type == PROJECTOR_TYPE_UNKNOWN) {
            // If a projector type was specified but unknown, it's an error.
            // If no projector type was specified, it might default to NONE or MLP depending on other GGUF fields.
            // For SigLIP, if "clip.projector_type" is "none" or absent, this is fine.
            // We might need to refine this logic if "none" is not explicitly in GGUF for SigLIP.
            // For now, if explicitly unknown, it's an error.
            if (!proj_type_str.empty()) {
                throw std::runtime_error(
                    string_format("%s: unknown projector type in GGUF: %s\n", __func__, proj_type_str.c_str()));
            }
            // Heuristic: if it's a siglip model based on other keys, assume PROJECTOR_TYPE_NONE
            // For now, let's assume it defaults to MLP or this will be caught by later checks
            // For pure SigLIP, we'll set it to PROJECTOR_TYPE_NONE if text_projection is found.
            // This part can be tricky; the conversion script should ideally always write a projector_type.
        }
        // For SigLIP models, explicitly set projector_type to NONE if it's not already set
        // and we find text tower specific tensors, implying a dual encoder.
        // This might need refinement based on how SigLIP GGUFs are structured.
        // A more robust way is if the GGUF explicitly states "none".
        if (proj_type_str.empty() || proj_type_str == "none") {  // Check if "none" is explicitly set or if it's empty
            // Attempt to find a text-specific tensor to infer if it's a dual encoder like SigLIP
            int test_text_tensor_idx = gguf_find_key(ctx_gguf.get(), "clip.text.block_count");  // A common text hparam
            if (test_text_tensor_idx != -1) {  // If text tower metadata exists
                bool has_text_proj_tensor = (gguf_find_key(ctx_gguf.get(), "text_projection.weight") != -1);
                if (has_text_proj_tensor ||
                    gguf_find_key(ctx_gguf.get(), "t.head.weight") != -1) {  // SigLIP has t.head.weight
                    LOG_INF(
                        "%s: Detected text tower specific keys. Setting projector_type to NONE for SigLIP-like "
                        "model.\n",
                        __func__);
                    ctx_clip.proj_type = PROJECTOR_TYPE_NONE;
                }
            }
        }

        // Load VISION hparams
        {
            get_i32(KEY_MINICPMV_VERSION, ctx_clip.minicpmv_version, false);  // This seems global

            get_u32(KEY_N_EMBD, vision_hparams.n_embd);
            get_u32(KEY_N_HEAD, vision_hparams.n_head);
            get_u32(KEY_N_FF, vision_hparams.n_ff);
            get_u32(KEY_N_BLOCK, vision_hparams.n_layer);
            get_u32(KEY_PROJ_DIM, vision_hparams.projection_dim);  // For vision tower's own projection if it has one
            get_f32(KEY_LAYER_NORM_EPS, vision_hparams.eps);
            get_u32(KEY_IMAGE_SIZE, vision_hparams.image_size);
            get_u32(KEY_PATCH_SIZE, vision_hparams.patch_size);
            get_u32(KEY_IMAGE_CROP_RESOLUTION, vision_hparams.image_crop_resolution, false);
            get_arr_int(KEY_IMAGE_GRID_PINPOINTS, vision_hparams.image_grid_pinpoints, false);

            vision_hparams.warmup_image_size = vision_hparams.image_size;

            ctx_clip.has_llava_projector =
                (ctx_clip.proj_type != PROJECTOR_TYPE_NONE &&
                 ctx_clip.proj_type != PROJECTOR_TYPE_GEMMA3 &&    // Gemma3 has its own specific projector path
                 ctx_clip.proj_type != PROJECTOR_TYPE_IDEFICS3 &&  // Idefics3 has its own specific projector path
                 ctx_clip.proj_type != PROJECTOR_TYPE_INTERNVL &&  // InternVL has its own specific projector path
                 ctx_clip.proj_type != PROJECTOR_TYPE_PIXTRAL &&   // Pixtral has its own specific projector path
                 ctx_clip.proj_type != PROJECTOR_TYPE_QWEN2VL &&   // Qwen2VL variants
                 ctx_clip.proj_type != PROJECTOR_TYPE_QWEN25VL);

            // Vision FFN type
            bool use_gelu_vision = false;
            bool use_silu_vision = false;
            get_bool(KEY_USE_GELU, use_gelu_vision, false);  // Assumes this key is for vision, or global
            get_bool(KEY_USE_SILU, use_silu_vision, false);  // Same assumption

            if (use_gelu_vision && use_silu_vision) {
                throw std::runtime_error(
                    string_format("%s: both use_gelu and use_silu are set for vision hparams\n", __func__));
            }
            if (use_gelu_vision) {
                vision_hparams.ffn_op = FFN_GELU;
                log_ffn_op_vision     = "gelu";
            } else if (use_silu_vision) {
                vision_hparams.ffn_op = FFN_SILU;
                log_ffn_op_vision     = "silu";
            } else {
                vision_hparams.ffn_op = FFN_GELU_QUICK;  // Default for many vision models
                log_ffn_op_vision     = "gelu_quick";
            }

            std::string mm_patch_merge_type_str;
            get_string(KEY_MM_PATCH_MERGE_TYPE, mm_patch_merge_type_str, false);
            if (mm_patch_merge_type_str == "spatial_unpad") {
                vision_hparams.mm_patch_merge_type = PATCH_MERGE_SPATIAL_UNPAD;
            } else {
                vision_hparams.mm_patch_merge_type = PATCH_MERGE_FLAT;
            }

            int idx_mean_vision = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_MEAN);
            int idx_std_vision  = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_STD);
            GGML_ASSERT(idx_mean_vision >= 0 && "Vision image_mean not found in GGUF");
            GGML_ASSERT(idx_std_vision >= 0 && "Vision image_std not found in GGUF");
            const float * mean_data_vision = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_mean_vision);
            const float * std_data_vision  = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_std_vision);
            for (int i = 0; i < 3; ++i) {
                ctx_clip.image_mean[i] = mean_data_vision[i];
                ctx_clip.image_std[i]  = std_data_vision[i];
            }

            std::vector<int> vision_feature_layer_vec;
            get_arr_int(KEY_FEATURE_LAYER, vision_feature_layer_vec, false);
            for (auto & layer_idx : vision_feature_layer_vec) {
                vision_hparams.vision_feature_layer.insert(layer_idx);
            }

            // Vision model-specific projector params or adjustments
            switch (ctx_clip.proj_type) {
                case PROJECTOR_TYPE_IDEFICS3:
                case PROJECTOR_TYPE_INTERNVL:
                case PROJECTOR_TYPE_GEMMA3:  // Gemma3 uses proj_scale_factor for its specific pooling
                    get_u32(KEY_PROJ_SCALE_FACTOR, vision_hparams.proj_scale_factor, false);
                    if (ctx_clip.proj_type == PROJECTOR_TYPE_GEMMA3 && vision_hparams.proj_scale_factor == 0) {
                        vision_hparams.proj_scale_factor = 4;  // Default for Gemma3 if not specified
                    }
                    break;
                case PROJECTOR_TYPE_PIXTRAL:
                    vision_hparams.rope_theta        = 10000.0f;
                    vision_hparams.warmup_image_size = vision_hparams.patch_size * 8;
                    get_u32(KEY_SPATIAL_MERGE_SIZE, vision_hparams.spatial_merge_size, false);
                    break;
                case PROJECTOR_TYPE_QWEN2VL:
                    vision_hparams.image_size        = 1024;
                    vision_hparams.warmup_image_size = vision_hparams.patch_size * 8;
                    break;
                case PROJECTOR_TYPE_QWEN25VL:
                    vision_hparams.image_size        = 1024;
                    vision_hparams.warmup_image_size = vision_hparams.patch_size * 8;
                    get_u32(KEY_WIN_ATTN_PATTERN, vision_hparams.n_wa_pattern);
                    break;
                case PROJECTOR_TYPE_NONE:  // No specific vision hparams adjustments for "none" projector type
                default:
                    break;
            }
        }

        // Load TEXT hparams
        // These will only be used if text tower tensors (like token_embeddings) are actually loaded later.
        auto & text_hparams         = ctx_clip.text_model.hparams;
        // Initialize to known defaults or sentinels before attempting to read from GGUF
        text_hparams.n_layer        = 0;
        text_hparams.n_embd         = 0;
        text_hparams.n_ff           = 0;
        text_hparams.n_head         = 0;
        text_hparams.projection_dim = 0;
        text_hparams.eps            = 1e-5f;           // A common default for text models
        text_hparams.ffn_op         = FFN_GELU_QUICK;  // A common default, adjust if GGUF has text-specific FFN key

        fprintf(stderr, "[LOAD_HPARAMS_TEXT_DEBUG_ENTRY] Initial text_model.hparams.n_layer: %d, n_embd: %d\n",
                text_hparams.n_layer, text_hparams.n_embd);

        int      key_idx = -1;
        uint32_t val_u32 = 0;

        key_idx = gguf_find_key(ctx_gguf.get(), "clip.text.block_count");
        if (key_idx >= 0) {
            val_u32              = gguf_get_val_u32(ctx_gguf.get(), key_idx);
            // if (val_u32 > 0) // Allow 0 if explicitly set in GGUF, though unusual
            text_hparams.n_layer = val_u32;
            fprintf(stderr, "[LOAD_HPARAMS_TEXT_DEBUG] GGUF 'clip.text.block_count': %u -> text_hparams.n_layer: %d\n",
                    val_u32, text_hparams.n_layer);
        } else {
            fprintf(
                stderr,
                "[LOAD_HPARAMS_TEXT_DEBUG] Key 'clip.text.block_count' not found. text_hparams.n_layer remains: %d\n",
                text_hparams.n_layer);
        }

        key_idx = gguf_find_key(ctx_gguf.get(), "clip.text.embedding_length");
        if (key_idx >= 0) {
            val_u32             = gguf_get_val_u32(ctx_gguf.get(), key_idx);
            text_hparams.n_embd = val_u32;
            fprintf(stderr,
                    "[LOAD_HPARAMS_TEXT_DEBUG] GGUF 'clip.text.embedding_length': %u -> text_hparams.n_embd: %d\n",
                    val_u32, text_hparams.n_embd);
        } else {
            fprintf(stderr,
                    "[LOAD_HPARAMS_TEXT_DEBUG] Key 'clip.text.embedding_length' not found. text_hparams.n_embd "
                    "remains: %d\n",
                    text_hparams.n_embd);
        }

        key_idx = gguf_find_key(ctx_gguf.get(), "clip.text.feed_forward_length");
        if (key_idx >= 0) {
            val_u32           = gguf_get_val_u32(ctx_gguf.get(), key_idx);
            text_hparams.n_ff = val_u32;
            fprintf(stderr,
                    "[LOAD_HPARAMS_TEXT_DEBUG] GGUF 'clip.text.feed_forward_length': %u -> text_hparams.n_ff: %d\n",
                    val_u32, text_hparams.n_ff);
        } else {
            fprintf(stderr,
                    "[LOAD_HPARAMS_TEXT_DEBUG] Key 'clip.text.feed_forward_length' not found. text_hparams.n_ff "
                    "remains: %d\n",
                    text_hparams.n_ff);
        }

        key_idx = gguf_find_key(ctx_gguf.get(), "clip.text.attention.head_count");
        if (key_idx >= 0) {
            val_u32             = gguf_get_val_u32(ctx_gguf.get(), key_idx);
            text_hparams.n_head = val_u32;
            fprintf(stderr,
                    "[LOAD_HPARAMS_TEXT_DEBUG] GGUF 'clip.text.attention.head_count': %u -> text_hparams.n_head: %d\n",
                    val_u32, text_hparams.n_head);
        } else {
            fprintf(stderr,
                    "[LOAD_HPARAMS_TEXT_DEBUG] Key 'clip.text.attention.head_count' not found. text_hparams.n_head "
                    "remains: %d\n",
                    text_hparams.n_head);
        }

        key_idx = gguf_find_key(ctx_gguf.get(), "clip.text.projection_dim");
        if (key_idx >= 0) {
            val_u32                     = gguf_get_val_u32(ctx_gguf.get(), key_idx);
            text_hparams.projection_dim = val_u32;
            fprintf(
                stderr,
                "[LOAD_HPARAMS_TEXT_DEBUG] GGUF 'clip.text.projection_dim': %u -> text_hparams.projection_dim: %d\n",
                val_u32, text_hparams.projection_dim);
        } else {
            fprintf(stderr,
                    "[LOAD_HPARAMS_TEXT_DEBUG] Key 'clip.text.projection_dim' not found. text_hparams.projection_dim "
                    "remains: %d\n",
                    text_hparams.projection_dim);
        }

        // Final Logging
        LOG_INF("%s: projector type:     %s\n", __func__,
                proj_type_str.empty() ? "default/inferred" : proj_type_str.c_str());
        LOG_INF("%s: vision_n_embd:      %d\n", __func__, vision_hparams.n_embd);
        LOG_INF("%s: vision_n_head:      %d\n", __func__, vision_hparams.n_head);
        LOG_INF("%s: vision_n_ff:        %d\n", __func__, vision_hparams.n_ff);
        LOG_INF("%s: vision_n_layer:       %d\n", __func__, vision_hparams.n_layer);
        LOG_INF("%s: vision_projection_dim:%d\n", __func__, vision_hparams.projection_dim);
        LOG_INF("%s: image_size:         %d\n", __func__, vision_hparams.image_size);
        LOG_INF("%s: patch_size:         %d\n", __func__, vision_hparams.patch_size);
        LOG_INF("\n");
        LOG_INF("%s: has_llava_proj:     %d\n", __func__, ctx_clip.has_llava_projector);
        LOG_INF("%s: minicpmv_version:   %d\n", __func__, ctx_clip.minicpmv_version);
        LOG_INF("%s: vision_proj_scale_factor: %d\n", __func__, vision_hparams.proj_scale_factor);
        LOG_INF("%s: vision_n_wa_pattern:  %d\n", __func__, vision_hparams.n_wa_pattern);
        LOG_INF("%s: vision_ffn_op:      %s\n", __func__, log_ffn_op_vision.c_str());

        if (text_hparams.n_layer > 0 && text_hparams.n_embd > 0) {
            LOG_INF("%s: text_n_embd:        %d\n", __func__, text_hparams.n_embd);
            LOG_INF("%s: text_n_head:        %d\n", __func__, text_hparams.n_head);
            LOG_INF("%s: text_n_ff:          %d\n", __func__, text_hparams.n_ff);
            LOG_INF("%s: text_n_layer:         %d\n", __func__, text_hparams.n_layer);
            LOG_INF("%s: text_projection_dim:  %d\n", __func__, text_hparams.projection_dim);
        } else {
            LOG_INF("%s: Text model hparams indicate no text tower or incomplete configuration.\n", __func__);
        }

        LOG_INF("%s: model size:         %.2f MiB\n", __func__, model_size / 1024.0 / 1024.0);
        LOG_INF("%s: metadata size:      %.2f MiB\n", __func__, ggml_get_mem_size(ctx_meta.get()) / 1024.0 / 1024.0);

        fprintf(stderr, "[LOAD_HPARAMS_DEBUG_EXIT] ctx_clip.vision_model.hparams.n_layer = %d\n",
                ctx_clip.vision_model.hparams.n_layer);
        fprintf(stderr, "[LOAD_HPARAMS_DEBUG_EXIT] ctx_clip.text_model.hparams.n_layer = %d\n",
                ctx_clip.text_model.hparams.n_layer);
        fprintf(stderr, "[LOAD_HPARAMS_DEBUG_EXIT] ctx_clip.text_model.hparams.n_embd = %d\n",
                ctx_clip.text_model.hparams.n_embd);
        fprintf(stderr, "[LOAD_HPARAMS_DEBUG_EXIT] ctx_clip.text_model.hparams.n_ff = %d\n",
                ctx_clip.text_model.hparams.n_ff);
        fprintf(stderr, "[LOAD_HPARAMS_DEBUG_EXIT] ctx_clip.text_model.hparams.n_head = %d\n",
                ctx_clip.text_model.hparams.n_head);
    }

    void load_tensors() {
        // auto & hparams = ctx_clip.vision_model.hparams; // Vision hparams, already used where needed
        std::map<std::string, size_t> tensor_offset;
        std::vector<ggml_tensor *>    tensors_to_load;

        gguf_context * gguf_ctx_ptr = this->ctx_gguf.get();
        GGML_ASSERT(gguf_ctx_ptr != nullptr && "Loader's GGUF context is null in load_tensors");

        fprintf(stderr,
                "[CLIP_LOAD_TENSORS_DEBUG_ENTRY] ctx_clip.text_model.hparams.n_layer = %d, "
                "vision_model.hparams.n_layer = %d\n",
                ctx_clip.text_model.hparams.n_layer, ctx_clip.vision_model.hparams.n_layer);
        fprintf(stderr,
                "[CLIP_LOAD_TENSORS_DEBUG_ENTRY] ctx_clip.text_model.hparams.n_embd = %d, vision_model.hparams.n_embd "
                "= %d\n",
                ctx_clip.text_model.hparams.n_embd, ctx_clip.vision_model.hparams.n_embd);

        const size_t unpadded_ggml_tensor_struct_size_for_data_ctx = sizeof(struct ggml_tensor);
        const size_t size_of_ggml_object_struct_for_data_ctx =
            ggml_tensor_overhead() - unpadded_ggml_tensor_struct_size_for_data_ctx;
        const size_t padded_ggml_tensor_struct_size_for_data_ctx =
            GGML_PAD(unpadded_ggml_tensor_struct_size_for_data_ctx, GGML_MEM_ALIGN);
        const size_t actual_metadata_footprint_per_tensor_for_data_ctx =
            size_of_ggml_object_struct_for_data_ctx + padded_ggml_tensor_struct_size_for_data_ctx;

        fprintf(stderr,
                "[CLIP_LOAD_TENSORS_DEBUG_ENTRY] ctx_clip.text_model.hparams.n_layer = %d, "
                "vision_model.hparams.n_layer = %d\n",
                ctx_clip.text_model.hparams.n_layer, ctx_clip.vision_model.hparams.n_layer);
        fprintf(stderr,
                "[CLIP_LOAD_TENSORS_DEBUG_ENTRY] ctx_clip.text_model.hparams.n_embd = %d, vision_model.hparams.n_embd "
                "= %d\n",
                ctx_clip.text_model.hparams.n_embd, ctx_clip.vision_model.hparams.n_embd);

        int64_t num_base_tensors_duplicated = gguf_get_n_tensors(gguf_ctx_ptr);

        // Count ggml_cont from assign_canonical_or_transposed_tensor for FFNs
        int64_t vision_ffn_ggml_cont_count = 0;
        if (ctx_clip.vision_model.hparams.n_layer > 0) {
            vision_ffn_ggml_cont_count = ctx_clip.vision_model.hparams.n_layer * 2;
        }

        int64_t text_ffn_ggml_cont_count = 0;
        // **REVISED CRUCIAL CHANGE HERE:**
        // Base the text FFN cont count ONLY on whether text hparams indicate a text tower.
        // token_embeddings will be loaded LATER.
        if (ctx_clip.text_model.hparams.n_layer > 0 && ctx_clip.text_model.hparams.n_embd > 0) {
            text_ffn_ggml_cont_count = ctx_clip.text_model.hparams.n_layer * 2;
            fprintf(stderr,
                    "[CLIP_LOAD_TENSORS_DEBUG] Text tower hparams indicate presence (n_layer=%d, n_embd=%d). "
                    "text_ffn_ggml_cont_count = %lld\n",
                    ctx_clip.text_model.hparams.n_layer, ctx_clip.text_model.hparams.n_embd,
                    (long long) text_ffn_ggml_cont_count);
        } else {
            fprintf(stderr,
                    "[CLIP_LOAD_TENSORS_DEBUG] Text tower hparams (n_layer or n_embd) are zero/default. "
                    "text_ffn_ggml_cont_count = 0.\n");
            text_ffn_ggml_cont_count = 0;  // Stays 0
        }

        int64_t safety_slot_for_mystery_object     = 1;
        int64_t explicit_cont_plus_transpose_count = 0;  // Assuming assign_canonical handles FFNs

        int64_t total_estimated_slots = num_base_tensors_duplicated + vision_ffn_ggml_cont_count +
                                        text_ffn_ggml_cont_count + (explicit_cont_plus_transpose_count * 2) +
                                        safety_slot_for_mystery_object;

        struct ggml_init_params params_data_ctx = {
            /*.mem_size =*/total_estimated_slots * actual_metadata_footprint_per_tensor_for_data_ctx,
            /*.mem_buffer =*/NULL,
            /*.no_alloc =*/true,
        };

        fprintf(stderr, "[CLIP_LOAD_TENSORS_DEBUG] num_base_tensors_duplicated = %lld\n",
                (long long) num_base_tensors_duplicated);
        fprintf(stderr, "[CLIP_LOAD_TENSORS_DEBUG] vision_ffn_ggml_cont_count (n_layer*2) = %lld (vision_n_layer=%d)\n",
                (long long) vision_ffn_ggml_cont_count, ctx_clip.vision_model.hparams.n_layer);
        fprintf(stderr,
                "[CLIP_LOAD_TENSORS_DEBUG] text_ffn_ggml_cont_count (based on hparams: n_layer*2) = %lld "
                "(text_n_layer=%d)\n",  // Log updated
                (long long) text_ffn_ggml_cont_count, ctx_clip.text_model.hparams.n_layer);
        fprintf(stderr, "[CLIP_LOAD_TENSORS_DEBUG] explicit_cont_plus_transpose_count = %lld\n",
                (long long) explicit_cont_plus_transpose_count);
        fprintf(stderr, "[CLIP_LOAD_TENSORS_DEBUG] safety_slot_for_mystery_object = %lld\n",
                (long long) safety_slot_for_mystery_object);
        fprintf(stderr, "[CLIP_LOAD_TENSORS_DEBUG] total_estimated_slots for ctx_data = %lld\n",
                (long long) total_estimated_slots);
        fprintf(stderr, "[CLIP_LOAD_TENSORS_DEBUG] actual_footprint_per_tensor for ctx_data = %zu\n",
                actual_metadata_footprint_per_tensor_for_data_ctx);
        fprintf(stderr, "[CLIP_LOAD_TENSORS_DEBUG] Calculated mem_size for ctx_data = %zu\n", params_data_ctx.mem_size);

        ctx_clip.ctx_data.reset(ggml_init(params_data_ctx));
        if (!ctx_clip.ctx_data) {
            throw std::runtime_error(
                string_format("%s: failed to init ggml context for tensor metadata (ctx_data). Size requested: %zu "
                              "bytes for %lld slots.",
                              __func__, params_data_ctx.mem_size, (long long) total_estimated_slots));
        }

        // Populate tensor_offset map
        for (int64_t i = 0; i < gguf_get_n_tensors(gguf_ctx_ptr); ++i) {
            const char * name   = gguf_get_tensor_name(gguf_ctx_ptr, i);
            tensor_offset[name] = gguf_get_data_offset(gguf_ctx_ptr) + gguf_get_tensor_offset(gguf_ctx_ptr, i);
        }

        // Helper function to get tensor metadata into ctx_data
        auto get_tensor = [&](const std::string & name, bool required = true) {
            ggml_tensor * cur_meta =
                ggml_get_tensor(ctx_meta.get(), name.c_str());  // Tensor metadata from GGUF parsing
            // Specifically inspect the problematic bias
            if (name == string_format(TN_FFN_UP, "v", 0, "bias")) {  // Check for the first layer's vision FFN up bias
                if (cur_meta) {
                    fprintf(stderr,
                            "[GGUF_INSPECT_DEBUG] Tensor '%s' (from GGUF metadata ctx_meta):\n"
                            "  Name: %s, Type: %s(%d)\n"
                            "  Shape (ne): [%lld, %lld, %lld, %lld]\n"
                            "  Strides (nb): [%zu, %zu, %zu, %zu]\n",
                            name.c_str(), cur_meta->name, ggml_type_name(cur_meta->type), cur_meta->type,
                            (long long) cur_meta->ne[0], (long long) cur_meta->ne[1], (long long) cur_meta->ne[2],
                            (long long) cur_meta->ne[3], cur_meta->nb[0], cur_meta->nb[1], cur_meta->nb[2],
                            cur_meta->nb[3]);
                } else {
                    fprintf(stderr, "[GGUF_INSPECT_DEBUG] Tensor '%s' NOT FOUND in GGUF metadata (ctx_meta).\n",
                            name.c_str());
                }
            }
            if (!cur_meta && required) {
                throw std::runtime_error(
                    string_format("%s: unable to find tensor %s in GGUF metadata\n", __func__, name.c_str()));
            }
            if (cur_meta) {
                // Duplicate the metadata into ctx_clip.ctx_data
                // The actual data pointer will be set later if not no_alloc, or remain null for no_alloc=true on ctx_data
                ggml_tensor * data_tensor = ggml_dup_tensor(ctx_clip.ctx_data.get(), cur_meta);
                if (!data_tensor) {
                    throw std::runtime_error(
                        string_format("%s: ggml_dup_tensor failed for %s into ctx_data\n", __func__, name.c_str()));
                }
                ggml_set_name(data_tensor, cur_meta->name);
                tensors_to_load.push_back(data_tensor);  // Keep track of tensors whose data needs to be mmaped/read
                return data_tensor;                      // Return the tensor that is part of ctx_clip.ctx_data
            }
            return (ggml_tensor *) nullptr;
        };

        // Vision model tensors
        auto & vision_model   = ctx_clip.vision_model;
        auto & vision_hparams = vision_model.hparams;  // Use a direct reference for vision_hparams

        vision_model.class_embedding     = get_tensor(TN_CLASS_EMBD, false);
        vision_model.pre_ln_w            = get_tensor(string_format(TN_LN_PRE, "v", "weight"), false);
        vision_model.pre_ln_b            = get_tensor(string_format(TN_LN_PRE, "v", "bias"), false);
        vision_model.post_ln_w           = get_tensor(string_format(TN_LN_POST, "v", "weight"), false);
        vision_model.post_ln_b           = get_tensor(string_format(TN_LN_POST, "v", "bias"), false);
        vision_model.patch_bias          = get_tensor(TN_PATCH_BIAS, false);
        vision_model.patch_embeddings_0  = get_tensor(TN_PATCH_EMBD, false);
        vision_model.patch_embeddings_1  = get_tensor(TN_PATCH_EMBD_1, false);
        vision_model.position_embeddings = get_tensor(string_format(TN_POS_EMBD, "v"), false);
        vision_model.projection =
            get_tensor("visual_projection.weight", false);  // Standard CLIP/SigLIP vision projection
        if (!vision_model.projection) {                     // Fallback for some naming conventions
            vision_model.projection = get_tensor("v.head.weight", false);
        }

        vision_model.layers.resize(vision_hparams.n_layer);
        for (int il = 0; il < vision_hparams.n_layer; ++il) {
            auto & layer = vision_model.layers[il];
            layer.k_w    = get_tensor(string_format(TN_ATTN_K, "v", il, "weight"));
            layer.q_w    = get_tensor(string_format(TN_ATTN_Q, "v", il, "weight"));
            layer.v_w    = get_tensor(string_format(TN_ATTN_V, "v", il, "weight"));
            layer.o_w    = get_tensor(string_format(TN_ATTN_OUTPUT, "v", il, "weight"));
            layer.k_norm = get_tensor(string_format(TN_ATTN_K_NORM, "v", il, "weight"), false);
            layer.q_norm = get_tensor(string_format(TN_ATTN_Q_NORM, "v", il, "weight"), false);
            layer.ln_1_w = get_tensor(string_format(TN_LN_1, "v", il, "weight"), false);
            layer.ln_2_w = get_tensor(string_format(TN_LN_2, "v", il, "weight"), false);
            layer.ls_1_w = get_tensor(string_format(TN_LS_1, "v", il, "weight"), false);
            layer.ls_2_w = get_tensor(string_format(TN_LS_2, "v", il, "weight"), false);
            layer.k_b    = get_tensor(string_format(TN_ATTN_K, "v", il, "bias"), false);
            layer.q_b    = get_tensor(string_format(TN_ATTN_Q, "v", il, "bias"), false);
            layer.v_b    = get_tensor(string_format(TN_ATTN_V, "v", il, "bias"), false);
            layer.o_b    = get_tensor(string_format(TN_ATTN_OUTPUT, "v", il, "bias"), false);
            layer.ln_1_b = get_tensor(string_format(TN_LN_1, "v", il, "bias"), false);
            layer.ln_2_b = get_tensor(string_format(TN_LN_2, "v", il, "bias"), false);

            layer.ff_up_w = assign_canonical_or_transposed_tensor(
                ctx_clip.ctx_data.get(), get_tensor(string_format(TN_FFN_UP, "v", il, "weight")), vision_hparams.n_embd,
                vision_hparams.n_ff);
            layer.ff_up_b = get_tensor(string_format(TN_FFN_UP, "v", il, "bias"), false);

            layer.ff_gate_w = get_tensor(string_format(TN_FFN_GATE, "v", il, "weight"), false);
            layer.ff_gate_b = get_tensor(string_format(TN_FFN_GATE, "v", il, "bias"), false);

            layer.ff_down_w = assign_canonical_or_transposed_tensor(
                ctx_clip.ctx_data.get(), get_tensor(string_format(TN_FFN_DOWN, "v", il, "weight")), vision_hparams.n_ff,
                vision_hparams.n_embd);
            layer.ff_down_b = get_tensor(string_format(TN_FFN_DOWN, "v", il, "bias"), false);

            // Fallback for FFN biases if primary vision ones not found (e.g. if text_model prefix was used in GGUF for shared weights)
            if (!layer.ff_up_b) {
                layer.ff_up_b = get_tensor(string_format(TN_FFN_UP, "text_model", il, "bias"), false);
            }
            if (!layer.ff_down_b) {
                layer.ff_down_b = get_tensor(string_format(TN_FFN_DOWN, "text_model", il, "bias"), false);
            }

            // Simple swap for some models that might have FFN weights transposed in GGUF.
            // This relies on ff_up_w being [N, M] and ff_down_w being [M, P] where M = n_ff
            // if ff_up_w.ne[1] (cols) == n_ff, it implies it's already [n_embd, n_ff] or was made so by assign_canonical.
            // if ff_down_w.ne[0] (rows) != n_ff, AND ff_up_w.ne[1] (cols of up) == ff_down_w.ne[1] (cols of down after potential transpose)
            // then they might be swapped. This is a bit heuristic.
            // The assign_canonical_or_transposed_tensor should ideally make this unnecessary.
            // For now, commenting out the explicit transpose blocks for FFNs after assign_canonical.
        }

        // Text model tensors
        auto & tmodel       = ctx_clip.text_model;
        auto & text_hparams = tmodel.hparams;  // Use direct reference

        // Load text token embeddings first, as it's a key indicator for text tower presence.
        tmodel.token_embeddings = get_tensor("t.token_embd.weight", false);
        if (!tmodel.token_embeddings) {
            tmodel.token_embeddings = get_tensor("text_model.embeddings.token_embedding.weight", false);
        }

        if (tmodel.token_embeddings) {
            LOG_INF("%s: Text tower token embeddings found. Loading full text tower.\n", __func__);
            // If token_embeddings are found, ensure n_layer was correctly picked up for sizing.
            // The sizing logic already uses text_hparams.n_layer > 0 && text_hparams.n_embd > 0.
            // If text_hparams.n_layer is 0 here but token_embeddings exist, there's an inconsistency.
            GGML_ASSERT(
                text_hparams.n_layer > 0 &&
                "Text token_embeddings found, but text_hparams.n_layer is 0. Check GGUF metadata or hparam loading.");
            GGML_ASSERT(text_hparams.n_embd > 0 && "Text token_embeddings found, but text_hparams.n_embd is 0.");

            tmodel.position_embeddings = get_tensor(string_format(TN_POS_EMBD, "t"), false);
            tmodel.projection          = get_tensor("text_projection.weight", false);
            if (!tmodel.projection) {  // Fallback for some naming conventions (e.g. "t.head.weight")
                tmodel.projection = get_tensor("t.head.weight", false);
            }

            tmodel.layers.resize(text_hparams.n_layer);
            for (int il = 0; il < text_hparams.n_layer; ++il) {
                auto & layer = tmodel.layers[il];
                layer.k_w    = get_tensor(string_format(TN_ATTN_K, "t", il, "weight"), false);
                layer.k_b    = get_tensor(string_format(TN_ATTN_K, "t", il, "bias"), false);
                layer.q_w    = get_tensor(string_format(TN_ATTN_Q, "t", il, "weight"), false);
                layer.q_b    = get_tensor(string_format(TN_ATTN_Q, "t", il, "bias"), false);
                layer.v_w    = get_tensor(string_format(TN_ATTN_V, "t", il, "weight"), false);
                layer.v_b    = get_tensor(string_format(TN_ATTN_V, "t", il, "bias"), false);
                layer.o_w    = get_tensor(string_format(TN_ATTN_OUTPUT, "t", il, "weight"), false);
                layer.o_b    = get_tensor(string_format(TN_ATTN_OUTPUT, "t", il, "bias"), false);
                layer.ln_1_w = get_tensor(string_format(TN_LN_1, "t", il, "weight"), false);
                layer.ln_1_b = get_tensor(string_format(TN_LN_1, "t", il, "bias"), false);

                layer.ff_up_w = assign_canonical_or_transposed_tensor(
                    ctx_clip.ctx_data.get(), get_tensor(string_format(TN_FFN_UP, "t", il, "weight"), false),
                    text_hparams.n_embd, text_hparams.n_ff);
                layer.ff_up_b = get_tensor(string_format(TN_FFN_UP, "t", il, "bias"), false);
                if (!layer.ff_up_b) {  // Fallback if primary name not found
                    layer.ff_up_b = get_tensor(string_format(TN_FFN_UP, "text_model", il, "bias"), false);
                }

                layer.ff_gate_w = get_tensor(string_format(TN_FFN_GATE, "t", il, "weight"), false);
                layer.ff_gate_b = get_tensor(string_format(TN_FFN_GATE, "t", il, "bias"), false);

                layer.ff_down_w = assign_canonical_or_transposed_tensor(
                    ctx_clip.ctx_data.get(), get_tensor(string_format(TN_FFN_DOWN, "t", il, "weight"), false),
                    text_hparams.n_ff, text_hparams.n_embd);
                layer.ff_down_b = get_tensor(string_format(TN_FFN_DOWN, "t", il, "bias"), false);
                if (!layer.ff_down_b) {  // Fallback
                    layer.ff_down_b = get_tensor(string_format(TN_FFN_DOWN, "text_model", il, "bias"), false);
                }

                // Fix for swapped FFN weights (common in some GGUF conversions)
                // This check assumes assign_canonical_or_transposed_tensor has already tried to put them in canonical form.
                // If ff_up_w is [M, N] and ff_down_w is [N, P], but M (output of up) != N (input of down),
                // and if ff_up_w's N matches ff_down_w's P (output of down), they might be swapped.
                // This is specific to the case where ff_up is e.g. [n_embd, n_ff] and ff_down is [n_ff, n_embd].
                // If ff_down.ne[0] (rows of down) is n_embd (expected output dim), it implies ff_down might be the "up" weight.
                if (layer.ff_up_w && layer.ff_down_w &&
                    layer.ff_down_w->ne[0] == (int64_t) text_hparams.n_embd &&  // If down's input dim is text_embd
                    layer.ff_up_w->ne[1] == (int64_t) text_hparams.n_ff) {      // and up's output dim is text_ff
                    // This condition means ff_down_w might actually be ff_up_w if shapes are [n_embd, n_ff] for ff_down_w
                    // and ff_up_w is [n_ff, n_embd] (after assign_canonical tried to make it [n_embd, n_ff])
                    // This specific check might need refinement based on how often weights are truly swapped vs. just needing transpose.
                    // For now, let's trust assign_canonical_or_transposed_tensor and remove the explicit swap here.
                    // The primary role of assign_canonical is to get the dimensions right.
                }

                layer.ln_2_w = get_tensor(string_format(TN_LN_2, "t", il, "weight"), false);
                layer.ln_2_b = get_tensor(string_format(TN_LN_2, "t", il, "bias"), false);
            }
            tmodel.post_ln_w = get_tensor(string_format(TN_LN_POST, "t", "weight"), false);
            tmodel.post_ln_b = get_tensor(string_format(TN_LN_POST, "t", "bias"), false);

            // Update n_embd and n_embd_out based on loaded tensors for text model
            tmodel.n_embd     = tmodel.token_embeddings->ne[0];
            tmodel.n_embd_out = tmodel.projection ? tmodel.projection->ne[1] : tmodel.n_embd;

            // Note: SigLIP text_projection is [out_dim, in_dim], so ne[0] is out_dim.
            // If t.head.weight is used as projection, it's often [vocab_size, n_embd], which is not what we want for embedding projection.
            // For SigLIP, the text_projection.weight (if present, like in original HF) is the actual projection matrix.
            // If it's a "logit_bias" or "t.head.weight" style output layer, then n_embd_out should remain n_embd.
            // This needs careful checking depending on which tensor is used as 'tmodel.projection'.
            // For now, assuming tmodel.projection, if loaded, is a [dim_out, dim_in] matrix.
            // If tmodel.projection is from t.head.weight, then n_embd_out should be tmodel.n_embd.
            if (tmodel.projection && strcmp(tmodel.projection->name, "text_projection.weight") == 0) {
                tmodel.n_embd_out =
                    tmodel.projection->ne[0];       // For text_projection.weight -> [out_features, in_features]
            } else if (tmodel.projection && strcmp(tmodel.projection->name, "t.head.weight") == 0) {
                tmodel.n_embd_out = tmodel.n_embd;  // If it's the classification head, output dim is n_embd
            }

            LOG_INF("%s: Text tower loaded. n_layer: %d, n_embd: %d, n_embd_out: %d\n", __func__, text_hparams.n_layer,
                    tmodel.n_embd, tmodel.n_embd_out);

        } else {
            LOG_INF("%s: No text tower token embeddings found. Text encoder will be disabled.\n", __func__);
            // Ensure hparams reflect no text tower if embeddings aren't loaded, to prevent issues in graph building
            text_hparams.n_layer = 0;
            text_hparams.n_embd  = 0;
        }

        // Projector layers (LLaVA, MobileVLM, etc.)
        // This section remains largely the same as your diff, as it's specific to vision-language projectors
        // and not directly related to the dual-encoder SigLIP text tower.
        switch (ctx_clip.proj_type) {
            case PROJECTOR_TYPE_MLP:
            case PROJECTOR_TYPE_MLP_NORM:
                {
                    vision_model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"), false);
                    vision_model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    vision_model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"), false);
                    vision_model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    vision_model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"), false);
                    vision_model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    vision_model.mm_3_w = get_tensor(string_format(TN_LLAVA_PROJ, 3, "weight"), false);
                    vision_model.mm_3_b = get_tensor(string_format(TN_LLAVA_PROJ, 3, "bias"), false);
                    vision_model.mm_4_w = get_tensor(string_format(TN_LLAVA_PROJ, 4, "weight"), false);
                    vision_model.mm_4_b = get_tensor(string_format(TN_LLAVA_PROJ, 4, "bias"), false);
                    if (vision_model.mm_3_w) {
                        ctx_clip.proj_type = PROJECTOR_TYPE_MLP_NORM;
                    }
                    vision_model.image_newline = get_tensor(TN_IMAGE_NEWLINE, false);
                }
                break;
            // ... (all other projector types from your diff) ...
            case PROJECTOR_TYPE_NONE:  // Already handled by not loading any projector tensors.
                LOG_INF("%s: Projector type is NONE. No additional projector tensors loaded.\n", __func__);
                break;
            default:
                if (ctx_clip.proj_type != PROJECTOR_TYPE_UNKNOWN && ctx_clip.proj_type != PROJECTOR_TYPE_NONE) {
                    // This case should ideally not be hit if all known projector types are handled.
                    // If it's a new type, it might need specific tensor loading.
                    fprintf(stderr,
                            "%s: Warning: Projector type %d does not have specific tensor loading logic here.\n",
                            __func__, (int) ctx_clip.proj_type);
                }
                break;
        }

        // Load tensor data from file
        {
            std::vector<uint8_t> read_buf;
            std::ifstream        fin(fname, std::ios::binary);
            if (!fin) {
                throw std::runtime_error(string_format("%s: failed to open %s\n", __func__, fname.c_str()));
            }

            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(ctx_clip.backend);
            ctx_clip.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_clip.ctx_data.get(), buft));
            ggml_backend_buffer_set_usage(ctx_clip.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            for (ggml_tensor * cur_tensor_in_data_ctx : tensors_to_load) {  // Iterate over tensors already in ctx_data
                if (!cur_tensor_in_data_ctx) {
                    continue;  // Should not happen if get_tensor threw on required
                }

                const auto it = tensor_offset.find(cur_tensor_in_data_ctx->name);
                if (it == tensor_offset.end()) {
                    throw std::runtime_error(
                        string_format("%s: offset not found for tensor %s\n", __func__, cur_tensor_in_data_ctx->name));
                }
                const size_t offset = it->second;

                fin.seekg(offset, std::ios::beg);
                if (!fin) {
                    throw std::runtime_error(
                        string_format("%s: failed to seek for tensor %s\n", __func__, cur_tensor_in_data_ctx->name));
                }

                size_t num_bytes = ggml_nbytes(cur_tensor_in_data_ctx);
                if (ggml_backend_buft_is_host(buft)) {
                    fin.read(reinterpret_cast<char *>(cur_tensor_in_data_ctx->data), num_bytes);
                } else {
                    read_buf.resize(num_bytes);
                    fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                    ggml_backend_tensor_set(cur_tensor_in_data_ctx, read_buf.data(), 0, num_bytes);
                }
            }
            fin.close();
            LOG_DBG("%s: loaded data for %zu tensors from %s\n", __func__, tensors_to_load.size(), fname.c_str());
        }
    }  // end of load_tensors

    void alloc_compute_meta() {
        ctx_clip.buf_compute_meta.resize(ctx_clip.max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());

        // create a fake batch
        clip_image_f32_batch batch;
        clip_image_f32_ptr   img(clip_image_f32_init());
        img->nx = ctx_clip.vision_model.hparams.warmup_image_size;
        img->ny = ctx_clip.vision_model.hparams.warmup_image_size;
        img->buf.resize(img->nx * img->ny * 3);
        batch.entries.push_back(std::move(img));

        ggml_cgraph * gf = clip_image_build_graph(&ctx_clip, batch);
        ggml_backend_sched_reserve(ctx_clip.sched.get(), gf);

        for (size_t i = 0; i < ctx_clip.backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = ctx_clip.backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = ctx_clip.backend_buft[i];
            size_t                     size    = ggml_backend_sched_get_buffer_size(ctx_clip.sched.get(), backend);
            if (size > 1) {
                LOG_INF("%s: %10s compute buffer size = %8.2f MiB\n", __func__, ggml_backend_buft_name(buft),
                        size / 1024.0 / 1024.0);
            }
        }
    }

    void get_bool(const std::string & key, bool & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_bool(ctx_gguf.get(), i);
    }

    void get_i32(const std::string & key, int & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_i32(ctx_gguf.get(), i);
    }

    void get_u32(const std::string & key, int & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_u32(ctx_gguf.get(), i);
    }

    void get_f32(const std::string & key, float & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_f32(ctx_gguf.get(), i);
    }

    void get_string(const std::string & key, std::string & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = std::string(gguf_get_val_str(ctx_gguf.get(), i));
    }

    void get_arr_int(const std::string & key, std::vector<int> & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        int n = gguf_get_arr_n(ctx_gguf.get(), i);
        output.resize(n);
        const int32_t * values = (const int32_t *) gguf_get_arr_data(ctx_gguf.get(), i);
        for (int i = 0; i < n; ++i) {
            output[i] = values[i];
        }
    }
};

struct clip_ctx * clip_init(const char * fname, struct clip_context_params ctx_params) {
    g_logger_state.verbosity_thold = ctx_params.verbosity;
    clip_ctx * ctx_clip            = nullptr;

    try {
        ctx_clip = new clip_ctx(ctx_params);

        fprintf(stderr, "[CLIP_INIT_DEBUG_ADDR] Address of ctx_clip_ptr->text_model.hparams: %p\n",
                (void *) &(ctx_clip->text_model.hparams));
        fprintf(stderr, "[CLIP_INIT_DEBUG_ADDR] Address of ctx_clip_ptr->text_model: %p\n",
                (void *) &(ctx_clip->text_model));
        fprintf(stderr, "[CLIP_INIT_DEBUG_ADDR] Address of *ctx_clip_ptr: %p\n", (void *) ctx_clip);
        clip_model_loader loader(fname, *ctx_clip);
        loader.load_hparams();

        fprintf(stderr, "[CLIP_INIT_DEBUG] After load_hparams, ctx_clip_ptr->text_model.hparams.n_layer = %d\n",
                ctx_clip->text_model.hparams.n_layer);
        fprintf(stderr, "[CLIP_INIT_DEBUG] After load_hparams, ctx_clip_ptr->text_model.hparams.n_embd = %d\n",
                ctx_clip->text_model.hparams.n_embd);

        loader.load_tensors();
        // verify vision model loaded
        if (ctx_clip->vision_model.layers.empty()) {
            throw std::runtime_error(std::string("no vision_model layers loaded; '") + fname +
                                     "' missing vision encoder weights");
        }
        loader.alloc_compute_meta();

        // expose GGUF meta to outer context for tokenizer usage
        ctx_clip->ctx_gguf = std::move(loader.ctx_gguf);
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to load model '%s': %s\n", __func__, fname, e.what());
        delete ctx_clip;
        return nullptr;
    }

    return ctx_clip;
}

void clip_add_load_image_size(struct clip_ctx * ctx_clip, struct clip_image_size * load_image_size) {
    ctx_clip->load_image_size = *load_image_size;  // copy
}

struct clip_image_size * clip_get_load_image_size(struct clip_ctx * ctx_clip) {
    return &ctx_clip->load_image_size;
}

struct clip_image_size * clip_image_size_init() {
    struct clip_image_size * load_image_size = new struct clip_image_size();
    load_image_size->width                   = 448;
    load_image_size->height                  = 448;
    return load_image_size;
}

struct clip_image_u8 * clip_image_u8_init() {
    return new clip_image_u8();
}

struct clip_image_f32 * clip_image_f32_init() {
    return new clip_image_f32();
}

struct clip_image_f32_batch * clip_image_f32_batch_init() {
    return new clip_image_f32_batch();
}

unsigned char * clip_image_u8_get_data(struct clip_image_u8 * img, uint32_t * nx, uint32_t * ny) {
    if (nx) {
        *nx = img->nx;
    }
    if (ny) {
        *ny = img->ny;
    }
    return img->buf.data();
}

void clip_image_size_free(struct clip_image_size * load_image_size) {
    if (load_image_size == nullptr) {
        return;
    }
    delete load_image_size;
}

void clip_image_u8_free(struct clip_image_u8 * img) {
    if (img) {
        delete img;
    }
}

void clip_image_f32_free(struct clip_image_f32 * img) {
    if (img) {
        delete img;
    }
}

void clip_image_u8_batch_free(struct clip_image_u8_batch * batch) {
    if (batch) {
        delete batch;
    }
}

void clip_image_f32_batch_free(struct clip_image_f32_batch * batch) {
    if (batch) {
        delete batch;
    }
}

size_t clip_image_f32_batch_n_images(const struct clip_image_f32_batch * batch) {
    return batch->entries.size();
}

size_t clip_image_f32_batch_nx(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int) batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return 0;
    }
    return batch->entries[idx]->nx;
}

size_t clip_image_f32_batch_ny(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int) batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return 0;
    }
    return batch->entries[idx]->ny;
}

clip_image_f32 * clip_image_f32_get_img(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int) batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return nullptr;
    }
    return batch->entries[idx].get();
}

void clip_build_img_from_pixels(const unsigned char * rgb_pixels, int nx, int ny, clip_image_u8 * img) {
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    memcpy(img->buf.data(), rgb_pixels, img->buf.size());
}

bool clip_image_load_from_file(const char * fname, clip_image_u8 * img) {
    int    nx, ny, nc;
    auto * data = stbi_load(fname, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_ERR("%s: failed to load image '%s'\n", __func__, fname);
        return false;
    }
    clip_build_img_from_pixels(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img) {
    int    nx, ny, nc;
    auto * data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_ERR("%s: failed to decode image bytes\n", __func__);
        return false;
    }
    clip_build_img_from_pixels(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

// Normalize image to float32 - careful with pytorch .to(model.device, dtype=torch.float16) - this sometimes reduces precision (32>16>32), sometimes not
static void normalize_image_u8_to_f32(const clip_image_u8 & src, clip_image_f32 & dst, const float mean[3],
                                      const float std[3]) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(src.buf.size());

    // TODO @ngxson : seems like this could be done more efficiently on cgraph
    for (size_t i = 0; i < src.buf.size(); ++i) {
        int c      = i % 3;  // rgb
        dst.buf[i] = (static_cast<float>(src.buf[i]) / 255.0f - mean[c]) / std[c];
    }
}

// set of tools to manupulate images
// in the future, we can have HW acceleration by allowing this struct to access 3rd party lib like imagick or opencv
struct image_manipulation {
    // Bilinear resize function
    static void bilinear_resize(const clip_image_u8 & src, clip_image_u8 & dst, int target_width, int target_height) {
        dst.nx = target_width;
        dst.ny = target_height;
        dst.buf.resize(3 * target_width * target_height);

        float x_ratio = static_cast<float>(src.nx - 1) / target_width;
        float y_ratio = static_cast<float>(src.ny - 1) / target_height;

        for (int y = 0; y < target_height; y++) {
            for (int x = 0; x < target_width; x++) {
                float px      = x_ratio * x;
                float py      = y_ratio * y;
                int   x_floor = static_cast<int>(px);
                int   y_floor = static_cast<int>(py);
                float x_lerp  = px - x_floor;
                float y_lerp  = py - y_floor;

                for (int c = 0; c < 3; c++) {
                    float top = lerp(static_cast<float>(src.buf[3 * (y_floor * src.nx + x_floor) + c]),
                                     static_cast<float>(src.buf[3 * (y_floor * src.nx + (x_floor + 1)) + c]), x_lerp);
                    float bottom =
                        lerp(static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + x_floor) + c]),
                             static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + (x_floor + 1)) + c]), x_lerp);
                    dst.buf[3 * (y * target_width + x) + c] = static_cast<uint8_t>(lerp(top, bottom, y_lerp));
                }
            }
        }
    }

    // Bicubic resize function
    // part of image will be cropped if the aspect ratio is different
    static bool bicubic_resize(const clip_image_u8 & img, clip_image_u8 & dst, int target_width, int target_height) {
        const int nx = img.nx;
        const int ny = img.ny;

        dst.nx = target_width;
        dst.ny = target_height;
        dst.buf.resize(3 * target_width * target_height);

        float Cc;
        float C[5];
        float d0, d2, d3, a0, a1, a2, a3;
        int   i, j, k, jj;
        int   x, y;
        float dx, dy;
        float tx, ty;

        tx = (float) nx / (float) target_width;
        ty = (float) ny / (float) target_height;

        // Bicubic interpolation; adapted from ViT.cpp, inspired from :
        //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
        //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

        for (i = 0; i < target_height; i++) {
            for (j = 0; j < target_width; j++) {
                x = (int) (tx * j);
                y = (int) (ty * i);

                dx = tx * j - x;
                dy = ty * i - y;

                for (k = 0; k < 3; k++) {
                    for (jj = 0; jj <= 3; jj++) {
                        d0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] -
                             img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        d2 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] -
                             img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        d3 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] -
                             img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        a0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                        a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
                        a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;

                        C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                        d0 = C[0] - C[1];
                        d2 = C[2] - C[1];
                        d3 = C[3] - C[1];
                        a0 = C[1];
                        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                        a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
                        a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
                        Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                        const uint8_t Cc2                       = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                        dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                    }
                }
            }
        }

        return true;
    }

    // llava-1.6 type of resize_and_pad
    // if the ratio is not 1:1, padding with pad_color will be applied
    // pad_color is single channel, default is 0 (black)
    static void resize_and_pad_image(const clip_image_u8 & image, clip_image_u8 & dst,
                                     const clip_image_size & target_resolution,
                                     std::array<uint8_t, 3>  pad_color = { 0, 0, 0 }) {
        int target_width  = target_resolution.width;
        int target_height = target_resolution.height;

        float scale_w = static_cast<float>(target_width) / image.nx;
        float scale_h = static_cast<float>(target_height) / image.ny;

        int new_width, new_height;

        if (scale_w < scale_h) {
            new_width  = target_width;
            new_height = std::min(static_cast<int>(std::ceil(image.ny * scale_w)), target_height);
        } else {
            new_height = target_height;
            new_width  = std::min(static_cast<int>(std::ceil(image.nx * scale_h)), target_width);
        }

        clip_image_u8 resized_image;
        bicubic_resize(image, resized_image, new_width, new_height);

        clip_image_u8 padded_image;
        padded_image.nx = target_width;
        padded_image.ny = target_height;
        padded_image.buf.resize(3 * target_width * target_height);

        // Fill the padded image with the fill color
        for (size_t i = 0; i < padded_image.buf.size(); i += 3) {
            padded_image.buf[i]     = pad_color[0];
            padded_image.buf[i + 1] = pad_color[1];
            padded_image.buf[i + 2] = pad_color[2];
        }

        // Calculate padding offsets
        int pad_x = (target_width - new_width) / 2;
        int pad_y = (target_height - new_height) / 2;

        // Copy the resized image into the center of the padded buffer
        for (int y = 0; y < new_height; ++y) {
            for (int x = 0; x < new_width; ++x) {
                for (int c = 0; c < 3; ++c) {
                    padded_image.buf[3 * ((y + pad_y) * target_width + (x + pad_x)) + c] =
                        resized_image.buf[3 * (y * new_width + x) + c];
                }
            }
        }
        dst = std::move(padded_image);
    }

    static void crop_image(const clip_image_u8 & image, clip_image_u8 & dst, int x, int y, int w, int h) {
        dst.nx = w;
        dst.ny = h;
        dst.buf.resize(3 * w * h);

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int src_idx          = 3 * ((y + i) * image.nx + (x + j));
                int dst_idx          = 3 * (i * w + j);
                dst.buf[dst_idx]     = image.buf[src_idx];
                dst.buf[dst_idx + 1] = image.buf[src_idx + 1];
                dst.buf[dst_idx + 2] = image.buf[src_idx + 2];
            }
        }
    }

    // calculate the size of the **resized** image, while preserving the aspect ratio
    // the calculated size will be aligned to the nearest multiple of align_size
    // if H or W size is larger than max_dimension, it will be resized to max_dimension
    static clip_image_size calc_size_preserved_ratio(const clip_image_size & inp_size, const int align_size,
                                                     const int max_dimension) {
        if (inp_size.width <= 0 || inp_size.height <= 0 || align_size <= 0 || max_dimension <= 0) {
            return { 0, 0 };
        }

        float scale = std::min(1.0f, std::min(static_cast<float>(max_dimension) / inp_size.width,
                                              static_cast<float>(max_dimension) / inp_size.height));

        float target_width_f  = static_cast<float>(inp_size.width) * scale;
        float target_height_f = static_cast<float>(inp_size.height) * scale;

        int aligned_width  = CLIP_ALIGN((int) target_width_f, align_size);
        int aligned_height = CLIP_ALIGN((int) target_height_f, align_size);

        return { aligned_width, aligned_height };
    }

  private:
    static inline int clip(int x, int lower, int upper) { return std::max(lower, std::min(x, upper)); }

    // Linear interpolation between two points
    static inline float lerp(float s, float e, float t) { return s + (e - s) * t; }
};

/**
 * implementation of LLaVA-UHD:
 *  - https://arxiv.org/pdf/2403.11703
 *  - https://github.com/thunlp/LLaVA-UHD
 *  - https://github.com/thunlp/LLaVA-UHD/blob/302301bc2175f7e717fb8548516188e89f649753/llava_uhd/train/llava-uhd/slice_logic.py#L118
 *
 * overview:
 *   - an image always have a single overview (downscaled image)
 *   - an image can have 0 or multiple slices, depending on the image size
 *   - each slice can then be considered as a separate image
 *
 * for example:
 *
 * [overview] --> [slice 1] --> [slice 2]
 *           |                |
 *           +--> [slice 3] --> [slice 4]
 */
struct llava_uhd {
    struct slice_coordinates {
        int             x;
        int             y;
        clip_image_size size;
    };

    struct slice_instructions {
        clip_image_size overview_size;  // size of downscaled image
        clip_image_size refined_size;   // size of image right before slicing (must be multiple of slice size)
        clip_image_size grid_size;      // grid_size.width * grid_size.height = number of slices
        std::vector<slice_coordinates> slices;
        bool padding_refined = false;   // if true, refine image will be padded to the grid size (e.g. llava-1.6)
    };

    static int get_max_slices(struct clip_ctx * ctx) {
        if (clip_is_minicpmv(ctx)) {
            return 9;
        }
        return 0;
    }

    static slice_instructions get_slice_instructions(struct clip_ctx * ctx, const clip_image_size & original_size) {
        slice_instructions res;
        const int          patch_size      = clip_get_patch_size(ctx);
        const int          slice_size      = clip_get_image_size(ctx);
        const int          max_slice_nums  = get_max_slices(ctx);
        const int          original_width  = original_size.width;
        const int          original_height = original_size.height;
        const float        log_ratio       = log((float) original_width / original_height);
        const float        ratio           = (float) original_width * original_height / (slice_size * slice_size);
        const int          multiple        = fmin(ceil(ratio), max_slice_nums);
        const bool         has_slices      = (multiple > 1);
        const bool         has_pinpoints   = !ctx->vision_model.hparams.image_grid_pinpoints.empty();

        if (has_pinpoints) {
            // has pinpoints, use them to calculate the grid size (e.g. llava-1.6)
            auto refine_size =
                llava_uhd::select_best_resolution(ctx->vision_model.hparams.image_grid_pinpoints, original_size);
            res.overview_size   = clip_image_size{ slice_size, slice_size };
            res.refined_size    = refine_size;
            res.grid_size       = clip_image_size{ 0, 0 };
            res.padding_refined = true;

            for (int y = 0; y < refine_size.height; y += slice_size) {
                for (int x = 0; x < refine_size.width; x += slice_size) {
                    slice_coordinates slice;
                    slice.x           = x;
                    slice.y           = y;
                    slice.size.width  = std::min(slice_size, refine_size.width - x);
                    slice.size.height = std::min(slice_size, refine_size.height - y);
                    res.slices.push_back(slice);
                    if (x == 0) {
                        res.grid_size.width++;
                    }
                }
                res.grid_size.height++;
            }

            return res;
        }

        // no pinpoints, dynamically calculate the grid size (e.g. minicpmv)

        auto best_size    = get_best_resize(original_size, slice_size, patch_size, !has_slices);
        res.overview_size = best_size;

        if (!has_slices) {
            // skip slicing logic
            res.refined_size = clip_image_size{ 0, 0 };
            res.grid_size    = clip_image_size{ 0, 0 };

        } else {
            auto best_grid   = get_best_grid(max_slice_nums, multiple, log_ratio);
            auto refine_size = get_refine_size(original_size, best_grid, slice_size, patch_size, true);
            res.grid_size    = best_grid;
            res.refined_size = refine_size;

            int width  = refine_size.width;
            int height = refine_size.height;
            int grid_x = int(width / best_grid.width);
            int grid_y = int(height / best_grid.height);
            for (int patches_y = 0, ic = 0; patches_y < refine_size.height && ic < best_grid.height;
                 patches_y += grid_y, ic += 1) {
                for (int patches_x = 0, jc = 0; patches_x < refine_size.width && jc < best_grid.width;
                     patches_x += grid_x, jc += 1) {
                    slice_coordinates slice;
                    slice.x           = patches_x;
                    slice.y           = patches_y;
                    slice.size.width  = grid_x;
                    slice.size.height = grid_y;
                    res.slices.push_back(slice);
                    // LOG_INF("slice %d: %d %d %d %d\n", ic, patches_i, patches_j, grid_x, grid_y);
                }
            }
        }

        return res;
    }

    static std::vector<clip_image_u8_ptr> slice_image(const clip_image_u8 * img, const slice_instructions & inst) {
        std::vector<clip_image_u8_ptr> output;

        // resize to overview size
        clip_image_u8_ptr resized_img(clip_image_u8_init());
        image_manipulation::bicubic_resize(*img, *resized_img, inst.overview_size.width, inst.overview_size.height);
        output.push_back(std::move(resized_img));
        if (inst.slices.empty()) {
            // no slices, just return the resized image
            return output;
        }

        // resize to refined size
        clip_image_u8_ptr refined_img(clip_image_u8_init());
        if (inst.padding_refined) {
            image_manipulation::resize_and_pad_image(*img, *refined_img, inst.refined_size);
        } else {
            image_manipulation::bilinear_resize(*img, *refined_img, inst.refined_size.width, inst.refined_size.height);
        }

        // create slices
        for (const auto & slice : inst.slices) {
            int x = slice.x;
            int y = slice.y;
            int w = slice.size.width;
            int h = slice.size.height;

            clip_image_u8_ptr img_slice(clip_image_u8_init());
            image_manipulation::crop_image(*refined_img, *img_slice, x, y, w, h);
            output.push_back(std::move(img_slice));
        }

        return output;
    }

  private:
    static clip_image_size get_best_resize(const clip_image_size & original_size, int scale_resolution, int patch_size,
                                           bool allow_upscale = false) {
        int width  = original_size.width;
        int height = original_size.height;
        if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
            float r = static_cast<float>(width) / height;
            height  = static_cast<int>(scale_resolution / std::sqrt(r));
            width   = static_cast<int>(height * r);
        }
        clip_image_size res;
        res.width  = ensure_divide(width, patch_size);
        res.height = ensure_divide(height, patch_size);
        return res;
    }

    /**
     * Selects the best resolution from a list of possible resolutions based on the original size.
     *
     * @param original_size The original size of the image
     * @param possible_resolutions A list of possible resolutions
     * @return The best fit resolution
     */
    static clip_image_size select_best_resolution(const clip_image_size &              original_size,
                                                  const std::vector<clip_image_size> & possible_resolutions) {
        int             original_width  = original_size.width;
        int             original_height = original_size.height;
        clip_image_size best_fit;
        int             max_effective_resolution = 0;
        int             min_wasted_resolution    = std::numeric_limits<int>::max();

        for (const auto & resolution : possible_resolutions) {
            int   width  = resolution.width;
            int   height = resolution.height;
            float scale =
                std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
            int downscaled_width     = static_cast<int>(original_width * scale);
            int downscaled_height    = static_cast<int>(original_height * scale);
            int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
            int wasted_resolution    = (width * height) - effective_resolution;
            // LOG_INF("resolution: %d %d, scale: %f, downscaled: %d %d, effective: %d, wasted: %d\n", width, height, scale, downscaled_width, downscaled_height, effective_resolution, wasted_resolution);
            if (effective_resolution > max_effective_resolution ||
                (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution)) {
                max_effective_resolution = effective_resolution;
                min_wasted_resolution    = wasted_resolution;
                best_fit                 = resolution;
            }
        }

        return best_fit;
    }

    // used by llava 1.6 with custom list of pinpoints
    static clip_image_size select_best_resolution(const std::vector<int32_t> & pinpoints,
                                                  const clip_image_size &      original_size) {
        std::vector<clip_image_size> possible_resolutions;
        for (size_t i = 0; i < pinpoints.size(); i += 2) {
            possible_resolutions.push_back(clip_image_size{ pinpoints[i], pinpoints[i + 1] });
        }
        return select_best_resolution(original_size, possible_resolutions);
    }

    static int ensure_divide(int length, int patch_size) {
        return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
    }

    static clip_image_size get_refine_size(const clip_image_size & original_size, const clip_image_size & grid,
                                           int scale_resolution, int patch_size, bool allow_upscale = false) {
        int width  = original_size.width;
        int height = original_size.height;
        int grid_x = grid.width;
        int grid_y = grid.height;

        int refine_width  = ensure_divide(width, grid_x);
        int refine_height = ensure_divide(height, grid_y);

        clip_image_size grid_size;
        grid_size.width  = refine_width / grid_x;
        grid_size.height = refine_height / grid_y;

        auto best_grid_size   = get_best_resize(grid_size, scale_resolution, patch_size, allow_upscale);
        int  best_grid_width  = best_grid_size.width;
        int  best_grid_height = best_grid_size.height;

        clip_image_size refine_size;
        refine_size.width  = best_grid_width * grid_x;
        refine_size.height = best_grid_height * grid_y;
        return refine_size;
    }

    static clip_image_size get_best_grid(const int max_slice_nums, const int multiple, const float log_ratio) {
        std::vector<int> candidate_split_grids_nums;
        for (int i : { multiple - 1, multiple, multiple + 1 }) {
            if (i == 1 || i > max_slice_nums) {
                continue;
            }
            candidate_split_grids_nums.push_back(i);
        }

        std::vector<clip_image_size> candidate_grids;
        for (int split_grids_nums : candidate_split_grids_nums) {
            int m = 1;
            while (m <= split_grids_nums) {
                if (split_grids_nums % m == 0) {
                    candidate_grids.push_back(clip_image_size{ m, split_grids_nums / m });
                }
                ++m;
            }
        }

        clip_image_size best_grid{ 1, 1 };
        float           min_error = std::numeric_limits<float>::infinity();
        for (const auto & grid : candidate_grids) {
            float error = std::abs(log_ratio - std::log(1.0 * grid.width / grid.height));
            if (error < min_error) {
                best_grid = grid;
                min_error = error;
            }
        }
        return best_grid;
    }
};

// TODO @ngxson : decprecate the load_image_size singleton pattern
int clip_uhd_num_image_embeds_col(struct clip_ctx * ctx_clip) {
    const auto inst = llava_uhd::get_slice_instructions(ctx_clip, ctx_clip->load_image_size);
    return inst.grid_size.width;
}

// returns the normalized float tensor for llava-1.5, for spatial_unpad with anyres processing for llava-1.6 it returns the normalized image patch tensors as a vector
// res_imgs memory is being allocated here, previous allocations will be freed if found
bool clip_image_preprocess(struct clip_ctx * ctx, const clip_image_u8 * img, struct clip_image_f32_batch * res_imgs) {
    clip_image_size original_size{ img->nx, img->ny };
    bool            pad_to_square = true;
    auto &          params        = ctx->vision_model.hparams;
    // The model config actually contains all we need to decide on how to preprocess, here we automatically switch to the new llava-1.6 preprocessing
    if (params.mm_patch_merge_type == PATCH_MERGE_SPATIAL_UNPAD) {
        pad_to_square = false;
    }

    if (clip_is_minicpmv(ctx)) {
        const auto                     inst = llava_uhd::get_slice_instructions(ctx, original_size);
        std::vector<clip_image_u8_ptr> imgs = llava_uhd::slice_image(img, inst);

        for (size_t i = 0; i < imgs.size(); ++i) {
            // clip_image_save_to_bmp(*imgs[i], "slice_" + std::to_string(i) + ".bmp");
            clip_image_f32_ptr res(clip_image_f32_init());
            normalize_image_u8_to_f32(*imgs[i], *res, ctx->image_mean, ctx->image_std);
            res_imgs->entries.push_back(std::move(res));
        }
        return true;
    } else if (ctx->proj_type == PROJECTOR_TYPE_QWEN2VL || ctx->proj_type == PROJECTOR_TYPE_QWEN25VL) {
        clip_image_u8 resized;
        auto          patch_size = params.patch_size * 2;
        auto new_size = image_manipulation::calc_size_preserved_ratio(original_size, patch_size, params.image_size);
        image_manipulation::bicubic_resize(*img, resized, new_size.width, new_size.height);

        clip_image_f32_ptr img_f32(clip_image_f32_init());
        // clip_image_f32_ptr res(clip_image_f32_init());
        normalize_image_u8_to_f32(resized, *img_f32, ctx->image_mean, ctx->image_std);
        // res_imgs->data[0] = *res;
        res_imgs->entries.push_back(std::move(img_f32));
        return true;
    } else if (ctx->proj_type == PROJECTOR_TYPE_GLM_EDGE || ctx->proj_type == PROJECTOR_TYPE_GEMMA3 ||
               ctx->proj_type == PROJECTOR_TYPE_IDEFICS3 ||
               ctx->proj_type == PROJECTOR_TYPE_INTERNVL  // TODO @ngxson : support dynamic resolution
    ) {
        clip_image_u8 resized_image;
        int           sz = params.image_size;
        image_manipulation::resize_and_pad_image(*img, resized_image, { sz, sz });
        clip_image_f32_ptr img_f32(clip_image_f32_init());
        //clip_image_save_to_bmp(resized_image, "resized.bmp");
        normalize_image_u8_to_f32(resized_image, *img_f32, ctx->image_mean, ctx->image_std);
        res_imgs->entries.push_back(std::move(img_f32));
        return true;
    } else if (ctx->proj_type == PROJECTOR_TYPE_PIXTRAL) {
        clip_image_u8 resized_image;
        auto          new_size =
            image_manipulation::calc_size_preserved_ratio(original_size, params.patch_size, params.image_size);
        image_manipulation::bilinear_resize(*img, resized_image, new_size.width, new_size.height);
        clip_image_f32_ptr img_f32(clip_image_f32_init());
        normalize_image_u8_to_f32(resized_image, *img_f32, ctx->image_mean, ctx->image_std);
        res_imgs->entries.push_back(std::move(img_f32));
        return true;
    }

    // the logic below is to pad the shorter side to the longer side with a background color: rgb(122, 116, 104)
    // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156

    clip_image_u8_ptr temp(clip_image_u8_init());  // we will keep the input image data here temporarily

    if (pad_to_square) {
        // for llava-1.5, we resize image to a square, and pad the shorter side with a background color
        // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156
        const int longer_side = std::max(img->nx, img->ny);
        temp->nx              = longer_side;
        temp->ny              = longer_side;
        temp->buf.resize(3 * longer_side * longer_side);

        // background color in RGB from LLaVA (this is the mean rgb color * 255)
        const std::array<uint8_t, 3> pad_color = { 122, 116, 104 };

        // resize the image to the target_size
        image_manipulation::resize_and_pad_image(*img, *temp, clip_image_size{ params.image_size, params.image_size },
                                                 pad_color);

        clip_image_f32_ptr res(clip_image_f32_init());
        normalize_image_u8_to_f32(*temp, *res, ctx->image_mean, ctx->image_std);
        res_imgs->entries.push_back(std::move(res));
        return true;

    } else if (!params.image_grid_pinpoints.empty()) {
        // "spatial_unpad" with "anyres" processing for llava-1.6
        const auto                     inst = llava_uhd::get_slice_instructions(ctx, original_size);
        std::vector<clip_image_u8_ptr> imgs = llava_uhd::slice_image(img, inst);

        for (size_t i = 0; i < imgs.size(); ++i) {
            // clip_image_save_to_bmp(*imgs[i], "slice_" + std::to_string(i) + ".bmp");
            clip_image_f32_ptr res(clip_image_f32_init());
            normalize_image_u8_to_f32(*imgs[i], *res, ctx->image_mean, ctx->image_std);
            res_imgs->entries.push_back(std::move(res));
        }

        return true;
    }

    GGML_ASSERT(false && "Unknown image preprocessing type");
}

ggml_tensor * clip_get_newline_tensor(const struct clip_ctx * ctx) {
    return ctx->vision_model.image_newline;
}

void clip_free(clip_ctx * ctx) {
    if (ctx == nullptr) {
        return;
    }
    delete ctx;
}

// deprecated
size_t clip_embd_nbytes(const struct clip_ctx * ctx) {
    const int32_t nx = ctx->vision_model.hparams.image_size;
    const int32_t ny = ctx->vision_model.hparams.image_size;
    return clip_embd_nbytes_by_img(ctx, nx, ny);
}

size_t clip_embd_nbytes_by_img(const struct clip_ctx * ctx, int img_w, int img_h) {
    clip_image_f32 img;
    img.nx = img_w;
    img.ny = img_h;
    return clip_n_output_tokens(ctx, &img) * clip_n_mmproj_embd(ctx) * sizeof(float);
}

int32_t clip_get_image_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.image_size;
}

int32_t clip_get_patch_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.patch_size;
}

int32_t clip_get_hidden_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.n_embd;
}

const char * clip_patch_merge_type(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.mm_patch_merge_type == PATCH_MERGE_SPATIAL_UNPAD ? "spatial_unpad" : "flat";
}

const int32_t * clip_image_grid(const struct clip_ctx * ctx) {
    if (ctx->vision_model.hparams.image_grid_pinpoints.size()) {
        return &ctx->vision_model.hparams.image_grid_pinpoints.front();
    }
    return nullptr;
}

size_t get_clip_image_grid_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.image_grid_pinpoints.size();
}

int clip_n_output_tokens_x(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params  = ctx->vision_model.hparams;
    const int    n_total = clip_n_output_tokens(ctx, img);
    if (ctx->proj_type == PROJECTOR_TYPE_QWEN2VL || ctx->proj_type == PROJECTOR_TYPE_QWEN25VL) {
        return img->nx / (params.patch_size * 2) + (int) (img->nx % params.patch_size > 0);
    }
    return n_total;
}

int clip_n_output_tokens_y(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->vision_model.hparams;
    if (ctx->proj_type == PROJECTOR_TYPE_QWEN2VL || ctx->proj_type == PROJECTOR_TYPE_QWEN25VL) {
        return img->ny / (params.patch_size * 2) + (int) (img->ny % params.patch_size > 0);
    }
    return 1;
}

int clip_n_output_tokens(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->vision_model.hparams;

    int n_patches = (params.image_size / params.patch_size) * (params.image_size / params.patch_size);

    if (ctx->proj_type == PROJECTOR_TYPE_LDP || ctx->proj_type == PROJECTOR_TYPE_LDPV2 ||
        ctx->proj_type == PROJECTOR_TYPE_GLM_EDGE) {
        n_patches /= 4;
        if (ctx->vision_model.mm_glm_tok_boi) {
            n_patches += 2;  // for BOI and EOI token embeddings
        }
    } else if (ctx->proj_type == PROJECTOR_TYPE_MINICPMV) {
        if (ctx->minicpmv_version == 2) {
            n_patches = 96;
        } else if (ctx->minicpmv_version == 3) {
            n_patches = 64;
        } else if (ctx->minicpmv_version == 4) {
            n_patches = 64;
        } else {
            GGML_ABORT("Unknown minicpmv version");
        }
    } else if (ctx->proj_type == PROJECTOR_TYPE_QWEN2VL || ctx->proj_type == PROJECTOR_TYPE_QWEN25VL) {
        int patch_size = params.patch_size * 2;
        int x_patch    = img->nx / patch_size + (int) (img->nx % patch_size > 0);
        int y_patch    = img->ny / patch_size + (int) (img->ny % patch_size > 0);
        n_patches      = x_patch * y_patch;
    } else if (ctx->proj_type == PROJECTOR_TYPE_GEMMA3) {
        int n_per_side         = params.image_size / params.patch_size;
        int n_per_side_2d_pool = n_per_side / params.proj_scale_factor;
        n_patches              = n_per_side_2d_pool * n_per_side_2d_pool;
    } else if (ctx->proj_type == PROJECTOR_TYPE_IDEFICS3 || ctx->proj_type == PROJECTOR_TYPE_INTERNVL) {
        // both W and H are divided by proj_scale_factor
        n_patches /= (params.proj_scale_factor * params.proj_scale_factor);
    } else if (ctx->proj_type == PROJECTOR_TYPE_PIXTRAL) {
        int n_merge     = params.spatial_merge_size;
        int n_patches_x = img->nx / params.patch_size / (n_merge > 0 ? n_merge : 1);
        int n_patches_y = img->ny / params.patch_size / (n_merge > 0 ? n_merge : 1);
        n_patches = n_patches_y * n_patches_x + n_patches_y - 1;  // + one [IMG_BREAK] per row, except the last row
    }

    return n_patches;
}

static std::vector<std::vector<std::vector<float>>> get_1d_sincos_pos_embed_from_grid_new(
    int embed_dim, const std::vector<std::vector<float>> & pos) {
    assert(embed_dim % 2 == 0);
    int H = pos.size();
    int W = pos[0].size();

    std::vector<float> omega(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; ++i) {
        omega[i] = 1.0 / pow(10000.0, static_cast<float>(i) / (embed_dim / 2));
    }

    std::vector<std::vector<std::vector<float>>> emb(H,
                                                     std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                float out_value              = pos[h][w] * omega[d];
                emb[h][w][d]                 = sin(out_value);
                emb[h][w][d + embed_dim / 2] = cos(out_value);
            }
        }
    }

    return emb;
}

static std::vector<std::vector<std::vector<float>>> get_2d_sincos_pos_embed_from_grid(
    int embed_dim, const std::vector<std::vector<std::vector<float>>> & grid) {
    assert(embed_dim % 2 == 0);
    std::vector<std::vector<std::vector<float>>> emb_h =
        get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[0]);  // (H, W, D/2)
    std::vector<std::vector<std::vector<float>>> emb_w =
        get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[1]);  // (H, W, D/2)

    int                                          H = emb_h.size();
    int                                          W = emb_h[0].size();
    std::vector<std::vector<std::vector<float>>> emb(H,
                                                     std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                emb[h][w][d]                 = emb_h[h][w][d];
                emb[h][w][d + embed_dim / 2] = emb_w[h][w][d];
            }
        }
    }
    return emb;
}

static std::vector<std::vector<float>> get_2d_sincos_pos_embed(int embed_dim, const std::pair<int, int> image_size) {
    int grid_h_size = image_size.first;
    int grid_w_size = image_size.second;

    std::vector<float> grid_h(grid_h_size);
    std::vector<float> grid_w(grid_w_size);

    for (int i = 0; i < grid_h_size; ++i) {
        grid_h[i] = static_cast<float>(i);
    }
    for (int i = 0; i < grid_w_size; ++i) {
        grid_w[i] = static_cast<float>(i);
    }

    std::vector<std::vector<float>> grid(grid_h_size, std::vector<float>(grid_w_size));
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid[h][w] = grid_w[w];
        }
    }
    std::vector<std::vector<std::vector<float>>> grid_2d = { grid, grid };
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid_2d[0][h][w] = grid_h[h];
            grid_2d[1][h][w] = grid_w[w];
        }
    }

    std::vector<std::vector<std::vector<float>>> pos_embed_3d = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_2d);

    int                             H = image_size.first;
    int                             W = image_size.second;
    std::vector<std::vector<float>> pos_embed_2d(H * W, std::vector<float>(embed_dim));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            pos_embed_2d[w * H + h] = pos_embed_3d[h][w];
        }
    }

    return pos_embed_2d;
}

bool clip_image_encode(struct clip_ctx * ctx, const int n_threads, clip_image_f32 * img, float * vec) {
    clip_image_f32_batch imgs;
    clip_image_f32_ptr   img_copy(clip_image_f32_init());
    *img_copy = *img;
    imgs.entries.push_back(std::move(img_copy));

    return clip_image_batch_encode(ctx, n_threads, &imgs, vec);
}

bool clip_image_batch_encode(clip_ctx * ctx, const int n_threads, const clip_image_f32_batch * imgs_c_ptr,
                             float * vec) {
    const clip_image_f32_batch & imgs       = *imgs_c_ptr;
    int                          batch_size = imgs.entries.size();

    // TODO @ngxson : implement batch size > 1 as a loop
    //                we don't need true batching support because the cgraph will gonna be big anyway
    if (batch_size != 1) {
        return false;  // only support batch size of 1
    }

    // build the inference graph
    ggml_backend_sched_reset(ctx->sched.get());
    ggml_cgraph * gf = clip_image_build_graph(ctx, imgs);
    ggml_backend_sched_alloc_graph(ctx->sched.get(), gf);

    // set inputs
    const auto & model   = ctx->vision_model;
    const auto & hparams = model.hparams;

    const int image_size_width  = imgs.entries[0]->nx;
    const int image_size_height = imgs.entries[0]->ny;

    const int patch_size  = hparams.patch_size;
    const int num_patches = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int n_pos       = num_patches + (model.class_embedding ? 1 : 0);
    const int pos_w       = ctx->load_image_size.width / patch_size;
    const int pos_h       = ctx->load_image_size.height / patch_size;

    const bool use_window_attn = hparams.n_wa_pattern > 0;  // for qwen2.5vl

    auto get_inp_tensor = [&gf](const char * name) {
        ggml_tensor * inp = ggml_graph_get_tensor(gf, name);
        if (inp == nullptr) {
            GGML_ABORT("Failed to get tensor %s", name);
        }
        if (!(inp->flags & GGML_TENSOR_FLAG_INPUT)) {
            GGML_ABORT("Tensor %s is not an input tensor", name);
        }
        return inp;
    };

    auto set_input_f32 = [&get_inp_tensor](const char * name, std::vector<float> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t) values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    auto set_input_i32 = [&get_inp_tensor](const char * name, std::vector<int32_t> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t) values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    // set input pixel values
    {
        size_t nelem = 0;
        for (const auto & img : imgs.entries) {
            nelem += img->nx * img->ny * 3;
        }
        std::vector<float> inp_raw(nelem);

        // layout of data (note: the channel dim is unrolled to better visualize the layout):
        //
        // ┌──W──┐
        // │     H │  channel = R
        // ├─────┤ │
        // │     H │  channel = G
        // ├─────┤ │
        // │     H │  channel = B
        // └─────┘ │
        //   ──────┘ x B

        for (size_t i = 0; i < imgs.entries.size(); i++) {
            const int nx = imgs.entries[i]->nx;
            const int ny = imgs.entries[i]->ny;
            const int n  = nx * ny;

            for (int b = 0; b < batch_size; b++) {
                float * batch_entry = inp_raw.data() + b * (3 * n);
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                        size_t base_src               = 3 * (y * nx + x);  // idx of the first channel
                        size_t base_dst               = y * nx + x;        // idx of the first channel
                        batch_entry[base_dst]         = imgs.entries[b]->buf[base_src];
                        batch_entry[1 * n + base_dst] = imgs.entries[b]->buf[base_src + 1];
                        batch_entry[2 * n + base_dst] = imgs.entries[b]->buf[base_src + 2];
                    }
                }
            }
        }
        set_input_f32("inp_raw", inp_raw);
    }

    // set input per projector
    switch (ctx->proj_type) {
        case PROJECTOR_TYPE_MINICPMV:
            {
                // inspired from siglip:
                //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
                //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit/blob/d66538faeba44480d0bfaa42145eef26f9423199/modeling_siglip.py#L316
                std::vector<int32_t> positions(pos_h * pos_w);
                int                  bucket_coords_h[1024];
                int                  bucket_coords_w[1024];
                for (int i = 0; i < pos_h; i++) {
                    bucket_coords_h[i] = std::floor(70.0 * i / pos_h);
                }
                for (int i = 0; i < pos_w; i++) {
                    bucket_coords_w[i] = std::floor(70.0 * i / pos_w);
                }
                for (int i = 0, id = 0; i < pos_h; i++) {
                    for (int j = 0; j < pos_w; j++) {
                        positions[id++] = bucket_coords_h[i] * 70 + bucket_coords_w[j];
                    }
                }
                set_input_i32("positions", positions);

                // inspired from resampler of Qwen-VL:
                //    -> https://huggingface.co/Qwen/Qwen-VL/tree/main
                //    -> https://huggingface.co/Qwen/Qwen-VL/blob/0547ed36a86561e2e42fecec8fd0c4f6953e33c4/visual.py#L23
                int embed_dim = clip_n_mmproj_embd(ctx);

                // TODO @ngxson : this is very inefficient, can we do this using ggml_sin and ggml_cos?
                auto pos_embed_t = get_2d_sincos_pos_embed(embed_dim, std::make_pair(pos_w, pos_h));

                std::vector<float> pos_embed(embed_dim * pos_w * pos_h);
                for (int i = 0; i < pos_w * pos_h; ++i) {
                    for (int j = 0; j < embed_dim; ++j) {
                        pos_embed[i * embed_dim + j] = pos_embed_t[i][j];
                    }
                }

                set_input_f32("pos_embed", pos_embed);
            }
            break;
        case PROJECTOR_TYPE_QWEN2VL:
            {
                const int        merge_ratio = 2;
                const int        pw          = image_size_width / patch_size;
                const int        ph          = image_size_height / patch_size;
                std::vector<int> positions(n_pos * 4);
                int              ptr = 0;
                for (int y = 0; y < ph; y += merge_ratio) {
                    for (int x = 0; x < pw; x += merge_ratio) {
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                positions[ptr]                   = y + dy;
                                positions[num_patches + ptr]     = x + dx;
                                positions[2 * num_patches + ptr] = y + dy;
                                positions[3 * num_patches + ptr] = x + dx;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            }
            break;
        case PROJECTOR_TYPE_QWEN25VL:
            {
                // pw * ph = number of tokens output by ViT after apply patch merger
                // ipw * ipw = number of vision token been processed inside ViT
                const int merge_ratio = 2;
                const int pw          = image_size_width / patch_size / merge_ratio;
                const int ph          = image_size_height / patch_size / merge_ratio;
                const int ipw         = image_size_width / patch_size;
                const int iph         = image_size_height / patch_size;

                std::vector<int> idx(ph * pw);
                std::vector<int> inv_idx(ph * pw);

                if (use_window_attn) {
                    const int          attn_window_size = 112;
                    const int          grid_window      = attn_window_size / patch_size / merge_ratio;
                    int                dst              = 0;
                    // [num_vision_tokens, num_vision_tokens] attention mask tensor
                    std::vector<float> mask(pow(ipw * iph, 2), std::numeric_limits<float>::lowest());
                    int                mask_row = 0;

                    for (int y = 0; y < ph; y += grid_window) {
                        for (int x = 0; x < pw; x += grid_window) {
                            const int win_h = std::min(grid_window, ph - y);
                            const int win_w = std::min(grid_window, pw - x);
                            const int dst_0 = dst;
                            // group all tokens belong to the same window togather (to a continue range)
                            for (int dy = 0; dy < win_h; dy++) {
                                for (int dx = 0; dx < win_w; dx++) {
                                    const int src = (y + dy) * pw + (x + dx);
                                    GGML_ASSERT(src < (int) idx.size());
                                    GGML_ASSERT(dst < (int) inv_idx.size());
                                    idx[src]     = dst;
                                    inv_idx[dst] = src;
                                    dst++;
                                }
                            }

                            for (int r = 0; r < win_h * win_w * merge_ratio * merge_ratio; r++) {
                                int row_offset = mask_row * (ipw * iph);
                                std::fill(mask.begin() + row_offset + (dst_0 * merge_ratio * merge_ratio),
                                          mask.begin() + row_offset + (dst * merge_ratio * merge_ratio), 0.0);
                                mask_row++;
                            }
                        }
                    }

                    set_input_i32("window_idx", idx);
                    set_input_i32("inv_window_idx", inv_idx);
                    set_input_f32("window_mask", mask);
                } else {
                    for (int i = 0; i < ph * pw; i++) {
                        idx[i] = i;
                    }
                }

                const int        mpow = merge_ratio * merge_ratio;
                std::vector<int> positions(n_pos * 4);

                int ptr = 0;
                for (int y = 0; y < iph; y += merge_ratio) {
                    for (int x = 0; x < ipw; x += merge_ratio) {
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                auto remap = idx[ptr / mpow];
                                remap      = (remap * mpow) + (ptr % mpow);

                                positions[remap]                   = y + dy;
                                positions[num_patches + remap]     = x + dx;
                                positions[2 * num_patches + remap] = y + dy;
                                positions[3 * num_patches + remap] = x + dx;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            }
            break;
        case PROJECTOR_TYPE_PIXTRAL:
            {
                // set the 2D positions
                int              n_patches_per_col = image_size_width / patch_size;
                std::vector<int> pos_data(n_pos);
                // dimension H
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i / n_patches_per_col;
                }
                set_input_i32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i % n_patches_per_col;
                }
                set_input_i32("pos_w", pos_data);
            }
            break;
        case PROJECTOR_TYPE_GLM_EDGE:
            {
                // llava and other models
                std::vector<int32_t> positions(n_pos);
                for (int i = 0; i < n_pos; i++) {
                    positions[i] = i;
                }
                set_input_i32("positions", positions);
            }
            break;
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
            {
                // llava and other models
                std::vector<int32_t> positions(n_pos);
                for (int i = 0; i < n_pos; i++) {
                    positions[i] = i;
                }
                set_input_i32("positions", positions);

                // The patches vector is used to get rows to index into the embeds with;
                // we should skip dim 0 only if we have CLS to avoid going out of bounds
                // when retrieving the rows.
                int                  patch_offset = model.class_embedding ? 1 : 0;
                std::vector<int32_t> patches(num_patches);
                for (int i = 0; i < num_patches; i++) {
                    patches[i] = i + patch_offset;
                }
                set_input_i32("patches", patches);
            }
            break;
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_NONE:
            {
                // do nothing
            }
            break;
        default:
            GGML_ABORT("Unknown projector type");
    }

    // ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backend_cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto ggml_backend_set_n_threads_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(ctx->backend_cpu, n_threads);
        }
    }

    auto status = ggml_backend_sched_graph_compute(ctx->sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
        return false;
    }

    // wait until all device work is complete before accessing tensors from CPU
    ggml_backend_sched_synchronize(ctx->sched.get());

    // the last node is the embedding tensor
    ggml_tensor * embeddings = ggml_graph_node(gf, -1);

    // sanity check (only support batch size of 1 for now)
    const int n_tokens_out          = embeddings->ne[1];
    const int expected_n_tokens_out = clip_n_output_tokens(ctx, imgs.entries[0].get());
    if (n_tokens_out != expected_n_tokens_out) {
        LOG_ERR("%s: expected %d tokens, got %d\n", __func__, expected_n_tokens_out, n_tokens_out);
        GGML_ABORT("Invalid number of output tokens");
    }

    // copy the embeddings to the location passed by the user
    ggml_backend_tensor_get(embeddings, vec, 0, ggml_nbytes(embeddings));

    return true;
}

int clip_n_mmproj_embd(const struct clip_ctx * ctx) {
    switch (ctx->proj_type) {
        case PROJECTOR_TYPE_LDP:
            return ctx->vision_model.mm_model_block_1_block_2_1_b->ne[0];
        case PROJECTOR_TYPE_LDPV2:
            return ctx->vision_model.mm_model_peg_0_b->ne[0];
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_PIXTRAL:
            return ctx->vision_model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_MLP_NORM:
            return ctx->vision_model.mm_3_b->ne[0];
        case PROJECTOR_TYPE_MINICPMV:
            if (ctx->minicpmv_version == 2) {
                return 4096;
            } else if (ctx->minicpmv_version == 3) {
                return 3584;
            } else if (ctx->minicpmv_version == 4) {
                return 3584;
            }
            GGML_ABORT("Unknown minicpmv version");
        case PROJECTOR_TYPE_GLM_EDGE:
            return ctx->vision_model.mm_model_mlp_3_w->ne[1];
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
            return ctx->vision_model.mm_1_b->ne[0];
        case PROJECTOR_TYPE_GEMMA3:
            return ctx->vision_model.mm_input_proj_w->ne[0];
        case PROJECTOR_TYPE_IDEFICS3:
            return ctx->vision_model.projection->ne[1];
        case PROJECTOR_TYPE_INTERNVL:
            return ctx->vision_model.mm_3_w->ne[1];
        case PROJECTOR_TYPE_NONE:
            // dual-encoder with no projector; embedding dim equals ViT hidden size
            return ctx->vision_model.hparams.n_embd;
        default:
            GGML_ABORT("Unknown projector type");
    }
}

int clip_is_minicpmv(const struct clip_ctx * ctx) {
    if (ctx->proj_type == PROJECTOR_TYPE_MINICPMV) {
        return ctx->minicpmv_version;
    }
    return 0;
}

bool clip_is_glm(const struct clip_ctx * ctx) {
    return ctx->proj_type == PROJECTOR_TYPE_GLM_EDGE;
}

bool clip_is_qwen2vl(const struct clip_ctx * ctx) {
    return ctx->proj_type == PROJECTOR_TYPE_QWEN2VL || ctx->proj_type == PROJECTOR_TYPE_QWEN25VL;
}

bool clip_is_llava(const struct clip_ctx * ctx) {
    return ctx->has_llava_projector;
}

bool clip_is_gemma3(const struct clip_ctx * ctx) {
    return ctx->proj_type == PROJECTOR_TYPE_GEMMA3;
}

bool clip_encode_float_image(struct clip_ctx * ctx, int n_threads, float * img, int h, int w, float * vec) {
    clip_image_f32 clip_img;
    clip_img.buf.resize(h * w * 3);
    for (int i = 0; i < h * w * 3; i++) {
        clip_img.buf[i] = img[i];
    }
    clip_img.nx = w;
    clip_img.ny = h;
    clip_image_encode(ctx, n_threads, &clip_img, vec);
    return true;
}

//
// API used internally with mtmd
//

projector_type clip_get_projector_type(const struct clip_ctx * ctx) {
    return ctx->proj_type;
}

gguf_context * clip_get_gguf(struct clip_ctx * ctx) {
    return ctx ? ctx->ctx_gguf.get() : nullptr;
}

// -----------------------------------------------------------------------------
// Text encoding helper (mean-pooled embeddings + optional projection)
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Text transformer graph builder (Phase 2 full implementation)
// -----------------------------------------------------------------------------
// Builds the full Transformer encoder graph for text sequences:
// - input token IDs -> token & positional embeddings
// - N transformer layers (pre-LN, multi-head attention with contiguity handling, post-attention, and feed-forward)
// - final layer norm, CLS token slicing, and optional projection
// Caches the graph in ctx->text_gf to avoid rebuilding per call.

static ggml_cgraph * clip_text_build_graph_impl(ggml_context *    ctx0,  // The ggml_context to build the graph on
                                                clip_text_model & tm,    // Reference to the text model
                                                int               n_tokens) {
    ggml_cgraph * gf = ggml_new_graph(ctx0);
    if (gf == nullptr) {
        gf = ggml_new_graph(ctx0);  // Create a new graph if one isn't provided for reuse
    } else {
        ggml_graph_clear(gf);       // Clear the nodes if we are reusing an existing graph object
    }

    const int64_t n_embd       = tm.hparams.n_embd;
    const int64_t n_head       = tm.hparams.n_head;
    const int64_t n_layer      = tm.hparams.n_layer;
    const float   eps          = tm.hparams.eps;
    const float   d_head_float = (float) n_embd / (float) n_head;
    const float   kq_scale     = 1.0f / sqrtf(d_head_float);

    GGML_ASSERT(n_layer > 0 && "Text tower n_layer is 0, cannot build graph.");
    GGML_ASSERT(n_embd > 0 && "Text tower n_embd is 0.");
    GGML_ASSERT(n_head > 0 && "Text tower n_head is 0.");

    // Input token IDs
    ggml_tensor * t_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(t_ids, "text_tokens_input");
    ggml_set_input(t_ids);  // Mark as graph input

    // Token embeddings
    ggml_tensor * cur = ggml_get_rows(ctx0, tm.token_embeddings, t_ids);
    fprintf(stderr, "[TEXT_GRAPH_DEBUG] token_embeddings shape: [%lld, %lld], t_ids shape: [%lld]\n",
            (long long) tm.token_embeddings->ne[0], (long long) tm.token_embeddings->ne[1], (long long) t_ids->ne[0]);
    fprintf(stderr, "[TEXT_GRAPH_DEBUG] After token_embeddings lookup, cur shape: [%lld, %lld, %lld, %lld]\n",
            (long long) cur->ne[0], (long long) cur->ne[1], (long long) cur->ne[2], (long long) cur->ne[3]);

    // Positional IDs
    ggml_tensor * p_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(p_ids, "text_pos_ids_input");
    ggml_set_input(p_ids);  // Mark as graph input

    // Positional embeddings
    ggml_tensor * pos_emb = ggml_get_rows(ctx0, tm.position_embeddings, p_ids);
    fprintf(stderr, "[TEXT_GRAPH_DEBUG] position_embeddings shape: [%lld, %lld], p_ids shape: [%lld]\n",
            (long long) tm.position_embeddings->ne[0], (long long) tm.position_embeddings->ne[1],
            (long long) p_ids->ne[0]);
    fprintf(stderr, "[TEXT_GRAPH_DEBUG] After pos_embeddings lookup, pos_emb shape: [%lld, %lld, %lld, %lld]\n",
            (long long) pos_emb->ne[0], (long long) pos_emb->ne[1], (long long) pos_emb->ne[2],
            (long long) pos_emb->ne[3]);

    // Add token and positional embeddings
    fprintf(stderr, "[TEXT_GRAPH_ADD_DEBUG] Adding token_embds and pos_embds\n");
    fprintf(stderr, "[TEXT_GRAPH_ADD_DEBUG] cur (token_embds) ne: [%lld, %lld, %lld, %lld], nb: [%zu, %zu, %zu, %zu]\n",
            (long long) cur->ne[0], (long long) cur->ne[1], (long long) cur->ne[2], (long long) cur->ne[3], cur->nb[0],
            cur->nb[1], cur->nb[2], cur->nb[3]);
    fprintf(stderr, "[TEXT_GRAPH_ADD_DEBUG] pos_emb   ne: [%lld, %lld, %lld, %lld], nb: [%zu, %zu, %zu, %zu]\n",
            (long long) pos_emb->ne[0], (long long) pos_emb->ne[1], (long long) pos_emb->ne[2],
            (long long) pos_emb->ne[3], pos_emb->nb[0], pos_emb->nb[1], pos_emb->nb[2], pos_emb->nb[3]);
    cur = ggml_add(ctx0, cur, pos_emb);
    fprintf(stderr, "[TEXT_GRAPH_DEBUG] After adding pos_emb, cur shape: [%lld, %lld, %lld, %lld]\n",
            (long long) cur->ne[0], (long long) cur->ne[1], (long long) cur->ne[2], (long long) cur->ne[3]);

    // Transformer Layers
    for (int il = 0; il < n_layer; ++il) {
        fprintf(stderr, "[TEXT_GRAPH_DEBUG] Layer %d START\n", il);
        auto &        layer = tm.layers[il];
        ggml_tensor * inp_L = cur;  // Residual connection

        // LayerNorm 1
        cur = ggml_norm(ctx0, cur, eps);
        fprintf(stderr, "[TEXT_GRAPH_DEBUG] Layer %d LN1_norm ne: [%lld, %lld]\n", il, (long long) cur->ne[0],
                (long long) cur->ne[1]);
        cur = ggml_mul(ctx0, cur, layer.ln_1_w);
        cur = ggml_add(ctx0, cur, layer.ln_1_b);
        fprintf(stderr, "[TEXT_GRAPH_DEBUG] Layer %d After LN1 ne: [%lld, %lld]\n", il, (long long) cur->ne[0],
                (long long) cur->ne[1]);

        ggml_tensor * attn_input = cur;

        // Self-Attention
        ggml_tensor * Q = ggml_mul_mat(ctx0, layer.q_w, attn_input);
        if (layer.q_b) {
            Q = ggml_add(ctx0, Q, layer.q_b);
        }
        ggml_tensor * K = ggml_mul_mat(ctx0, layer.k_w, attn_input);
        if (layer.k_b) {
            K = ggml_add(ctx0, K, layer.k_b);
        }
        ggml_tensor * V = ggml_mul_mat(ctx0, layer.v_w, attn_input);
        if (layer.v_b) {
            V = ggml_add(ctx0, V, layer.v_b);
        }

        Q = ggml_reshape_3d(ctx0, Q, (int64_t) d_head_float, n_head, n_tokens);
        K = ggml_reshape_3d(ctx0, K, (int64_t) d_head_float, n_head, n_tokens);
        V = ggml_reshape_3d(ctx0, V, (int64_t) d_head_float, n_head, n_tokens);

        ggml_tensor * q_perm = ggml_permute(ctx0, Q, 0, 2, 1, 3);
        ggml_tensor * k_perm = ggml_permute(ctx0, K, 0, 2, 1, 3);
        ggml_tensor * v_perm = ggml_permute(ctx0, V, 1, 2, 0, 3);
        v_perm               = ggml_cont(ctx0, v_perm);

        ggml_tensor * KQ  = ggml_mul_mat(ctx0, k_perm, q_perm);
        KQ                = ggml_soft_max_ext(ctx0, KQ, nullptr, kq_scale, 0.0f);
        ggml_tensor * KQV = ggml_mul_mat(ctx0, v_perm, KQ);

        cur = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, n_embd, n_tokens);

        cur = ggml_mul_mat(ctx0, layer.o_w, cur);
        if (layer.o_b) {
            cur = ggml_add(ctx0, cur, layer.o_b);
        }
        // Your detailed log for this was:
        // [TEXT_GRAPH_ADD_DEBUG] Layer %d Attn Out Bias: cur ne: [%lld, %lld], o_b ne: [%lld, %lld]\n"
        // This is fine, just ensure the shapes are compatible as per ggml_can_repeat.

        cur   = ggml_add(ctx0, cur, inp_L);  // Residual Add 1
        inp_L = cur;                         // Update for next residual

        // LayerNorm 2
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.ln_2_w);
        cur = ggml_add(ctx0, cur, layer.ln_2_b);

        // FFN
        ggml_tensor * ffn_inp = cur;
        ggml_tensor * ff_up   = ggml_mul_mat(ctx0, layer.ff_up_w, ffn_inp);
        if (layer.ff_up_b) {
            ggml_tensor * ff_up_b_corrected = layer.ff_up_b;
            if (ff_up->ne[0] != layer.ff_up_b->ne[0] ||
                (ff_up->ne[1] != layer.ff_up_b->ne[1] && layer.ff_up_b->ne[1] == 1)) {
                ff_up_b_corrected = ggml_repeat(ctx0, layer.ff_up_b, ff_up);
            }
            ff_up = ggml_add(ctx0, ff_up, ff_up_b_corrected);
        }
        ggml_tensor * ff_act  = ggml_gelu_quick(ctx0, ff_up);
        ggml_tensor * ff_down = ggml_mul_mat(ctx0, layer.ff_down_w, ff_act);
        if (layer.ff_down_b) {
            ggml_tensor * ff_down_b_corrected = layer.ff_down_b;
            if (ff_down->ne[0] != layer.ff_down_b->ne[0] ||
                (ff_down->ne[1] != layer.ff_down_b->ne[1] && layer.ff_down_b->ne[1] == 1)) {
                ff_down_b_corrected = ggml_repeat(ctx0, layer.ff_down_b, ff_down);
            }
            ff_down = ggml_add(ctx0, ff_down, ff_down_b_corrected);
        }

        // Your detailed log for the crash site was:
        // [TEXT_GRAPH_ADD_DEBUG] Layer %d Residual 2 PRE-ADD (CRASH SITE):
        //   ff_down (a): name='%s', type=%d, ne=[%lld,%lld,%lld,%lld], nb=[%zu,%zu,%zu,%zu], data=%p, op=%s
        //   ffn_inp (b): name='%s', type=%d, ne=[%lld,%lld,%lld,%lld], nb=[%zu,%zu,%zu,%zu], data=%p, op=%s
        cur = ggml_add(ctx0, ff_down, ffn_inp);  // Residual Add 2 (was ffn_inp, not inp_L)

        fprintf(stderr, "[TEXT_GRAPH_DEBUG] Layer %d END, cur ne: [%lld, %lld]\n", il, (long long) cur->ne[0],
                (long long) cur->ne[1]);
    }

    // Final LayerNorm
    cur = ggml_norm(ctx0, cur, eps);
    if (tm.post_ln_w && tm.post_ln_b) {
        cur = ggml_mul(ctx0, cur, tm.post_ln_w);
        cur = ggml_add(ctx0, cur, tm.post_ln_b);
    }
    fprintf(stderr, "[TEXT_GRAPH_DEBUG] After Final LN, cur ne: [%lld, %lld]\n", (long long) cur->ne[0],
            (long long) cur->ne[1]);

    // Slice CLS token (first token, index 0)
    const size_t  nb1_cls        = cur->nb[1];
    ggml_tensor * cls_token_embd = ggml_view_2d(ctx0, cur, n_embd, 1, nb1_cls, 0);
    ggml_set_name(cls_token_embd, "cls_token_embd");
    fprintf(stderr, "[TEXT_GRAPH_DEBUG] CLS token embd ne: [%lld, %lld]\n", (long long) cls_token_embd->ne[0],
            (long long) cls_token_embd->ne[1]);

    // Optional Projection
    ggml_tensor * final_output = cls_token_embd;
    if (tm.projection) {
        fprintf(
            stderr,
            "[TEXT_GRAPH_DEBUG] Applying final projection. Input (CLS) ne: [%lld, %lld], Projection ne: [%lld, %lld]\n",
            (long long) cls_token_embd->ne[0], (long long) cls_token_embd->ne[1], (long long) tm.projection->ne[0],
            (long long) tm.projection->ne[1]);
        final_output = ggml_mul_mat(ctx0, tm.projection, cls_token_embd);
        fprintf(stderr, "[TEXT_GRAPH_DEBUG] After projection, final_output ne: [%lld, %lld]\n",
                (long long) final_output->ne[0], (long long) final_output->ne[1]);
    }
    ggml_set_name(final_output, "text_embedding_output");

    ggml_build_forward_expand(gf, final_output);
    return gf;
}

bool clip_text_encode(struct clip_ctx * ctx, int n_threads_arg, const int32_t * tokens_input_arg, int n_tokens,
                      float * vec_out) {
    GGML_UNUSED(n_threads_arg);

    if (ctx == nullptr || tokens_input_arg == nullptr || n_tokens <= 0 || vec_out == nullptr) {
        fprintf(stderr, "%s: Invalid arguments.\n", __func__);
        return false;
    }

    auto & tm = ctx->text_model;
    if (tm.token_embeddings == nullptr || tm.layers.empty() || !tm.position_embeddings) {
        fprintf(stderr, "%s: Text encoder not fully initialized or present.\n", __func__);
        return false;
    }

    const int dim_out_actual = tm.n_embd_out > 0 ? tm.n_embd_out : tm.n_embd;

    ggml_context_ptr temp_text_ctx0_ptr = nullptr;
    ggml_cgraph *    temp_text_gf       = nullptr;

    struct ggml_init_params iparams_text_ctx0 = {
        ctx->buf_compute_meta.size(), ctx->buf_compute_meta.data(), true, /* no_alloc */
    };
    temp_text_ctx0_ptr.reset(ggml_init(iparams_text_ctx0));
    if (!temp_text_ctx0_ptr) {
        fprintf(stderr, "%s: Failed to initialize temporary text_ctx0_ptr for graph building.\n", __func__);
        return false;
    }

    fprintf(stderr, "%s: Building temporary text graph for n_tokens = %d.\n", __func__, n_tokens);
    temp_text_gf = clip_text_build_graph_impl(temp_text_ctx0_ptr.get(), tm, n_tokens);  // Pass nullptr for gf_reuse
    if (!temp_text_gf) {
        fprintf(stderr, "%s: Failed to build temporary text graph.\n", __func__);
        return false;
    }

    ggml_backend_sched_reset(ctx->sched.get());
    if (!ggml_backend_sched_alloc_graph(ctx->sched.get(), temp_text_gf)) {
        fprintf(stderr, "%s: ggml_backend_sched_alloc_graph failed for text graph.\n", __func__);
        return false;
    }

    struct ggml_tensor * t_ids_graph = ggml_graph_get_tensor(temp_text_gf, "text_tokens_input");
    struct ggml_tensor * p_ids_graph = ggml_graph_get_tensor(temp_text_gf, "text_pos_ids_input");

    if (!t_ids_graph || !p_ids_graph) {
        fprintf(stderr, "%s: Could not find input tensors in temporary graph.\n", __func__);
        // ggml_backend_sched_free was removed, reset will handle next time
        return false;
    }
    std::vector<int32_t> position_ids_vec(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        position_ids_vec[i] = i;
    }

    ggml_backend_tensor_set(t_ids_graph, tokens_input_arg, 0, ggml_nbytes(t_ids_graph));
    ggml_backend_tensor_set(p_ids_graph, position_ids_vec.data(), 0, ggml_nbytes(p_ids_graph));

    // Corrected debug prints:
    const char * t_ids_buft_name = "N/A (tensor has no buffer)";
    if (t_ids_graph->buffer) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t_ids_graph->buffer);
        if (buft) {
            t_ids_buft_name = ggml_backend_buft_name(buft);
        }
    }
    fprintf(stderr, "[CLIP_TEXT_ENCODE_DEBUG] t_ids_graph (name: %s) data set. Buffer type: %s\n", t_ids_graph->name,
            t_ids_buft_name);

    const char * p_ids_buft_name = "N/A (tensor has no buffer)";
    if (p_ids_graph->buffer) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(p_ids_graph->buffer);
        if (buft) {
            p_ids_buft_name = ggml_backend_buft_name(buft);
        }
    }
    fprintf(stderr, "[CLIP_TEXT_ENCODE_DEBUG] p_ids_graph (name: %s) data set. Buffer type: %s\n", p_ids_graph->name,
            p_ids_buft_name);

    if (n_tokens > 0) {
        fprintf(stderr, "[CLIP_TEXT_ENCODE_DEBUG] tokens_input_arg[0] = %d\n", tokens_input_arg[0]);
        fprintf(stderr, "[CLIP_TEXT_ENCODE_DEBUG] position_ids_vec[0] = %d\n", position_ids_vec[0]);
    }

    auto status = ggml_backend_sched_graph_compute(ctx->sched.get(), temp_text_gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
        return false;
    }

    ggml_backend_sched_synchronize(ctx->sched.get());
    ggml_tensor * out = ggml_graph_node(temp_text_gf, -1);

    if (!out) {
        fprintf(stderr, "%s: Failed to get output node from graph.\n", __func__);
        return false;
    }
    const int64_t ne0 = out->ne[0];
    const int64_t ne1 = out->ne[1];
    if (ne0 != dim_out_actual || ne1 != 1) {
        fprintf(stderr, "%s: Output tensor shape mismatch. Expected [%d, 1], Got [%lld, %lld].\n", __func__,
                dim_out_actual, (long long) ne0, (long long) ne1);
        return false;
    }

    bool has_valid_output_buffer = false;
    if (out->buffer) {
        if (ggml_backend_buffer_get_base(out->buffer) != nullptr) {
            has_valid_output_buffer = true;
        }
    }
    if (!out->data && !has_valid_output_buffer) {
        fprintf(stderr, "%s: Output tensor data (and buffer base) is NULL.\n", __func__);
        return false;
    }

    ggml_backend_tensor_get(out, vec_out, 0, ggml_nbytes(out));

    // No ggml_backend_sched_free(ctx->sched.get()) here.
    // temp_text_ctx0_ptr and its graph temp_text_gf are freed when temp_text_ctx0_ptr goes out of scope.

    return true;
}
