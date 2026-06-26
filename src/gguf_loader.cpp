#include "gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace qwen3_tts {

GGUFLoader::GGUFLoader() = default;

GGUFLoader::~GGUFLoader() {
    close();
}

ggml_backend_t init_preferred_backend(const char * component_name, std::string * error_msg) {
    if (error_msg) error_msg->clear();

    std::string dev_str;
    const char * env_val = nullptr;

    if (component_name) {
        if (strcmp(component_name, "TTSTransformer") == 0) {
            env_val = std::getenv("QWEN3_TTS_TRANSFORMER_DEVICE");
        } else if (strcmp(component_name, "AudioTokenizerDecoder") == 0) {
            env_val = std::getenv("QWEN3_TTS_DECODER_DEVICE");
        } else if (strcmp(component_name, "AudioTokenizerEncoder") == 0) {
            env_val = std::getenv("QWEN3_TTS_ENCODER_DEVICE");
        }
    }

    if (!env_val || env_val[0] == '\0') {
        env_val = std::getenv("QWEN3_TTS_DEVICE");
    }

    if (env_val && env_val[0] != '\0') {
        dev_str = env_val;
    }

    if (!dev_str.empty()) {
        fprintf(stderr, "[Backend Selection] Component '%s' requested device string: '%s'\n",
                component_name ? component_name : "unknown", dev_str.c_str());
    }

    auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        return s;
    };

    std::string dev_str_lower = to_lower(dev_str);

    if (dev_str_lower == "cpu") {
        ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend && error_msg) {
            *error_msg = "Failed to initialize CPU backend";
        }
        return backend;
    }

    if (!dev_str.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_str.c_str());
        if (dev) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) {
                return backend;
            }
        }

        bool is_digit = !dev_str.empty() && std::all_of(dev_str.begin(), dev_str.end(), ::isdigit);
        if (is_digit) {
            int idx = std::stoi(dev_str);

            std::string vk_name = "Vulkan" + dev_str;
            dev = ggml_backend_dev_by_name(vk_name.c_str());
            if (dev) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend) return backend;
            }

            std::string cuda_name = "CUDA" + dev_str;
            dev = ggml_backend_dev_by_name(cuda_name.c_str());
            if (dev) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend) return backend;
            }

            std::string hip_name = "HIP" + dev_str;
            dev = ggml_backend_dev_by_name(hip_name.c_str());
            if (dev) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend) return backend;
            }

            int non_cpu_count = 0;
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                    if (non_cpu_count == idx) {
                        ggml_backend_t backend = ggml_backend_dev_init(d, nullptr);
                        if (backend) return backend;
                    }
                    non_cpu_count++;
                }
            }
        }

        fprintf(stderr, "[Backend Selection Warning] Requested device '%s' was not found or failed to initialize. Falling back to default order.\n", dev_str.c_str());
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, nullptr);
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    if (!backend && error_msg) {
        const char * name = component_name ? component_name : "component";
        *error_msg = "Failed to initialize backend (IGPU/GPU/ACCEL/CPU) for " + std::string(name);
    }

    return backend;
}

void release_preferred_backend(ggml_backend_t backend) {
    if (backend) {
        ggml_backend_free(backend);
    }
}

bool GGUFLoader::open(const std::string & path) {
    close();  // Close any previously opened file
    
    file_path_ = path;
    
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx_,
    };
    
    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!ctx_) {
        error_msg_ = "Failed to open GGUF file: " + path;
        return false;
    }
    
    return true;
}

void GGUFLoader::close() {
    if (ctx_) {
        gguf_free(ctx_);
        ctx_ = nullptr;
    }
    if (meta_ctx_) {
        ggml_free(meta_ctx_);
        meta_ctx_ = nullptr;
    }
    file_path_.clear();
}

int64_t GGUFLoader::get_n_tensors() const {
    if (!ctx_) return 0;
    return gguf_get_n_tensors(ctx_);
}

const char * GGUFLoader::get_tensor_name(int64_t idx) const {
    if (!ctx_) return nullptr;
    return gguf_get_tensor_name(ctx_, idx);
}

enum ggml_type GGUFLoader::get_tensor_type(int64_t idx) const {
    if (!ctx_) return GGML_TYPE_F32;
    return gguf_get_tensor_type(ctx_, idx);
}

size_t GGUFLoader::get_tensor_offset(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_offset(ctx_, idx);
}

size_t GGUFLoader::get_tensor_size(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_size(ctx_, idx);
}

int32_t GGUFLoader::get_u32(const char * key, int32_t default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return (int32_t)gguf_get_val_u32(ctx_, idx);
}

float GGUFLoader::get_f32(const char * key, float default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return gguf_get_val_f32(ctx_, idx);
}

size_t GGUFLoader::get_data_offset() const {
    if (!ctx_) return 0;
    return gguf_get_data_offset(ctx_);
}

bool load_tensor_data_from_file(
    const std::string & path,
    struct gguf_context * ctx,
    struct ggml_context * model_ctx,
    const std::map<std::string, struct ggml_tensor *> & tensors,
    ggml_backend_buffer_t & buffer,
    std::string & error_msg,
    ggml_backend_t backend
) {
    if (!backend) {
        error_msg = "Null backend passed to load_tensor_data_from_file";
        return false;
    }
    
    // Allocate buffer for all tensors
    buffer = ggml_backend_alloc_ctx_tensors(model_ctx, backend);
    if (!buffer) {
        error_msg = "Failed to allocate tensor buffer";
        return false;
    }
    
    // Open file for reading tensor data
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg = "Failed to open file for reading: " + path;
        return false;
    }
    
    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);
        
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            continue;  // Skip tensors not in our map
        }
        
        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);
        
        read_buf.resize(nbytes);
        
        if (fseek(f, data_offset + offset, SEEK_SET) != 0) {
            error_msg = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            return false;
        }
        
        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            error_msg = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            return false;
        }
        
        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }
    
    fclose(f);
    
    return true;
}

void free_ggml_resources(struct ggml_context * ctx, ggml_backend_buffer_t buffer) {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

} // namespace qwen3_tts
