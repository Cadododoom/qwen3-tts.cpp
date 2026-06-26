#include "qwen3_tts.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s [options] -m <model_dir> -t <text>\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <dir>      Model directory (required)\n");
    fprintf(stderr, "  --model-name <name>    Specific GGUF model filename in model directory\n");
    fprintf(stderr, "  -t, --text <text>      Text to synthesize (required). Use 'extract' to just dump embedding.\n");
    fprintf(stderr, "  --instruction <text>   Descriptive instruction prompt for VoiceDesign model\n");
    fprintf(stderr, "  -o, --output <file>    Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -r, --reference <file> Reference audio for voice cloning\n");
    fprintf(stderr, "  --dump-speaker-embedding <file> Dump extracted speaker embedding to JSON file\n");
    fprintf(stderr, "  --speaker-embedding <file> Load pre-computed speaker embedding from JSON file\n");
    fprintf(stderr, "  --temperature <val>    Sampling temperature (default: 0.9, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>            Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <val>          Top-p sampling (default: 1.0)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum audio tokens (default: 4096)\n");
    fprintf(stderr, "  --repetition-penalty <val> Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  -l, --language <lang>  Language: en,ru,zh,ja,ko,de,fr,es (default: en)\n");
    fprintf(stderr, "  -j, --threads <n>      Number of threads (default: 4)\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello, world!\" -o hello.wav\n", program);
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" -r reference.wav -o cloned.wav\n", program);
}

int main(int argc, char ** argv) {
    std::string model_dir;
    std::string model_name;
    std::string text;
    std::string instruction;
    std::string output_file = "output.wav";
    std::string reference_audio;
    std::string dump_speaker_embedding;
    std::string speaker_embedding_file;
    
    qwen3_tts::tts_params params;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing model directory\n");
                return 1;
            }
            model_dir = argv[i];
        } else if (arg == "--model-name") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing model name\n");
                return 1;
            }
            model_name = argv[i];
        } else if (arg == "-t" || arg == "--text") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing text\n");
                return 1;
            }
            text = argv[i];
        } else if (arg == "--instruction") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing instruction text\n");
                return 1;
            }
            instruction = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing output file\n");
                return 1;
            }
            output_file = argv[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing reference audio\n");
                return 1;
            }
            reference_audio = argv[i];
        } else if (arg == "--dump-speaker-embedding") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing dump speaker embedding file\n");
                return 1;
            }
            dump_speaker_embedding = argv[i];
        } else if (arg == "--speaker-embedding") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing speaker embedding file\n");
                return 1;
            }
            speaker_embedding_file = argv[i];
        } else if (arg == "--temperature") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing temperature value\n");
                return 1;
            }
            params.temperature = std::stof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing top-k value\n");
                return 1;
            }
            params.top_k = std::stoi(argv[i]);
        } else if (arg == "--top-p") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing top-p value\n");
                return 1;
            }
            params.top_p = std::stof(argv[i]);
        } else if (arg == "--max-tokens") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing max-tokens value\n");
                return 1;
            }
            params.max_audio_tokens = std::stoi(argv[i]);
        } else if (arg == "--repetition-penalty") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing repetition-penalty value\n");
                return 1;
            }
            params.repetition_penalty = std::stof(argv[i]);
        } else if (arg == "-l" || arg == "--language") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing language value\n");
                return 1;
            }
            std::string lang = argv[i];
            if (lang == "en" || lang == "english")       params.language_id = 2050;
            else if (lang == "ru" || lang == "russian")  params.language_id = 2069;
            else if (lang == "zh" || lang == "chinese")  params.language_id = 2055;
            else if (lang == "ja" || lang == "japanese")  params.language_id = 2058;
            else if (lang == "ko" || lang == "korean")   params.language_id = 2064;
            else if (lang == "de" || lang == "german")   params.language_id = 2053;
            else if (lang == "fr" || lang == "french")   params.language_id = 2061;
            else if (lang == "es" || lang == "spanish")  params.language_id = 2054;
            else if (lang == "it" || lang == "italian")  params.language_id = 2070;
            else if (lang == "pt" || lang == "portuguese") params.language_id = 2071;
            else {
                fprintf(stderr, "Error: unknown language '%s'. Supported: en,ru,zh,ja,ko,de,fr,es,it,pt\n", lang.c_str());
                return 1;
            }
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing threads value\n");
                return 1;
            }
            params.n_threads = std::stoi(argv[i]);
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate required arguments
    if (model_dir.empty()) {
        fprintf(stderr, "Error: model directory is required\n");
        print_usage(argv[0]);
        return 1;
    }
    
    if (text.empty()) {
        fprintf(stderr, "Error: text is required\n");
        print_usage(argv[0]);
        return 1;
    }

    params.instruction = instruction;
    
    // Initialize TTS
    qwen3_tts::Qwen3TTS tts;
    
    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    if (!tts.load_models(model_dir, model_name)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }
    
    // Set progress callback
    tts.set_progress_callback([](int tokens, int max_tokens) {
        fprintf(stderr, "\rGenerating: %d/%d tokens", tokens, max_tokens);
    });

    // Load precomputed speaker embedding if specified
    std::vector<float> loaded_embedding;
    if (!speaker_embedding_file.empty()) {
        std::ifstream in(speaker_embedding_file);
        if (!in) {
            fprintf(stderr, "Error: failed to open speaker embedding file: %s\n", speaker_embedding_file.c_str());
            return 1;
        }
        std::string str((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        size_t pos = 0;
        while (pos < str.size()) {
            if (isdigit(str[pos]) || str[pos] == '-' || str[pos] == '.' || str[pos] == 'e' || str[pos] == 'E') {
                size_t next_pos;
                try {
                    float val = std::stof(str.substr(pos), &next_pos);
                    loaded_embedding.push_back(val);
                    pos += next_pos;
                } catch (...) {
                    pos++;
                }
            } else {
                pos++;
            }
        }
        if (loaded_embedding.empty()) {
            fprintf(stderr, "Error: failed to parse speaker embedding from: %s\n", speaker_embedding_file.c_str());
            return 1;
        }
        fprintf(stderr, "Loaded speaker embedding: %zu floats\n", loaded_embedding.size());
    }

    // Extract speaker embedding from reference audio if specified
    std::vector<float> speaker_emb;
    if (!reference_audio.empty()) {
        std::vector<float> ref_samples;
        int ref_sample_rate;
        if (!qwen3_tts::load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
            fprintf(stderr, "Error: failed to load reference audio: %s\n", reference_audio.c_str());
            return 1;
        }
        const int target_rate = 24000;
        if (ref_sample_rate != target_rate) {
            fprintf(stderr, "Resampling reference audio from %d Hz to %d Hz...\n", ref_sample_rate, target_rate);
            std::vector<float> resampled;
            qwen3_tts::resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sample_rate, resampled, target_rate);
            ref_samples = std::move(resampled);
        }
        if (!tts.extract_speaker_embedding(ref_samples.data(), (int32_t)ref_samples.size(), speaker_emb, params)) {
            fprintf(stderr, "Error: failed to extract speaker embedding: %s\n", tts.get_error().c_str());
            return 1;
        }
        if (!dump_speaker_embedding.empty()) {
            std::ofstream out(dump_speaker_embedding);
            if (!out) {
                fprintf(stderr, "Error: failed to open file for dumping speaker embedding: %s\n", dump_speaker_embedding.c_str());
                return 1;
            }
            out << "[";
            for (size_t i = 0; i < speaker_emb.size(); ++i) {
                out << speaker_emb[i];
                if (i + 1 < speaker_emb.size()) out << ",";
            }
            out << "]";
            fprintf(stderr, "Dumped speaker embedding to: %s\n", dump_speaker_embedding.c_str());
        }
    }

    // If only extracting embedding, exit successfully now
    if (text == "extract") {
        fprintf(stderr, "Speaker embedding extraction complete.\n");
        return 0;
    }
    
    // Generate speech
    qwen3_tts::tts_result result;
    
    if (!loaded_embedding.empty()) {
        fprintf(stderr, "Synthesizing with loaded speaker embedding: \"%s\"\n", text.c_str());
        result = tts.synthesize_with_embedding(text, loaded_embedding.data(), (int32_t)loaded_embedding.size(), params);
    } else if (!speaker_emb.empty()) {
        fprintf(stderr, "Synthesizing with extracted speaker embedding: \"%s\"\n", text.c_str());
        result = tts.synthesize_with_embedding(text, speaker_emb.data(), (int32_t)speaker_emb.size(), params);
    } else {
        fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
        result = tts.synthesize(text, params);
    }
    
    if (!result.success) {
        fprintf(stderr, "\nError: %s\n", result.error_msg.c_str());
        return 1;
    }
    
    fprintf(stderr, "\n");
    
    // Save output
    if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
        fprintf(stderr, "Error: failed to save output file: %s\n", output_file.c_str());
        return 1;
    }
    
    fprintf(stderr, "Output saved to: %s\n", output_file.c_str());
    fprintf(stderr, "Audio duration: %.2f seconds\n", 
            (float)result.audio.size() / result.sample_rate);
    
    // Print timing
    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Load:      %6lld ms\n", (long long)result.t_load_ms);
        fprintf(stderr, "  Tokenize:  %6lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Encode:    %6lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Generate:  %6lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Decode:    %6lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:     %6lld ms\n", (long long)result.t_total_ms);
    }
    
    return 0;
}
