#include "mmq_bundle.h"
#include "generated/mmq_bundle_table.cuh"

#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace torch_ggml_ops::mmq_bundle {
namespace {

constexpr int kQuantQ8_0 = 8;
constexpr int kQuantQ2_K = 10;
constexpr int kQuantQ3_K = 11;
constexpr int kQuantQ4_K = 12;
constexpr int kQuantQ5_K = 13;
constexpr int kQuantQ6_K = 14;
constexpr int kQuantIQ2_XXS = 16;
constexpr int kQuantIQ2_S = 22;
constexpr int kForwardTileI = 64;
constexpr int kForwardTileYK = 36;
constexpr int kForwardThreads = 128;
constexpr int kForwardBlockX = 32;
constexpr int kForwardBlockY = 4;
constexpr int kBackwardThreads = 128;
constexpr int kBackwardWaves = 4;
constexpr int kBackwardTile = 16;
constexpr int kGroupedBackwardThreads = 256;
constexpr int kGroupedBackwardN = 16;
constexpr int kGroupedBackwardTiledN = 128;

struct LoadedKernel {
    std::vector<std::uint8_t> image;
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
};

std::mutex loaded_kernels_mutex;
std::map<std::pair<int, MMQKernelId>, std::unique_ptr<LoadedKernel>> loaded_kernels;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("MMQ gfx1151 bundle: " + message);
}

void check_hip(hipError_t status, const std::string & operation) {
    if (status != hipSuccess) {
        fail(operation + " failed: " + hipGetErrorString(status));
    }
}

void bundle_path_anchor() {}

std::filesystem::path bundle_directory() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&bundle_path_anchor), &info) == 0 ||
        info.dli_fname == nullptr) {
        fail("dladdr could not locate the extension shared object");
    }
    return std::filesystem::path(info.dli_fname).parent_path() /
        "kernels" / "gfx1151";
}

std::vector<std::uint8_t> read_artifact(const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        fail("cannot open kernel artifact " + path.string());
    }
    const std::streamoff end = file.tellg();
    if (end < 0) {
        fail("cannot determine kernel artifact size " + path.string());
    }
    std::vector<std::uint8_t> image(static_cast<std::size_t>(end));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char *>(image.data()), end)) {
        fail("cannot read kernel artifact " + path.string());
    }
    return image;
}

LoadedKernel & resolve_kernel(MMQKernelId id) {
    int device = -1;
    check_hip(hipGetDevice(&device), "hipGetDevice");

    const auto key = std::make_pair(device, id);
    std::lock_guard<std::mutex> lock(loaded_kernels_mutex);
    const auto found = loaded_kernels.find(key);
    if (found != loaded_kernels.end()) {
        return *found->second;
    }

    const MMQKernelSpec & spec = mmq_kernel_spec(id);
    const std::filesystem::path path = bundle_directory() / spec.filename;
    auto loaded = std::make_unique<LoadedKernel>();
    loaded->image = read_artifact(path);
    const hipError_t load_status = hipModuleLoadData(
        &loaded->module, loaded->image.data());
    if (load_status != hipSuccess) {
        fail(
            "hipModuleLoadData failed for " + path.string() + " (" +
            spec.symbol + "): " + hipGetErrorString(load_status));
    }
    const hipError_t function_status = hipModuleGetFunction(
        &loaded->function, loaded->module, spec.symbol);
    if (function_status != hipSuccess) {
        (void)hipModuleUnload(loaded->module);
        loaded->module = nullptr;
        fail(
            "hipModuleGetFunction failed for symbol " + std::string(spec.symbol) +
            " in " + path.string() + ": " + hipGetErrorString(function_status));
    }

    LoadedKernel & result = *loaded;
    loaded_kernels.emplace(key, std::move(loaded));
    return result;
}

void launch(
        MMQKernelId id,
        unsigned int grid_x,
        unsigned int grid_y,
        unsigned int grid_z,
        unsigned int block_x,
        unsigned int block_y,
        unsigned int block_z,
        unsigned int shared_memory,
        hipStream_t stream,
        void ** arguments) {
    const MMQKernelSpec & spec = mmq_kernel_spec(id);
    LoadedKernel & loaded = resolve_kernel(id);
    const hipError_t status = hipModuleLaunchKernel(
        loaded.function,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        shared_memory,
        stream,
        arguments,
        nullptr);
    if (status != hipSuccess) {
        fail(
            "hipModuleLaunchKernel failed for " + std::string(spec.symbol) +
            ": " + hipGetErrorString(status));
    }
}

int forward_quant_index(std::int32_t quant_type) {
    switch (quant_type) {
        case kQuantQ8_0: return 0;
        case kQuantQ2_K: return 1;
        case kQuantQ3_K: return 2;
        case kQuantQ4_K: return 3;
        case kQuantQ5_K: return 4;
        case kQuantQ6_K: return 5;
        case kQuantIQ2_XXS: return 6;
        case kQuantIQ2_S: return 7;
        default: fail("unsupported forward quant_type " + std::to_string(quant_type));
    }
}

int backward_quant_index(std::int32_t quant_type) {
    switch (quant_type) {
        case kQuantQ3_K: return 0;
        case kQuantQ4_K: return 1;
        case kQuantQ5_K: return 2;
        case kQuantQ6_K: return 3;
        case kQuantIQ2_S: return 4;
        default: fail("unsupported quant_type " + std::to_string(quant_type) +
            " for backward MMQ");
    }
}

MMQKernelId quantize_kernel(std::int32_t quant_type) {
    if (quant_type == kQuantQ2_K) {
        return MMQKernelId::QuantizeQ81D2S6;
    }
    if (quant_type == kQuantQ4_K || quant_type == kQuantQ5_K) {
        return MMQKernelId::QuantizeQ81DS4;
    }
    (void)forward_quant_index(quant_type);
    return MMQKernelId::QuantizeQ81D4;
}

MMQKernelId dense_forward_kernel(std::int32_t quant_type, int rows_padded) {
    if (quant_type == kQuantQ6_K && rows_padded == 64) {
        return MMQKernelId::DenseFwdQ6KJ64;
    }
    constexpr std::array ids{
        MMQKernelId::DenseFwdQ80J128,
        MMQKernelId::DenseFwdQ2KJ128,
        MMQKernelId::DenseFwdQ3KJ128,
        MMQKernelId::DenseFwdQ4KJ128,
        MMQKernelId::DenseFwdQ5KJ128,
        MMQKernelId::DenseFwdQ6KJ128,
        MMQKernelId::DenseFwdIQ2XXSJ128,
        MMQKernelId::DenseFwdIQ2SJ128,
    };
    return ids[forward_quant_index(quant_type)];
}

int forward_j(MMQKernelId id) {
    return id == MMQKernelId::DenseFwdQ6KJ64 ? 64 : 128;
}

int padded(int value, int alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

int forward_sram_stride(std::int32_t quant_type) {
    switch (quant_type) {
        case kQuantQ2_K: return 100;
        case kQuantQ3_K:
        case kQuantIQ2_S: return 84;
        case kQuantQ8_0:
        case kQuantQ4_K:
        case kQuantQ5_K:
        case kQuantQ6_K:
        case kQuantIQ2_XXS: return 76;
        default: fail("unsupported forward quant_type " + std::to_string(quant_type));
    }
}

unsigned int forward_shared_bytes(int j, std::int32_t quant_type) {
    const int shared_ints = j + padded(j * kForwardTileYK, kForwardThreads) +
        kForwardTileI * forward_sram_stride(quant_type);
    return static_cast<unsigned int>(shared_ints * sizeof(int));
}

struct GroupedForwardSelection {
    MMQKernelId id;
    int j;
    int out_features;
    int blocks_per_weight_row;
};

MMQKernelId grouped_forward_generic_kernel(std::int32_t quant_type) {
    constexpr std::array ids{
        MMQKernelId::GroupedFwdSerialQ80GenericJ128,
        MMQKernelId::GroupedFwdSerialQ2KGenericJ128,
        MMQKernelId::GroupedFwdSerialQ3KGenericJ128,
        MMQKernelId::GroupedFwdSerialQ4KGenericJ128,
        MMQKernelId::GroupedFwdSerialQ5KGenericJ128,
        MMQKernelId::GroupedFwdSerialQ6KGenericJ128,
        MMQKernelId::GroupedFwdSerialIQ2XXSGenericJ128,
        MMQKernelId::GroupedFwdSerialIQ2SGenericJ128,
    };
    return ids[forward_quant_index(quant_type)];
}

MMQKernelId grouped_forward_n512_k2048_kernel(std::int32_t quant_type) {
    constexpr std::array ids{
        MMQKernelId::GroupedFwdSerialQ80N512K2048J64,
        MMQKernelId::GroupedFwdSerialQ2KN512K2048J64,
        MMQKernelId::GroupedFwdSerialQ3KN512K2048J64,
        MMQKernelId::GroupedFwdSerialQ4KN512K2048J64,
        MMQKernelId::GroupedFwdSerialQ5KN512K2048J64,
        MMQKernelId::GroupedFwdSerialQ6KN512K2048J64,
        MMQKernelId::GroupedFwdSerialIQ2XXSN512K2048J64,
        MMQKernelId::GroupedFwdSerialIQ2SN512K2048J64,
    };
    return ids[forward_quant_index(quant_type)];
}

MMQKernelId grouped_forward_n2048_k512_kernel(std::int32_t quant_type) {
    constexpr std::array ids{
        MMQKernelId::GroupedFwdSerialQ80N2048K512J64,
        MMQKernelId::GroupedFwdSerialQ2KN2048K512J64,
        MMQKernelId::GroupedFwdSerialQ3KN2048K512J64,
        MMQKernelId::GroupedFwdSerialQ4KN2048K512J64,
        MMQKernelId::GroupedFwdSerialQ5KN2048K512J64,
        MMQKernelId::GroupedFwdSerialQ6KN2048K512J64,
        MMQKernelId::GroupedFwdSerialIQ2XXSN2048K512J64,
        MMQKernelId::GroupedFwdSerialIQ2SN2048K512J64,
    };
    return ids[forward_quant_index(quant_type)];
}

GroupedForwardSelection grouped_forward_selection(
        std::int32_t quant_type,
        int rows,
        int num_groups,
        int in_features,
        int out_features) {
    if (out_features == 512 && in_features == 2048) {
        return {
            grouped_forward_n512_k2048_kernel(quant_type), 64, 512, 8};
    }
    if (out_features == 2048 && in_features == 512) {
        if (quant_type == kQuantIQ2_S && rows < num_groups * 128) {
            return {
                MMQKernelId::GroupedFwdSerialIQ2SN2048K512J64J32,
                64,
                2048,
                2};
        }
        if (quant_type == kQuantQ5_K && rows < num_groups * 128) {
            return {
                MMQKernelId::GroupedFwdSerialQ5KN2048K512J32,
                32,
                2048,
                2};
        }
        return {
            grouped_forward_n2048_k512_kernel(quant_type), 64, 2048, 2};
    }
    if (quant_type == kQuantIQ2_XXS &&
        out_features == 2048 && in_features == 4096) {
        if (rows >= num_groups * 512) {
            return {
                MMQKernelId::GroupedFwdSerialIQ2XXSN2048K4096J80,
                80,
                2048,
                16};
        }
        return {
            MMQKernelId::GroupedFwdSerialIQ2XXSN2048K4096J64,
            64,
            2048,
            16};
    }
    if (quant_type == kQuantQ2_K &&
        out_features == 4096 && in_features == 2048) {
        const MMQKernelId id = rows < num_groups * 64
            ? MMQKernelId::GroupedFwdSerialQ2KN4096K2048J32J16
            : MMQKernelId::GroupedFwdSerialQ2KN4096K2048J32;
        return {id, 32, 4096, 8};
    }
    return {
        grouped_forward_generic_kernel(quant_type),
        128,
        out_features,
        in_features / 256};
}

MMQKernelId grouped_forward_row_task_kernel(std::int32_t quant_type) {
    switch (quant_type) {
        case kQuantQ3_K:
            return MMQKernelId::GroupedFwdRowTaskQ3KN512K2048J64;
        case kQuantQ4_K:
            return MMQKernelId::GroupedFwdRowTaskQ4KN512K2048J64;
        case kQuantQ5_K:
            return MMQKernelId::GroupedFwdRowTaskQ5KN512K2048J64;
        case kQuantQ6_K:
            return MMQKernelId::GroupedFwdRowTaskQ6KN512K2048J64;
        case kQuantIQ2_S:
            return MMQKernelId::GroupedFwdRowTaskIQ2SN512K2048J64;
        default:
            fail("unsupported grouped forward row-task quant_type " +
                std::to_string(quant_type));
    }
}

struct DenseBackwardSelection {
    MMQKernelId id;
    int n_tiles;
    int k_iteration;
    int group_m;
    int m_tiles_per_wave;
};

DenseBackwardSelection dense_backward_generic(
        std::int32_t quant_type,
        int n_tiles,
        int group_m) {
#define MMQ_DENSE_BWD_GENERIC_CASE(TYPE, PREFIX) \
    case TYPE: \
        if (group_m == 0) { \
            if (n_tiles == 1) return {MMQKernelId::DenseBwd##PREFIX##NT16KI16G0, 1, 16, 0, 1}; \
            if (n_tiles == 4) return {MMQKernelId::DenseBwd##PREFIX##NT64KI16G0, 4, 16, 0, 1}; \
        } else { \
            if (n_tiles == 4) return {MMQKernelId::DenseBwd##PREFIX##NT64KI16G2, 4, 16, 2, 1}; \
            if (n_tiles == 8) return {MMQKernelId::DenseBwd##PREFIX##NT128KI16G2, 8, 16, 2, 1}; \
            if (n_tiles == 12) return {MMQKernelId::DenseBwd##PREFIX##NT192KI16G2, 12, 16, 2, 1}; \
            if (n_tiles == 16) return {MMQKernelId::DenseBwd##PREFIX##NT256KI16G2, 16, 16, 2, 1}; \
        } \
        break

    switch (quant_type) {
        MMQ_DENSE_BWD_GENERIC_CASE(kQuantQ3_K, Q3K);
        case kQuantIQ2_S:
            if (group_m == 0) {
                if (n_tiles == 1) return {MMQKernelId::DenseBwdIQ2SNT16KI16G0, 1, 16, 0, 1};
                if (n_tiles == 4) return {MMQKernelId::DenseBwdIQ2SNT64KI16G0, 4, 16, 0, 1};
            } else {
                if (n_tiles == 4) return {MMQKernelId::DenseBwdIQ2SNT64KI16G2, 4, 16, 2, 1};
                if (n_tiles == 12) return {MMQKernelId::DenseBwdIQ2SNT192KI16G2, 12, 16, 2, 1};
                if (n_tiles == 16) return {MMQKernelId::DenseBwdIQ2SNT256KI16G2, 16, 16, 2, 1};
            }
            break;
        case kQuantQ4_K:
            if (group_m == 0) {
                if (n_tiles == 1) return {MMQKernelId::DenseBwdQ4KNT16KI16G0, 1, 16, 0, 1};
                if (n_tiles == 4) return {MMQKernelId::DenseBwdQ4KNT64KI16G0, 4, 16, 0, 1};
            } else {
                if (n_tiles == 8) return {MMQKernelId::DenseBwdQ4KNT128KI16G2, 8, 16, 2, 1};
                if (n_tiles == 12) return {MMQKernelId::DenseBwdQ4KNT192KI16G2, 12, 16, 2, 1};
                if (n_tiles == 16) return {MMQKernelId::DenseBwdQ4KNT256KI16G2, 16, 16, 2, 1};
            }
            break;
        case kQuantQ5_K:
            if (group_m == 0) {
                if (n_tiles == 1) return {MMQKernelId::DenseBwdQ5KNT16KI16G0, 1, 16, 0, 1};
                if (n_tiles == 4) return {MMQKernelId::DenseBwdQ5KNT64KI16G0, 4, 16, 0, 1};
            } else {
                if (n_tiles == 8) return {MMQKernelId::DenseBwdQ5KNT128KI16G2, 8, 16, 2, 1};
                if (n_tiles == 12) return {MMQKernelId::DenseBwdQ5KNT192KI16G2, 12, 16, 2, 1};
                if (n_tiles == 16) return {MMQKernelId::DenseBwdQ5KNT256KI16G2, 16, 16, 2, 1};
            }
            break;
        default:
            break;
    }
#undef MMQ_DENSE_BWD_GENERIC_CASE
    fail("unsupported dense backward generic specialization");
}

DenseBackwardSelection dense_backward_selection(
        std::int32_t quant_type,
        int rows,
        int out_features,
        int in_features) {
    if (quant_type == kQuantQ3_K || quant_type == kQuantQ4_K ||
        quant_type == kQuantQ5_K) {
        const bool full_tiles = rows > 256 && rows % 128 == 0 &&
            out_features % 32 == 0 && in_features % 128 == 0;
        if (full_tiles) {
            if (quant_type == kQuantQ3_K) {
                return {
                    out_features >= 2048
                        ? MMQKernelId::DenseBwdQ3KFullWide
                        : MMQKernelId::DenseBwdQ3KFullNarrow,
                    8, 32, 1, 2};
            }
            if (quant_type == kQuantQ4_K) {
                const MMQKernelId id = in_features >= 4096
                    ? MMQKernelId::DenseBwdQ4KFullK4096
                    : in_features >= 2048
                        ? MMQKernelId::DenseBwdQ4KFullK2048
                        : MMQKernelId::DenseBwdQ4KFullK512;
                return {id, 8, 32, 1, 2};
            }
            return {
                in_features >= 2048
                    ? MMQKernelId::DenseBwdQ5KFullK2048
                    : MMQKernelId::DenseBwdQ5KFullK512,
                8, 32, 1, 2};
        }
    }
    if (quant_type == kQuantQ6_K) {
        if (rows <= 64) {
            return {
                rows == 64 && out_features % 64 == 0 && in_features % 32 == 0
                    ? MMQKernelId::DenseBwdQ6KM64Full
                    : MMQKernelId::DenseBwdQ6KM64Bounded,
                2, 64, 0, 1};
        }
        if (rows <= 128) {
            return {
                rows == 128 && out_features % 32 == 0 && in_features % 64 == 0
                    ? MMQKernelId::DenseBwdQ6KM128Full
                    : MMQKernelId::DenseBwdQ6KM128Bounded,
                4, 32, 0, 2};
        }
        if (rows <= 256) {
            return {
                rows == 256 && out_features % 32 == 0 && in_features % 64 == 0
                    ? MMQKernelId::DenseBwdQ6KM256Full
                    : MMQKernelId::DenseBwdQ6KM256Bounded,
                4, 32, 0, 2};
        }
        if (rows <= 2048) {
            return {MMQKernelId::DenseBwdQ6KNT128KI16G2, 8, 16, 2, 1};
        }
        return {MMQKernelId::DenseBwdQ6KNT256KI16G2, 16, 16, 2, 1};
    }
    (void)backward_quant_index(quant_type);
    if (rows <= 128) {
        return dense_backward_generic(quant_type, 1, 0);
    }
    if (rows <= 256) {
        return dense_backward_generic(quant_type, 4, 0);
    }
    if (rows <= 2048) {
        if (quant_type == kQuantQ3_K) {
            return dense_backward_generic(
                quant_type, out_features <= 512 ? 8 : 4, 2);
        }
        if (quant_type == kQuantIQ2_S) {
            return dense_backward_generic(quant_type, 4, 2);
        }
        return dense_backward_generic(
            quant_type, in_features >= 4096 ? 16 : 8, 2);
    }
    if (rows <= 8192) {
        if (quant_type == kQuantQ4_K || quant_type == kQuantQ5_K) {
            return dense_backward_generic(
                quant_type, in_features == 2048 ? 16 : 12, 2);
        }
        return dense_backward_generic(quant_type, 12, 2);
    }
    return dense_backward_generic(quant_type, 16, 2);
}

MMQKernelId grouped_backward_generic_kernel(
        std::int32_t quant_type,
        bool pair) {
    constexpr std::array single_ids{
        MMQKernelId::GroupedBwdSingleQ3KGeneric,
        MMQKernelId::GroupedBwdSingleQ4KGeneric,
        MMQKernelId::GroupedBwdSingleQ5KGeneric,
        MMQKernelId::GroupedBwdSingleQ6KGeneric,
        MMQKernelId::GroupedBwdSingleIQ2SGeneric,
    };
    constexpr std::array pair_ids{
        MMQKernelId::GroupedBwdPairQ3KGeneric,
        MMQKernelId::GroupedBwdPairQ4KGeneric,
        MMQKernelId::GroupedBwdPairQ5KGeneric,
        MMQKernelId::GroupedBwdPairQ6KGeneric,
        MMQKernelId::GroupedBwdPairIQ2SGeneric,
    };
    const int index = backward_quant_index(quant_type);
    return pair ? pair_ids[index] : single_ids[index];
}

} // namespace

void launch_quantize(
        std::int32_t quant_type,
        const void * input,
        void * output,
        std::int64_t rows,
        std::int64_t rows_padded,
        std::int64_t in_features,
        hipStream_t stream) {
    const MMQKernelId id = quantize_kernel(quant_type);
    void * arguments[]{&input, &output, &rows, &rows_padded, &in_features};
    launch(
        id,
        static_cast<unsigned int>(rows), 1, 1,
        512, 1, 1,
        0,
        stream,
        arguments);
}

void launch_dense_forward(
        std::int32_t quant_type,
        const char * packed,
        const int * activations,
        void * output,
        int rows,
        int rows_padded,
        int in_features,
        int out_features,
        hipStream_t stream) {
    const MMQKernelId id = dense_forward_kernel(quant_type, rows_padded);
    const int j = forward_j(id);
    int blocks_per_weight_row = in_features / 256;
    void * arguments[]{
        &packed,
        &activations,
        &output,
        &out_features,
        &rows,
        &rows_padded,
        &blocks_per_weight_row,
    };
    launch(
        id,
        static_cast<unsigned int>((out_features + kForwardTileI - 1) /
            kForwardTileI),
        static_cast<unsigned int>(rows_padded / j),
        1,
        kForwardBlockX,
        kForwardBlockY,
        1,
        forward_shared_bytes(j, quant_type),
        stream,
        arguments);
}

void launch_grouped_row_task_setup(
        const std::int64_t * expert_indices,
        const std::int32_t * expert_offsets,
        std::int32_t * task_count,
        std::int32_t * task_experts,
        std::int32_t * task_row_starts,
        std::int32_t * task_row_ends,
        int num_experts,
        int num_groups,
        int rows,
        int row_tile,
        hipStream_t stream) {
    void * arguments[]{
        &expert_indices,
        &expert_offsets,
        &task_count,
        &task_experts,
        &task_row_starts,
        &task_row_ends,
        &num_experts,
        &num_groups,
        &rows,
        &row_tile,
    };
    launch(
        MMQKernelId::GroupedRowTaskSetup,
        1, 1, 1,
        256, 1, 1,
        0,
        stream,
        arguments);
}

void launch_fixed_grouped_forward(
        const char * packed,
        const int * activations,
        void * output,
        int tokens,
        int out_features,
        std::int64_t bytes_per_group,
        hipStream_t stream) {
    const MMQKernelId id = out_features % kForwardTileI == 0
        ? MMQKernelId::GroupedFwdFixedQ80G8K4096J64Full
        : MMQKernelId::GroupedFwdFixedQ80G8K4096J64Bounded;
    void * arguments[]{
        &packed,
        &activations,
        &output,
        &tokens,
        &out_features,
        &bytes_per_group,
    };
    launch(
        id,
        static_cast<unsigned int>((out_features + kForwardTileI - 1) /
            kForwardTileI),
        static_cast<unsigned int>((tokens + 63) / 64),
        8,
        kForwardBlockX,
        kForwardBlockY,
        1,
        forward_shared_bytes(64, kQuantQ8_0),
        stream,
        arguments);
}

void launch_grouped_forward(
        std::int32_t quant_type,
        const char * packed,
        const int * activations,
        void * output,
        const std::int64_t * expert_indices,
        const std::int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int in_features,
        int out_features,
        std::int64_t bytes_per_expert,
        hipStream_t stream) {
    const GroupedForwardSelection selection = grouped_forward_selection(
        quant_type, rows, num_groups, in_features, out_features);
    int kernel_out_features = selection.out_features;
    int kernel_blocks_per_weight_row = selection.blocks_per_weight_row;
    void * arguments[]{
        &packed,
        &activations,
        &output,
        &expert_indices,
        &expert_offsets,
        &num_experts,
        &kernel_out_features,
        &rows,
        &kernel_blocks_per_weight_row,
        &bytes_per_expert,
    };
    launch(
        selection.id,
        static_cast<unsigned int>((kernel_out_features + kForwardTileI - 1) /
            kForwardTileI),
        static_cast<unsigned int>(num_groups),
        1,
        kForwardBlockX,
        kForwardBlockY,
        1,
        forward_shared_bytes(selection.j, quant_type),
        stream,
        arguments);
}

void launch_grouped_forward_row_tasks(
        std::int32_t quant_type,
        const char * packed,
        const int * activations,
        void * output,
        const std::int32_t * task_count,
        const std::int32_t * task_experts,
        const std::int32_t * task_row_starts,
        const std::int32_t * task_row_ends,
        int max_tasks,
        int rows,
        std::int64_t bytes_per_expert,
        hipStream_t stream) {
    const MMQKernelId id = grouped_forward_row_task_kernel(quant_type);
    void * arguments[]{
        &packed,
        &activations,
        &output,
        &task_count,
        &task_experts,
        &task_row_starts,
        &task_row_ends,
        &rows,
        &bytes_per_expert,
    };
    launch(
        id,
        512 / kForwardTileI,
        static_cast<unsigned int>(max_tasks),
        1,
        kForwardBlockX,
        kForwardBlockY,
        1,
        forward_shared_bytes(64, quant_type),
        stream,
        arguments);
}

void launch_dense_backward(
        std::int32_t quant_type,
        const void * grad_output,
        const char * packed_weight,
        void * grad_input,
        int rows,
        int out_features,
        int in_features,
        hipStream_t stream) {
    const DenseBackwardSelection selection = dense_backward_selection(
        quant_type, rows, out_features, in_features);
    const int n_per_block = selection.n_tiles * kBackwardTile;
    const int m_per_block = selection.m_tiles_per_wave * kBackwardTile *
        kBackwardWaves;
    const int m_blocks = (rows + m_per_block - 1) / m_per_block;
    const int grid_group_m = selection.group_m > 0 &&
        selection.group_m < m_blocks
        ? selection.group_m
        : m_blocks;
    int blocks_per_weight_row = in_features / 256;
    void * arguments[]{
        &grad_output,
        &packed_weight,
        &grad_input,
        &rows,
        &out_features,
        &in_features,
        &blocks_per_weight_row,
    };
    launch(
        selection.id,
        static_cast<unsigned int>(grid_group_m),
        static_cast<unsigned int>((in_features + n_per_block - 1) /
            n_per_block),
        static_cast<unsigned int>(selection.group_m > 0
            ? (m_blocks + selection.group_m - 1) / selection.group_m
            : 1),
        kBackwardThreads,
        1,
        1,
        0,
        stream,
        arguments);
}

void launch_grouped_backward(
        std::int32_t quant_type,
        const void * grad_output,
        const char * packed_weight,
        void * grad_input,
        const std::int64_t * expert_indices,
        const std::int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int out_features,
        int in_features,
        std::int64_t bytes_per_expert,
        hipStream_t stream) {
    MMQKernelId id = grouped_backward_generic_kernel(quant_type, false);
    int n_per_block = kGroupedBackwardN;
    int threads = kGroupedBackwardThreads;
    if (out_features == 2048 && in_features == 512) {
        if (rows >= num_groups * 128 &&
            (quant_type == kQuantQ4_K || quant_type == kQuantQ5_K ||
             quant_type == kQuantIQ2_S)) {
            fail("large grouped backward down projection requires row tasks");
        }
        n_per_block = 64;
        threads = kBackwardThreads;
        if (quant_type == kQuantQ4_K) {
            id = rows >= num_groups * 80
                ? MMQKernelId::GroupedBwdSingleQ4KN2048K512M128N64
                : MMQKernelId::GroupedBwdSingleQ4KN2048K512M64N64;
        } else if (quant_type == kQuantQ5_K) {
            id = MMQKernelId::GroupedBwdSingleQ5KN2048K512M64N64;
        } else if (quant_type == kQuantIQ2_S) {
            id = rows >= num_groups * 80
                ? MMQKernelId::GroupedBwdSingleIQ2SN2048K512M128N64
                : MMQKernelId::GroupedBwdSingleIQ2SN2048K512M64N64;
        } else {
            n_per_block = kGroupedBackwardN;
            threads = kGroupedBackwardThreads;
        }
    }
    int blocks_per_weight_row = in_features / 256;
    void * arguments[]{
        &grad_output,
        &packed_weight,
        &grad_input,
        &expert_indices,
        &expert_offsets,
        &num_experts,
        &rows,
        &out_features,
        &in_features,
        &blocks_per_weight_row,
        &bytes_per_expert,
    };
    if (threads == kBackwardThreads) {
        void * specialized_arguments[]{
            &grad_output,
            &packed_weight,
            &grad_input,
            &expert_indices,
            &expert_offsets,
            &num_experts,
            &rows,
            &bytes_per_expert,
        };
        launch(
            id,
            static_cast<unsigned int>((in_features + n_per_block - 1) /
                n_per_block),
            static_cast<unsigned int>(num_groups),
            1,
            threads,
            1,
            1,
            0,
            stream,
            specialized_arguments);
        return;
    }
    launch(
        id,
        static_cast<unsigned int>((in_features + n_per_block - 1) /
            n_per_block),
        static_cast<unsigned int>(num_groups),
        1,
        threads,
        1,
        1,
        0,
        stream,
        arguments);
}

void launch_grouped_pair_backward(
        std::int32_t quant_type,
        const void * first_grad_output,
        const void * second_grad_output,
        const char * first_packed_weight,
        const char * second_packed_weight,
        void * grad_input,
        const std::int64_t * expert_indices,
        const std::int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int out_features,
        int in_features,
        std::int64_t bytes_per_expert,
        hipStream_t stream) {
    MMQKernelId id = grouped_backward_generic_kernel(quant_type, true);
    int n_per_block = kGroupedBackwardN;
    int threads = kGroupedBackwardThreads;
    bool specialized = false;
    if (out_features == 512 && in_features == 2048) {
        if (quant_type == kQuantQ3_K) {
            id = rows >= num_groups * 128
                ? MMQKernelId::GroupedBwdPairQ3KN512K2048M128N64
                : MMQKernelId::GroupedBwdPairQ3KN512K2048M64N64;
            specialized = true;
        } else if (quant_type == kQuantIQ2_S) {
            id = rows >= num_groups * 128
                ? MMQKernelId::GroupedBwdPairIQ2SN512K2048M128N64
                : MMQKernelId::GroupedBwdPairIQ2SN512K2048M64N64;
            specialized = true;
        }
    }
    if (specialized) {
        n_per_block = 64;
        threads = kBackwardThreads;
        void * arguments[]{
            &first_grad_output,
            &second_grad_output,
            &first_packed_weight,
            &second_packed_weight,
            &grad_input,
            &expert_indices,
            &expert_offsets,
            &num_experts,
            &rows,
            &bytes_per_expert,
        };
        launch(
            id,
            static_cast<unsigned int>((in_features + n_per_block - 1) /
                n_per_block),
            static_cast<unsigned int>(num_groups),
            1,
            threads,
            1,
            1,
            0,
            stream,
            arguments);
        return;
    }
    int blocks_per_weight_row = in_features / 256;
    void * arguments[]{
        &first_grad_output,
        &second_grad_output,
        &first_packed_weight,
        &second_packed_weight,
        &grad_input,
        &expert_indices,
        &expert_offsets,
        &num_experts,
        &rows,
        &out_features,
        &in_features,
        &blocks_per_weight_row,
        &bytes_per_expert,
    };
    launch(
        id,
        static_cast<unsigned int>((in_features + n_per_block - 1) /
            n_per_block),
        static_cast<unsigned int>(num_groups),
        1,
        threads,
        1,
        1,
        0,
        stream,
        arguments);
}

void launch_grouped_backward_row_tasks(
        std::int32_t quant_type,
        const void * grad_output,
        const char * packed_weight,
        void * grad_input,
        const std::int32_t * task_count,
        const std::int32_t * task_experts,
        const std::int32_t * task_row_starts,
        const std::int32_t * task_row_ends,
        int max_tasks,
        std::int64_t bytes_per_expert,
        hipStream_t stream) {
    MMQKernelId id;
    switch (quant_type) {
        case kQuantQ4_K:
            id = MMQKernelId::GroupedBwdRowTaskQ4KN2048K512M128N128;
            break;
        case kQuantQ5_K:
            id = MMQKernelId::GroupedBwdRowTaskQ5KN2048K512M128N128;
            break;
        case kQuantIQ2_S:
            id = MMQKernelId::GroupedBwdRowTaskIQ2SN2048K512M128N128;
            break;
        default:
            fail("unsupported grouped backward row-task quant_type " +
                std::to_string(quant_type));
    }
    void * arguments[]{
        &grad_output,
        &packed_weight,
        &grad_input,
        &task_count,
        &task_experts,
        &task_row_starts,
        &task_row_ends,
        &bytes_per_expert,
    };
    launch(
        id,
        512 / kGroupedBackwardTiledN,
        static_cast<unsigned int>(max_tasks),
        1,
        kBackwardThreads,
        1,
        1,
        0,
        stream,
        arguments);
}

} // namespace torch_ggml_ops::mmq_bundle
