#include <torch/extension.h>
#include <ATen/cuda/CUDABlas.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <cmath>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x)                                                                   \
    CHECK_CUDA(x);                                                                       \
    CHECK_CONTIGUOUS(x)

struct BatchLoraArgs {
    BatchLoraArgs(int start_idx, int end_idx, int rank, double dropout, double scaling)
        : start_idx_(start_idx), end_idx_(end_idx), rank_(rank), dropout_(dropout),
          scaling_(scaling) {
    }

    int start_idx_;
    int end_idx_;
    int rank_;
    double dropout_;
    double scaling_;
};

/* maybe use the stream can speed up? but need more gpu memory */
/* Ye: not implement now.*/
class LoraStreamContext {
public:
    LoraStreamContext() = default;
    LoraStreamContext(const LoraStreamContext &) = delete;
    LoraStreamContext &operator=(const LoraStreamContext &) = delete;

    cublasHandle_t get_handle(int idx) {
        return 0;
    }

private:
    std::vector<cudaStream_t> g_streams;
    std::vector<cublasHandle_t> g_handles;
};

static LoraStreamContext g_lora_stream_context;

void init_dropout_and_scaling(float *tmp_dropout_ptr, float *batch_data_ptr,
                              int batch_size, int seq_len, int in_dim, at::Device device,
                              double scaling, double dropout) {
    auto dummy_free_mem_fn = [](void *) {
        return;
    };
    /* use the tensor to wrap dropout */
    auto options = at::TensorOptions(at::ScalarType::Float).device(device);
    torch::Tensor dropout_tensor =
        torch::from_blob(tmp_dropout_ptr, {batch_size, seq_len, in_dim},
                         {seq_len * in_dim, in_dim, 1}, dummy_free_mem_fn, options);
    /* use the tensor to wrap in data */
    torch::Tensor batch_data_tensor =
        torch::from_blob(batch_data_ptr, {batch_size, seq_len, in_dim},
                         {seq_len * in_dim, in_dim, 1}, dummy_free_mem_fn, options);

    if (dropout > 1e-6) {
        dropout_tensor.bernoulli_(1 - dropout);
    }

    double to_scaling = scaling / (1 - dropout);
    if (std::abs(to_scaling - 1.0) > 1e-6) {
        dropout_tensor.mul_(to_scaling);
    }

    dropout_tensor.mul_(batch_data_tensor);

    return;
}

/* ret: the tensor for the dropout */
/* linear_result dim is batch_size * seq_len * out_dim */
/* batch_data dim is batch_size * seq_len * in_dim */
/* the lora_a dim is rank * in_dim */
/* the lora_b dim is out_dim * rank */
/* only accepte the float tensor */
void batch_lora_forward(c10::optional<torch::Tensor> &tmp_dropout,
                        torch::Tensor &linear_result, const torch::Tensor &batch_data,
                        torch::Tensor &tmp_data, const std::vector<BatchLoraArgs> &inargs,
                        int in_dim, int out_dim, int seq_len,
                        const std::vector<c10::optional<torch::Tensor>> &loras) {
    CHECK_INPUT(linear_result);
    CHECK_INPUT(batch_data);
    CHECK_INPUT(tmp_data);

    /* do not change the batch_data, cannot use const */
    float *batch_data_ptr = batch_data.data_ptr<float>();
    float *result_data_ptr = linear_result.data_ptr<float>();
    float *tmp_data_ptr = tmp_data.data_ptr<float>();
    float *tmp_dropout_ptr = nullptr;

    if (tmp_dropout.has_value()) {
        CHECK_INPUT((*tmp_dropout));
        tmp_dropout_ptr = tmp_dropout->data_ptr<float>();
    }

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    /* calc the in @ lora_a @ lora_b */
    for (size_t idx = 0; idx < inargs.size(); ++idx) {
        int batch_size = inargs[idx].end_idx_ - inargs[idx].start_idx_;
        int r = inargs[idx].rank_;

        const c10::optional<torch::Tensor> &lora_a = loras[idx * 2];
        const c10::optional<torch::Tensor> &lora_b = loras[idx * 2 + 1];

        if (r <= 0) {
            TORCH_CHECK(!lora_a.has_value(), "batch arg error - f lora_a");
            TORCH_CHECK(!lora_b.has_value(), "batch arg error - f lora_b");
            batch_data_ptr += (batch_size * seq_len * in_dim);
            result_data_ptr += (batch_size * seq_len * out_dim);
            continue;
        }

        /* dropout */
        const float *after_dropout_ptr = batch_data_ptr;
        if (tmp_dropout_ptr != nullptr) {
            init_dropout_and_scaling(tmp_dropout_ptr, batch_data_ptr, batch_size, seq_len,
                                     in_dim, tmp_dropout->device(), inargs[idx].scaling_,
                                     inargs[idx].dropout_);
            after_dropout_ptr = tmp_dropout_ptr;
        }

        const float *lora_a_ptr = lora_a->data_ptr<float>();
        const float *lora_b_ptr = lora_b->data_ptr<float>();

        float alpha = 1.0;
        float beta = 0.0;

        /* calc the data @ lora_a */
        TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, r, seq_len, in_dim, &alpha, lora_a_ptr,
            in_dim, 0, after_dropout_ptr, in_dim, in_dim * seq_len, &beta, tmp_data_ptr,
            r, r * seq_len, batch_size));

        beta = 1.0;

        /* calc the data @ lora_b */
        TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, seq_len, r, &alpha, lora_b_ptr, r,
            0, tmp_data_ptr, r, r * seq_len, &beta, result_data_ptr, out_dim,
            out_dim * seq_len, batch_size));

        batch_data_ptr += (batch_size * seq_len * in_dim);
        result_data_ptr += (batch_size * seq_len * out_dim);
    }

    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    cudaStreamSynchronize(stream);

    return;
}

void batch_lora_backward(const torch::Tensor &grad_out, const torch::Tensor &batch_data,
                         c10::optional<torch::Tensor> &grad_batch_data,
                         torch::Tensor &tmp_data, torch::Tensor &tmp_lora_data,
                         const torch::Tensor &tmp_one_vector,
                         const std::vector<BatchLoraArgs> &inargs, int in_dim,
                         int out_dim, int seq_len,
                         const std::vector<c10::optional<torch::Tensor>> &loras,
                         std::vector<c10::optional<torch::Tensor>> &grad_loras) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(batch_data);
    CHECK_INPUT(tmp_data);
    CHECK_INPUT(tmp_lora_data);
    CHECK_INPUT(tmp_one_vector);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    /* calc the grad_out @ lora_b */
    float *grad_batch_data_ptr = nullptr;
    if (grad_batch_data.has_value()) {
        CHECK_INPUT((*grad_batch_data));
        grad_batch_data_ptr = grad_batch_data->data_ptr<float>();
    }

    const float *grad_out_ptr = grad_out.data_ptr<float>();
    const float *batch_data_ptr = batch_data.data_ptr<float>();
    const float *tmp_one_vector_ptr = tmp_one_vector.data_ptr<float>();

    float *tmp_data_ptr = tmp_data.data_ptr<float>();
    float *tmp_lora_data_ptr = tmp_lora_data.data_ptr<float>();

    for (size_t idx = 0; idx < inargs.size(); ++idx) {
        int batch_size = inargs[idx].end_idx_ - inargs[idx].start_idx_;
        int r = inargs[idx].rank_;

        const c10::optional<torch::Tensor> &lora_a = loras[idx * 2];
        const c10::optional<torch::Tensor> &lora_b = loras[idx * 2 + 1];

        c10::optional<torch::Tensor> &grad_lora_a = grad_loras[idx * 2];
        c10::optional<torch::Tensor> &grad_lora_b = grad_loras[idx * 2 + 1];

        if (r <= 0) {
            TORCH_CHECK(!lora_b.has_value(), "batch arg error - b lora_b");
            TORCH_CHECK(!lora_a.has_value(), "batch arg error - b lora_a");
            TORCH_CHECK(!grad_lora_b.has_value(), "batch arg error - b grad_lora_b");
            TORCH_CHECK(!grad_lora_a.has_value(), "batch arg error - b grad_lora_a");
            grad_out_ptr += (batch_size * seq_len * out_dim);
            batch_data_ptr += (batch_size * seq_len * in_dim);
            if (grad_batch_data_ptr != nullptr) {
                grad_batch_data_ptr += (batch_size * seq_len * in_dim);
            }
            continue;
        }

        const float *lora_b_ptr = lora_b->data_ptr<float>();
        const float *lora_a_ptr = lora_a->data_ptr<float>();

        float *grad_lora_a_ptr = grad_lora_a->data_ptr<float>();
        float *grad_lora_b_ptr = grad_lora_b->data_ptr<float>();

        float alpha = 1.0;
        float beta = 0.0;

        /* dy b */
        TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, r, seq_len, out_dim, &alpha, lora_b_ptr, r,
            0, grad_out_ptr, out_dim, seq_len * out_dim, &beta, tmp_data_ptr, r,
            r * seq_len, batch_size));

        alpha = inargs[idx].scaling_ / (1 - inargs[idx].dropout_);
        beta = 0.0;

        if (grad_batch_data_ptr != nullptr) {
            TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, in_dim, seq_len, r, &alpha, lora_a_ptr,
                in_dim, 0, tmp_data_ptr, r, r * seq_len, &beta, grad_batch_data_ptr,
                in_dim, seq_len * in_dim, batch_size));

            grad_batch_data_ptr += (batch_size * seq_len * in_dim);
        }

        /* da */
        TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_T, in_dim, r, seq_len, &alpha, batch_data_ptr,
            in_dim, in_dim * seq_len, tmp_data_ptr, r, r * seq_len, &beta,
            tmp_lora_data_ptr, in_dim, in_dim * r, batch_size));

        /* sum da -> da */
        alpha = 1.0;
        beta = 0.0;
        TORCH_CUDABLAS_CHECK(cublasSgemv(
            handle, CUBLAS_OP_N, in_dim * r, batch_size, &alpha, tmp_lora_data_ptr,
            in_dim * r, tmp_one_vector_ptr, 1, &beta, grad_lora_a_ptr, 1));

        /* ax */
        alpha = inargs[idx].scaling_ / (1 - inargs[idx].dropout_);
        beta = 0.0;
        TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, r, seq_len, in_dim, &alpha, lora_a_ptr,
            in_dim, 0, batch_data_ptr, in_dim, in_dim * seq_len, &beta, tmp_data_ptr, r,
            r * seq_len, batch_size));

        alpha = 1.0;
        beta = 0.0;
        TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_T, r, out_dim, seq_len, &alpha, tmp_data_ptr,
            r, r * seq_len, grad_out_ptr, out_dim, out_dim * seq_len, &beta,
            tmp_lora_data_ptr, r, r * out_dim, batch_size));

        alpha = 1.0;
        beta = 0.0;
        TORCH_CUDABLAS_CHECK(cublasSgemv(
            handle, CUBLAS_OP_N, out_dim * r, batch_size, &alpha, tmp_lora_data_ptr,
            out_dim * r, tmp_one_vector_ptr, 1, &beta, grad_lora_b_ptr, 1));

        batch_data_ptr += (batch_size * seq_len * in_dim);
        grad_out_ptr += (batch_size * seq_len * out_dim);
    }

    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    cudaStreamSynchronize(stream);

    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BatchLoraArgs>(m, "BatchLoraArgs")
        .def(py::init<int, int, int, double, double>());
    m.def("batch_lora_forward", &batch_lora_forward, "batch_lora_forward (CUDA)");
    m.def("batch_lora_backward", &batch_lora_backward, "batch_lora_backward (CUDA)");
}
