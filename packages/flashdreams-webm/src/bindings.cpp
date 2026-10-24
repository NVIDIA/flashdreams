// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <EbSvtAv1Enc.h>
#include <dav1d/dav1d.h>
#include <mkvmuxer/mkvmuxer.h>
#include <mkvmuxer/mkvwriter.h>
#include <mkvparser/mkvparser.h>
#include <mkvparser/mkvreader.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace py = pybind11;

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

using File = std::unique_ptr<FILE, decltype(&fclose)>;
static File open_file(const std::string& path, bool write) {
    if (path.find('\0') != std::string::npos) throw std::invalid_argument("NUL in path");
#ifdef _WIN32
    FILE* fp = _wfopen(std::filesystem::u8path(path).c_str(), write ? L"wb" : L"rb");
#else
    FILE* fp = fopen(path.c_str(), write ? "wb" : "rb");
#endif
    require(fp != nullptr, "Cannot open WebM file");
    return File(fp, fclose);
}

struct Encoder {
    EbComponentType* handle = nullptr;
    bool initialized = false;
    void reset() noexcept {
        if (initialized) svt_av1_enc_deinit(handle);
        if (handle) svt_av1_enc_deinit_handle(handle);
        handle = nullptr;
        initialized = false;
    }
    ~Encoder() { reset(); }
};

static int padded_dimension(int value, int maximum) {
    if (value <= 0 || value > maximum) throw std::invalid_argument("Invalid WebM frame dimension");
    return std::max(64, (value + 1) & ~1);
}

class Writer {
    std::mutex mutex_;
    Encoder encoder_;
    File file_{nullptr, fclose};
    std::unique_ptr<mkvmuxer::MkvWriter> writer_;
    mkvmuxer::Segment segment_;
    uint32_t num_, den_;
    uint64_t frames_ = 0, packets_ = 0;
    int64_t last_pts_ = -1;
    bool closed_ = false;

    bool receive(bool flushing) {
        for (;;) {
            EbBufferHeaderType* packet = nullptr;
            const auto status = svt_av1_enc_get_packet(encoder_.handle, &packet, flushing);
            if (status == EB_NoErrorEmptyQueue) return false;
            require(status == EB_ErrorNone && packet, "SVT-AV1 packet retrieval failed");
            const bool eos = (packet->flags & EB_BUFFERFLAG_EOS) != 0;
            try {
                if (packet->n_filled_len) {
                    require(packet->pts > last_pts_, "Non-monotonic AV1 display timestamps");
                    last_pts_ = packet->pts;
                    const auto timestamp = static_cast<uint64_t>(
                        static_cast<long double>(packet->pts) * 1000000000 * den_ / num_);
                    require(segment_.AddFrame(packet->p_buffer, packet->n_filled_len, 1,
                        timestamp, packet->pic_type == EB_AV1_KEY_PICTURE), "WebM packet write failed");
                    ++packets_;
                }
            } catch (...) {
                svt_av1_enc_release_out_buffer(&packet);
                throw;
            }
            svt_av1_enc_release_out_buffer(&packet);
            if (eos) return true;
        }
    }

public:
    const int width, height;

    Writer(const std::string& path, int visible_width, int visible_height,
           uint32_t num, uint32_t den)
        : num_(num), den_(den), width(padded_dimension(visible_width, 16384)),
          height(padded_dimension(visible_height, 8704)) {
        if (visible_width <= 0 || visible_height <= 0 || visible_width > 16384 ||
            visible_height > 8704 || !num || !den || double(num) / den > 240)
            throw std::invalid_argument("Invalid WebM dimensions or frame rate (maximum 240 FPS)");
        EbSvtAv1EncConfiguration cfg{};
        require(svt_av1_enc_init_handle(&encoder_.handle, &cfg) == EB_ErrorNone, "SVT-AV1 handle failed");
        cfg.source_width = width;
        cfg.source_height = height;
        cfg.frame_rate_numerator = num;
        cfg.frame_rate_denominator = den;
        cfg.encoder_bit_depth = 8;
        cfg.encoder_color_format = EB_YUV420;
        cfg.enc_mode = 10;
        cfg.rate_control_mode = 0;
        cfg.qp = 35;
        cfg.tune = 1;
        cfg.level_of_parallelism = 2;
        cfg.color_primaries = EB_CICP_CP_BT_709;
        cfg.transfer_characteristics = EB_CICP_TC_SRGB;
        cfg.matrix_coefficients = EB_CICP_MC_BT_709;
        require(svt_av1_enc_set_parameter(encoder_.handle, &cfg) == EB_ErrorNone, "SVT-AV1 configuration failed");
        require(svt_av1_enc_init(encoder_.handle) == EB_ErrorNone, "SVT-AV1 initialization failed");
        encoder_.initialized = true;
        file_ = open_file(path, true);
        // MkvWriter borrows the FILE; File checks and owns its lifetime.
        writer_ = std::make_unique<mkvmuxer::MkvWriter>(file_.get());
        require(segment_.Init(writer_.get()), "WebM initialization failed");
        segment_.set_mode(mkvmuxer::Segment::kFile);
        segment_.GetSegmentInfo()->set_timecode_scale(1000000);
        require(segment_.AddVideoTrack(width, height, 1) == 1, "WebM video track failed");
        auto* track = static_cast<mkvmuxer::VideoTrack*>(segment_.GetTrackByNumber(1));
        track->set_codec_id("V_AV1");
        track->set_default_duration(uint64_t(1000000000) * den / num);
        track->set_crop_right(width - visible_width);
        track->set_crop_bottom(height - visible_height);
        track->set_display_width(visible_width);
        track->set_display_height(visible_height);
        EbBufferHeaderType* header = nullptr;
        require(svt_av1_enc_stream_header(encoder_.handle, &header) == EB_ErrorNone, "AV1 stream header failed");
        Dav1dSequenceHeader sequence{};
        const int parsed = dav1d_parse_sequence_header(&sequence, header->p_buffer, header->n_filled_len);
        svt_av1_enc_stream_header_release(header);
        require(parsed == 0 && sequence.profile == 0 && sequence.hbd == 0 &&
            sequence.layout == DAV1D_PIXEL_LAYOUT_I420, "Unexpected AV1 sequence header");
        const auto& op = sequence.operating_points[0];
        const uint8_t codec_private[4] = {0x81,
            uint8_t((op.major_level - 2) * 4 + op.minor_level),
            uint8_t((op.tier << 7) | 0x0c | sequence.chr), 0};
        require(track->SetCodecPrivate(codec_private, 4), "AV1 codec private data failed");
    }

private:
    void submit(const std::string& pixels) {
        const size_t luma = size_t(width) * height;
        if (pixels.size() != luma * 3 / 2) throw std::invalid_argument("Expected one padded I420 frame");
        EbSvtIOFormat picture{};
        picture.luma = reinterpret_cast<uint8_t*>(const_cast<char*>(pixels.data()));
        picture.cb = picture.luma + luma;
        picture.cr = picture.cb + luma / 4;
        picture.y_stride = width;
        picture.cb_stride = picture.cr_stride = width / 2;
        EbBufferHeaderType input{};
        input.size = sizeof(input);
        input.p_buffer = reinterpret_cast<uint8_t*>(&picture);
        input.n_filled_len = static_cast<uint32_t>(pixels.size());
        input.pts = frames_;
        input.pic_type = EB_AV1_INVALID_PICTURE;
        require(svt_av1_enc_send_picture(encoder_.handle, &input) == EB_ErrorNone, "SVT-AV1 frame submission failed");
        ++frames_;
        receive(false);
    }

    std::string convert_rgb(const uint8_t* rgb, int w, int h) const {
        const size_t luma = size_t(width) * height;
        std::string pixels(luma * 3 / 2, '\x80');
        std::fill_n(pixels.begin(), luma, char(16));
        auto quantize = [](float value, float upper) {
            return char(static_cast<uint8_t>(std::clamp(std::nearbyint(value), 16.0f, upper)));
        };
        for (int y = 0; y < h; y += 2) {
            for (int x = 0; x < w; x += 2) {
                float u = 0, v = 0;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const int yy = std::min(y + dy, h - 1);
                        const int xx = std::min(x + dx, w - 1);
                        const auto* p = rgb + (size_t(yy) * w + xx) * 3;
                        const float r = p[0], g = p[1], b = p[2];
                        pixels[size_t(yy) * width + xx] = quantize(
                            r * 0.182586f + g * 0.614231f + b * 0.062007f + 16, 235);
                        u += r * -0.100644f + g * -0.338572f + b * 0.439216f + 128;
                        v += r * 0.439216f + g * -0.398942f + b * -0.040274f + 128;
                    }
                }
                const size_t chroma = size_t(y / 2) * (width / 2) + x / 2;
                pixels[luma + chroma] = quantize(u * 0.25f, 240);
                pixels[luma + luma / 4 + chroma] = quantize(v * 0.25f, 240);
            }
        }
        return pixels;
    }

public:
    void write(const std::string& pixels) {
        std::lock_guard<std::mutex> lock(mutex_);
        require(!closed_, "WebM writer is closed");
        submit(pixels);
    }

    void write_rgb(const uint8_t* rgb, int w, int h) {
        std::lock_guard<std::mutex> lock(mutex_);
        require(!closed_, "WebM writer is closed");
        if (w <= 0 || h <= 0 || w > width || h > height)
            throw std::invalid_argument("RGB frame does not fit WebM canvas");
        submit(convert_rgb(rgb, w, h));
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return;
        closed_ = true;
        try {
            require(frames_ > 0, "Cannot finalize an empty WebM stream");
            EbBufferHeaderType end{};
            end.size = sizeof(end);
            end.flags = EB_BUFFERFLAG_EOS;
            require(svt_av1_enc_send_picture(encoder_.handle, &end) == EB_ErrorNone, "SVT-AV1 flush failed");
            require(receive(true) && packets_ == frames_, "AV1 flush lost frames");
            // Segment duration is in timecode-scale units, not nanoseconds.
            // Include the display time of the last frame, even in a one-frame file.
            segment_.set_duration(double(frames_) * den_ * 1000 / num_);
            require(segment_.Finalize(), "WebM finalization failed");
            writer_->Close();
            require(fclose(file_.release()) == 0, "WebM file close failed");
            encoder_.reset();
        } catch (...) {
            writer_->Close();
            file_.reset();
            encoder_.reset();
            throw;
        }
    }

    void abort() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        encoder_.reset();
        writer_->Close();
        file_.reset();
    }
};

struct Decoder {
    Dav1dContext* handle = nullptr;
    Dav1dData data{};
    ~Decoder() {
        dav1d_data_unref(&data);
        if (handle) dav1d_close(&handle);
    }
};

class Reader {
    File file_;
    mkvparser::MkvReader reader_;
    std::unique_ptr<mkvparser::Segment> segment_;
    Decoder decoder_;
    const mkvparser::Cluster* cluster_ = nullptr;
    const mkvparser::BlockEntry* entry_ = nullptr;
    int frame_index_ = 0;
    bool closed_ = false;
    bool draining_ = false;

    bool next_packet() {
        while (cluster_ && !cluster_->EOS()) {
            if (entry_ && !entry_->EOS()) {
                const auto* block = entry_->GetBlock();
                if (frame_index_ < block->GetFrameCount()) {
                    const auto& frame = block->GetFrame(frame_index_++);
                    require(frame.len > 0, "Empty WebM packet");
                    auto* data = dav1d_data_create(&decoder_.data, frame.len);
                    require(data && frame.Read(&reader_, data) == 0, "WebM packet read failed");
                    return true;
                }
                require(cluster_->GetNext(entry_, entry_) == 0, "Invalid WebM block");
                frame_index_ = 0;
            } else {
                cluster_ = segment_->GetNext(cluster_);
                if (cluster_ && !cluster_->EOS())
                    require(cluster_->GetFirst(entry_) == 0, "Invalid WebM cluster");
            }
        }
        return false;
    }

public:
    int width = 0, height = 0;
    int64_t duration_ns = 0;
    Reader(const std::string& path) : file_(open_file(path, false)), reader_(file_.get()) {
        long long position = 0;
        mkvparser::EBMLHeader header;
        require(header.Parse(&reader_, position) == 0, "Invalid WebM header");
        mkvparser::Segment* segment = nullptr;
        const auto status = mkvparser::Segment::CreateInstance(&reader_, position, segment);
        segment_.reset(segment);
        require(status == 0 && segment, "Invalid WebM segment");
        // ponytail: Load retains block metadata; use incremental parsing for very long recordings.
        require(segment_->Load() == 0, "Invalid or truncated WebM file");
        const auto* tracks = segment_->GetTracks();
        require(tracks && tracks->GetTracksCount() == 1, "Expected one WebM video track");
        const auto* track = tracks->GetTrackByIndex(0);
        require(track && track->GetType() == 1 && track->GetCodecId() &&
            std::strcmp(track->GetCodecId(), "V_AV1") == 0, "Expected AV1 video");
        const auto* video = static_cast<const mkvparser::VideoTrack*>(track);
        width = static_cast<int>(video->GetDisplayWidth());
        height = static_cast<int>(video->GetDisplayHeight());
        require(width > 0 && height > 0 && width <= video->GetWidth() &&
            height <= video->GetHeight(), "Invalid WebM display dimensions");
        duration_ns = segment_->GetInfo()->GetDuration();
        Dav1dSettings settings;
        dav1d_default_settings(&settings);
        settings.n_threads = 2;
        require(dav1d_open(&decoder_.handle, &settings) == 0, "dav1d initialization failed");
        cluster_ = segment_->GetFirst();
        if (cluster_ && !cluster_->EOS())
            require(cluster_->GetFirst(entry_) == 0, "Invalid first WebM block");
    }

    py::object read() {
        require(!closed_, "WebM reader is closed");
        for (;;) {
            Dav1dPicture picture{};
            const auto status = dav1d_get_picture(decoder_.handle, &picture);
            if (status == 0) {
                try {
                    require(picture.p.bpc == 8 && picture.p.layout == DAV1D_PIXEL_LAYOUT_I420 &&
                        picture.p.w >= width && picture.p.h >= height, "Unexpected decoded frame layout");
                    py::tuple planes(3);
                    for (int p = 0; p < 3; ++p) {
                        const int w = p ? (width + 1) / 2 : width;
                        const int h = p ? (height + 1) / 2 : height;
                        std::string bytes(size_t(w) * h, '\0');
                        const auto* source = static_cast<const char*>(picture.data[p]);
                        for (int y = 0; y < h; ++y)
                            std::memcpy(bytes.data() + size_t(y) * w,
                                source + y * picture.stride[p ? 1 : 0], w);
                        planes[p] = py::bytes(bytes);
                    }
                    dav1d_picture_unref(&picture);
                    return std::move(planes);
                } catch (...) {
                    dav1d_picture_unref(&picture);
                    throw;
                }
            }
            require(status == DAV1D_ERR(EAGAIN), "AV1 decoding failed");
            if (draining_) return py::none();
            if (!decoder_.data.sz && !next_packet()) {
                // EOF starts a separate drain pass, including after an EAGAIN
                // immediately following the last dav1d_send_data call.
                draining_ = true;
                continue;
            }
            const auto sent = dav1d_send_data(decoder_.handle, &decoder_.data);
            require(sent == 0 || sent == DAV1D_ERR(EAGAIN), "Invalid AV1 packet");
        }
    }

    void close() {
        if (closed_) return;
        closed_ = true;
        dav1d_data_unref(&decoder_.data);
        if (decoder_.handle) dav1d_close(&decoder_.handle);
        segment_.reset();
        reader_.Close();
        file_.reset();
    }
};

PYBIND11_MODULE(_native, module) {
    py::class_<Writer>(module, "Writer")
        .def(py::init<const std::string&, int, int, uint32_t, uint32_t>(),
             py::arg("path"), py::arg("width"), py::arg("height"),
             py::arg("fps_numerator"), py::arg("fps_denominator"))
        .def_readonly("width", &Writer::width)
        .def_readonly("height", &Writer::height)
        .def("write", &Writer::write, py::call_guard<py::gil_scoped_release>())
        .def("write_rgb", [](Writer& writer, py::buffer pixels) {
            const auto info = pixels.request();
            if (info.ndim != 3 || info.itemsize != 1 || info.format != "B" ||
                info.shape[2] != 3 || info.shape[0] <= 0 || info.shape[1] <= 0 ||
                info.shape[0] > writer.height || info.shape[1] > writer.width ||
                info.strides[2] != 1 || info.strides[1] != 3 ||
                info.strides[0] != info.shape[1] * 3)
                throw std::invalid_argument("Expected contiguous HWC uint8 RGB within the canvas");
            py::gil_scoped_release release;
            writer.write_rgb(static_cast<const uint8_t*>(info.ptr),
                static_cast<int>(info.shape[1]), static_cast<int>(info.shape[0]));
        })
        .def("close", &Writer::close, py::call_guard<py::gil_scoped_release>())
        .def("abort", &Writer::abort, py::call_guard<py::gil_scoped_release>());
    py::class_<Reader>(module, "Reader")
        .def(py::init<const std::string&>())
        .def_readonly("width", &Reader::width)
        .def_readonly("height", &Reader::height)
        .def_readonly("duration_ns", &Reader::duration_ns)
        .def("read", &Reader::read)
        .def("close", &Reader::close);
}
