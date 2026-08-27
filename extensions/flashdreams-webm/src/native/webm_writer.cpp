// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "webm_writer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opus.h>
#include <vpx/vp8cx.h>
#include <vpx/vpx_encoder.h>
#include <vpx/vpx_image.h>

#include "mkvmuxer/mkvmuxer.h"
#include "mkvmuxer/mkvwriter.h"

namespace flashdreams_webm {
namespace {

constexpr std::uint64_t kNanosecondsPerSecond = 1'000'000'000ULL;
constexpr std::uint64_t kWebmTimecodeScale = 1'000'000ULL;
constexpr std::uint64_t kOpusSeekPreRollNanoseconds = 80'000'000ULL;
constexpr int kRgbChannels = 3;
constexpr int kOpusFrameMilliseconds = 20;
constexpr int kOpusPacketBytes = 4000;

bool HostIsLittleEndian() {
  const std::uint16_t one = 1;
  return *reinterpret_cast<const std::uint8_t*>(&one) == 1;
}

std::uint8_t ClampByte(int value) {
  return static_cast<std::uint8_t>(std::clamp(value, 0, 255));
}

std::string VpxFailure(vpx_codec_ctx_t* context, const std::string& action,
                       vpx_codec_err_t result) {
  std::string message = action + ": " + vpx_codec_err_to_string(result);
  if (context != nullptr) {
    const char* detail = vpx_codec_error_detail(context);
    if (detail != nullptr && detail[0] != '\0') {
      message += " (";
      message += detail;
      message += ")";
    }
  }
  return message;
}

void RequireVpx(vpx_codec_ctx_t* context, const std::string& action,
                vpx_codec_err_t result) {
  if (result != VPX_CODEC_OK) {
    throw std::runtime_error(VpxFailure(context, action, result));
  }
}

template <typename Value>
void WriteValue(std::ofstream* stream, const Value& value) {
  stream->write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!*stream) {
    throw std::runtime_error("failed to write the private VPx packet spool");
  }
}

template <typename Value>
void ReadValue(std::ifstream* stream, Value* value, const char* field) {
  stream->read(reinterpret_cast<char*>(value), sizeof(*value));
  if (stream->gcount() != static_cast<std::streamsize>(sizeof(*value))) {
    throw std::runtime_error(std::string("truncated VPx packet spool while reading ") +
                             field);
  }
}

struct EncodedPacket {
  std::vector<std::uint8_t> data;
  std::uint64_t timestamp_ns = 0;
  bool key = false;
  std::int64_t discard_padding_ns = 0;
};

class VideoPacketReader final {
 public:
  VideoPacketReader(const std::string& path, int frames_per_second)
      : stream_(path, std::ios::binary), frames_per_second_(frames_per_second) {
    if (!stream_) {
      throw std::runtime_error("failed to reopen the private VPx packet spool");
    }
  }

  std::optional<EncodedPacket> Next() {
    const int next = stream_.peek();
    if (next == std::char_traits<char>::eof()) {
      if (stream_.eof()) {
        return std::nullopt;
      }
      throw std::runtime_error("failed while reading the private VPx packet spool");
    }

    std::uint64_t pts = 0;
    std::uint64_t size = 0;
    std::uint8_t key = 0;
    ReadValue(&stream_, &pts, "timestamp");
    ReadValue(&stream_, &size, "packet size");
    ReadValue(&stream_, &key, "keyframe flag");
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      throw std::runtime_error("VPx packet is too large for this platform");
    }
    EncodedPacket packet;
    packet.data.resize(static_cast<std::size_t>(size));
    stream_.read(reinterpret_cast<char*>(packet.data.data()),
                 static_cast<std::streamsize>(packet.data.size()));
    if (stream_.gcount() != static_cast<std::streamsize>(packet.data.size())) {
      throw std::runtime_error("truncated VPx packet spool while reading payload");
    }
    packet.timestamp_ns =
        pts * kNanosecondsPerSecond / static_cast<std::uint64_t>(frames_per_second_);
    packet.key = key != 0;
    return packet;
  }

 private:
  std::ifstream stream_;
  int frames_per_second_;
};

class AudioPacketReader final {
 public:
  AudioPacketReader(const std::string& path, int sample_rate, int channels)
      : stream_(path, std::ios::binary),
        sample_rate_(sample_rate),
        channels_(channels),
        frame_samples_(sample_rate * kOpusFrameMilliseconds / 1000) {
    if (!HostIsLittleEndian()) {
      throw std::runtime_error("native f32le audio requires a little-endian host");
    }
    if (!stream_) {
      throw std::runtime_error("failed to open staged f32le audio");
    }
    std::error_code file_error;
    const std::uintmax_t bytes = std::filesystem::file_size(path, file_error);
    if (file_error) {
      throw std::runtime_error("failed to size staged f32le audio: " +
                               file_error.message());
    }
    const std::uint64_t bytes_per_sample =
        static_cast<std::uint64_t>(channels_) * sizeof(float);
    if (bytes % bytes_per_sample != 0) {
      throw std::runtime_error("staged f32le audio contains a partial sample");
    }
    total_samples_ = bytes / bytes_per_sample;

    int opus_error = OPUS_OK;
    encoder_.reset(opus_encoder_create(sample_rate_, channels_,
                                       OPUS_APPLICATION_AUDIO, &opus_error));
    if (encoder_ == nullptr || opus_error != OPUS_OK) {
      throw std::runtime_error(std::string("failed to initialize libopus: ") +
                               opus_strerror(opus_error));
    }
    const int bitrate = channels_ == 1 ? 64'000 : 128'000;
    RequireControl(opus_encoder_ctl(encoder_.get(), OPUS_SET_BITRATE(bitrate)),
                   "set Opus bitrate");
    RequireControl(opus_encoder_ctl(encoder_.get(), OPUS_SET_COMPLEXITY(8)),
                   "set Opus complexity");
    RequireControl(
        opus_encoder_ctl(encoder_.get(), OPUS_SET_SIGNAL(OPUS_SIGNAL_MUSIC)),
        "set Opus signal type");
    RequireControl(
        opus_encoder_ctl(encoder_.get(),
                         OPUS_GET_LOOKAHEAD(&lookahead_samples_)),
        "read Opus lookahead");
  }

  AudioPacketReader(const AudioPacketReader&) = delete;
  AudioPacketReader& operator=(const AudioPacketReader&) = delete;

  std::optional<EncodedPacket> Next() {
    if (position_samples_ >= total_samples_) {
      return std::nullopt;
    }
    const std::uint64_t remaining = total_samples_ - position_samples_;
    const int actual_samples = static_cast<int>(
        std::min<std::uint64_t>(remaining, static_cast<std::uint64_t>(frame_samples_)));
    pcm_.assign(static_cast<std::size_t>(frame_samples_ * channels_), 0.0F);
    const std::size_t values = static_cast<std::size_t>(actual_samples * channels_);
    stream_.read(reinterpret_cast<char*>(pcm_.data()),
                 static_cast<std::streamsize>(values * sizeof(float)));
    if (stream_.gcount() != static_cast<std::streamsize>(values * sizeof(float))) {
      throw std::runtime_error("truncated staged f32le audio");
    }

    EncodedPacket packet;
    packet.data.resize(kOpusPacketBytes);
    const int bytes =
        opus_encode_float(encoder_.get(), pcm_.data(), frame_samples_,
                          packet.data.data(), kOpusPacketBytes);
    if (bytes < 0) {
      throw std::runtime_error(std::string("libopus encoding failed: ") +
                               opus_strerror(bytes));
    }
    packet.data.resize(static_cast<std::size_t>(bytes));
    packet.timestamp_ns =
        position_samples_ * kNanosecondsPerSecond /
        static_cast<std::uint64_t>(sample_rate_);
    packet.key = true;
    const int padded_samples = frame_samples_ - actual_samples;
    packet.discard_padding_ns =
        static_cast<std::int64_t>(padded_samples) *
        static_cast<std::int64_t>(kNanosecondsPerSecond) / sample_rate_;
    position_samples_ += static_cast<std::uint64_t>(actual_samples);
    return packet;
  }

  int lookahead_samples() const { return lookahead_samples_; }
  int frame_samples() const { return frame_samples_; }

 private:
  struct OpusEncoderDeleter {
    void operator()(OpusEncoder* encoder) const {
      opus_encoder_destroy(encoder);
    }
  };

  void RequireControl(int result, const char* action) {
    if (result != OPUS_OK) {
      throw std::runtime_error(std::string("failed to ") + action + ": " +
                               opus_strerror(result));
    }
  }

  std::ifstream stream_;
  int sample_rate_;
  int channels_;
  int frame_samples_;
  std::uint64_t total_samples_ = 0;
  std::uint64_t position_samples_ = 0;
  std::unique_ptr<OpusEncoder, OpusEncoderDeleter> encoder_;
  int lookahead_samples_ = 0;
  std::vector<float> pcm_;
};

std::array<std::uint8_t, 19> OpusHeader(int channels, int sample_rate,
                                        int pre_skip) {
  if (pre_skip < 0 || pre_skip > std::numeric_limits<std::uint16_t>::max()) {
    throw std::runtime_error("Opus pre-skip does not fit the WebM OpusHead");
  }
  std::array<std::uint8_t, 19> header{};
  const char signature[] = "OpusHead";
  std::copy(signature, signature + 8, header.begin());
  header[8] = 1;
  header[9] = static_cast<std::uint8_t>(channels);
  header[10] = static_cast<std::uint8_t>(pre_skip & 0xFF);
  header[11] = static_cast<std::uint8_t>((pre_skip >> 8) & 0xFF);
  const std::uint32_t rate = static_cast<std::uint32_t>(sample_rate);
  for (int index = 0; index < 4; ++index) {
    header[12 + index] = static_cast<std::uint8_t>((rate >> (8 * index)) & 0xFF);
  }
  header[16] = 0;
  header[17] = 0;
  header[18] = 0;
  return header;
}

bool IsOpusSampleRate(int sample_rate) {
  return sample_rate == 8000 || sample_rate == 12000 || sample_rate == 16000 ||
         sample_rate == 24000 || sample_rate == 48000;
}

}  // namespace

class WebmWriter::Impl final {
 public:
  Impl(std::string output_path, int width, int height, int frames_per_second,
       std::string codec, int audio_sample_rate, int audio_channels)
      : output_path_(std::move(output_path)),
        packet_path_(output_path_ + ".vpxpackets"),
        width_(width),
        height_(height),
        frames_per_second_(frames_per_second),
        codec_(std::move(codec)),
        audio_sample_rate_(audio_sample_rate),
        audio_channels_(audio_channels) {
    Validate();
    packet_stream_.open(packet_path_, std::ios::binary | std::ios::trunc);
    if (!packet_stream_) {
      throw std::runtime_error("failed to create the private VPx packet spool");
    }
    try {
      InitializeEncoder();
    } catch (...) {
      packet_stream_.close();
      std::error_code ignored;
      std::filesystem::remove(packet_path_, ignored);
      throw;
    }
  }

  ~Impl() { AbortNoThrow(); }

  void WriteVideo(const std::uint8_t* rgb24, std::size_t length) {
    RequireOpen();
    if (rgb24 == nullptr && length != 0) {
      throw std::invalid_argument("RGB24 buffer is null");
    }
    const std::size_t frame_bytes =
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) *
        kRgbChannels;
    if (length % frame_bytes != 0) {
      throw std::invalid_argument("RGB24 buffer does not contain whole frames");
    }
    const std::size_t frames = length / frame_bytes;
    for (std::size_t index = 0; index < frames; ++index) {
      ConvertRgbToI420(rgb24 + index * frame_bytes);
      int flags = 0;
      const std::uint64_t keyframe_interval =
          static_cast<std::uint64_t>(frames_per_second_) * 2;
      if (frames_submitted_ % keyframe_interval == 0) {
        flags |= VPX_EFLAG_FORCE_KF;
      }
      Encode(&image_, frames_submitted_, flags);
      ++frames_submitted_;
    }
  }

  void Close(const std::string& audio_path) {
    if (closed_) {
      return;
    }
    if (aborted_) {
      throw std::runtime_error("WebmWriter was aborted");
    }
    const bool expects_audio = audio_sample_rate_ != 0;
    if (expects_audio != !audio_path.empty()) {
      throw std::invalid_argument(expects_audio
                                      ? "staged audio is required for this writer"
                                      : "audio was supplied to a video-only writer");
    }
    FinishEncoder();
    try {
      Mux(audio_path);
    } catch (...) {
      std::error_code ignored;
      std::filesystem::remove(output_path_, ignored);
      throw;
    }
    std::error_code remove_error;
    std::filesystem::remove(packet_path_, remove_error);
    if (remove_error) {
      throw std::runtime_error("failed to remove the private VPx packet spool: " +
                               remove_error.message());
    }
    closed_ = true;
  }

  void Abort() {
    if (closed_ || aborted_) {
      return;
    }
    DestroyEncoderNoThrow();
    if (packet_stream_.is_open()) {
      packet_stream_.close();
    }
    std::string failures;
    RemoveForAbort(packet_path_, &failures);
    RemoveForAbort(output_path_, &failures);
    if (!failures.empty()) {
      throw std::runtime_error(failures);
    }
    aborted_ = true;
  }

  const std::string& codec() const { return codec_; }
  bool closed() const { return closed_; }

 private:
  void Validate() const {
    if (output_path_.empty()) {
      throw std::invalid_argument("output path must not be empty");
    }
    if (width_ <= 0 || height_ <= 0 || width_ % 2 != 0 || height_ % 2 != 0) {
      throw std::invalid_argument("WebM frame dimensions must be positive and even");
    }
    if (frames_per_second_ <= 0) {
      throw std::invalid_argument("frames_per_second must be positive");
    }
    if (codec_ != "vp8" && codec_ != "vp9") {
      throw std::invalid_argument("codec must be 'vp8' or 'vp9'");
    }
    if ((audio_sample_rate_ == 0) != (audio_channels_ == 0)) {
      throw std::invalid_argument(
          "audio_sample_rate and audio_channels must be set together");
    }
    if (audio_sample_rate_ != 0 && !IsOpusSampleRate(audio_sample_rate_)) {
      throw std::invalid_argument(
          "Opus audio sample rate must be 8000, 12000, 16000, 24000, or 48000 Hz");
    }
    if (audio_channels_ != 0 && audio_channels_ != 1 && audio_channels_ != 2) {
      throw std::invalid_argument("Opus audio must be mono or stereo");
    }
  }

  void InitializeEncoder() {
    const vpx_codec_iface_t* interface =
        codec_ == "vp9" ? vpx_codec_vp9_cx() : vpx_codec_vp8_cx();
    vpx_codec_enc_cfg_t config{};
    RequireVpx(nullptr, "failed to load the default libvpx configuration",
               vpx_codec_enc_config_default(interface, &config, 0));
    config.g_w = static_cast<unsigned int>(width_);
    config.g_h = static_cast<unsigned int>(height_);
    config.g_timebase.num = 1;
    config.g_timebase.den = frames_per_second_;
    config.g_threads = std::max(
        1U, std::min(8U, std::thread::hardware_concurrency()));
    config.g_lag_in_frames = 0;
    config.g_error_resilient = VPX_ERROR_RESILIENT_DEFAULT;
    config.rc_end_usage = VPX_VBR;
    const double kilobits_per_second =
        static_cast<double>(width_) * height_ * frames_per_second_ * 0.16 / 1000.0;
    config.rc_target_bitrate =
        static_cast<unsigned int>(std::max(1000.0, kilobits_per_second));
    config.kf_mode = VPX_KF_AUTO;
    config.kf_min_dist = 0;
    config.kf_max_dist = static_cast<unsigned int>(frames_per_second_ * 2);

    RequireVpx(&encoder_, "failed to initialize libvpx",
               vpx_codec_enc_init(&encoder_, interface, &config, 0));
    encoder_initialized_ = true;
    RequireVpx(&encoder_, "failed to set libvpx speed",
               vpx_codec_control(&encoder_, VP8E_SET_CPUUSED,
                                 codec_ == "vp9" ? 6 : 8));
    if (codec_ == "vp9") {
      RequireVpx(&encoder_, "failed to enable VP9 row multithreading",
                 vpx_codec_control(&encoder_, VP9E_SET_ROW_MT, 1));
      const int tile_columns = width_ >= 1024 ? 2 : (width_ >= 512 ? 1 : 0);
      RequireVpx(&encoder_, "failed to set VP9 tile columns",
                 vpx_codec_control(&encoder_, VP9E_SET_TILE_COLUMNS,
                                   tile_columns));
    }
    if (vpx_img_alloc(&image_, VPX_IMG_FMT_I420,
                      static_cast<unsigned int>(width_),
                      static_cast<unsigned int>(height_), 1) == nullptr) {
      throw std::runtime_error("failed to allocate the libvpx I420 image");
    }
    image_initialized_ = true;
  }

  void ConvertRgbToI420(const std::uint8_t* rgb24) {
    for (int y = 0; y < height_; ++y) {
      std::uint8_t* y_plane = image_.planes[VPX_PLANE_Y] + y * image_.stride[0];
      for (int x = 0; x < width_; ++x) {
        const std::uint8_t* pixel = rgb24 + (y * width_ + x) * kRgbChannels;
        const int value =
            ((66 * pixel[0] + 129 * pixel[1] + 25 * pixel[2] + 128) >> 8) + 16;
        y_plane[x] = ClampByte(value);
      }
    }
    for (int y = 0; y < height_; y += 2) {
      std::uint8_t* u_plane =
          image_.planes[VPX_PLANE_U] + (y / 2) * image_.stride[1];
      std::uint8_t* v_plane =
          image_.planes[VPX_PLANE_V] + (y / 2) * image_.stride[2];
      for (int x = 0; x < width_; x += 2) {
        int red = 0;
        int green = 0;
        int blue = 0;
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            const std::uint8_t* pixel =
                rgb24 + ((y + dy) * width_ + x + dx) * kRgbChannels;
            red += pixel[0];
            green += pixel[1];
            blue += pixel[2];
          }
        }
        red /= 4;
        green /= 4;
        blue /= 4;
        u_plane[x / 2] = ClampByte(
            ((-38 * red - 74 * green + 112 * blue + 128) >> 8) + 128);
        v_plane[x / 2] = ClampByte(
            ((112 * red - 94 * green - 18 * blue + 128) >> 8) + 128);
      }
    }
  }

  bool Encode(vpx_image_t* image, std::uint64_t pts, int flags) {
    RequireVpx(&encoder_, "libvpx failed to encode a frame",
               vpx_codec_encode(&encoder_, image, static_cast<vpx_codec_pts_t>(pts),
                                1, flags, VPX_DL_REALTIME));
    bool wrote_packet = false;
    vpx_codec_iter_t iterator = nullptr;
    const vpx_codec_cx_pkt_t* packet = nullptr;
    while ((packet = vpx_codec_get_cx_data(&encoder_, &iterator)) != nullptr) {
      if (packet->kind != VPX_CODEC_CX_FRAME_PKT) {
        continue;
      }
      const std::uint64_t packet_pts =
          static_cast<std::uint64_t>(packet->data.frame.pts);
      const std::uint64_t packet_size =
          static_cast<std::uint64_t>(packet->data.frame.sz);
      const std::uint8_t key =
          (packet->data.frame.flags & VPX_FRAME_IS_KEY) != 0 ? 1 : 0;
      WriteValue(&packet_stream_, packet_pts);
      WriteValue(&packet_stream_, packet_size);
      WriteValue(&packet_stream_, key);
      packet_stream_.write(static_cast<const char*>(packet->data.frame.buf),
                           static_cast<std::streamsize>(packet_size));
      if (!packet_stream_) {
        throw std::runtime_error("failed to write a VPx packet payload");
      }
      wrote_packet = true;
    }
    return wrote_packet;
  }

  void FinishEncoder() {
    if (encoder_finished_) {
      return;
    }
    while (Encode(nullptr, frames_submitted_, 0)) {
    }
    packet_stream_.flush();
    if (!packet_stream_) {
      throw std::runtime_error("failed to flush the private VPx packet spool");
    }
    packet_stream_.close();
    DestroyEncoder();
    encoder_finished_ = true;
  }

  void DestroyEncoder() {
    if (image_initialized_) {
      vpx_img_free(&image_);
      image_initialized_ = false;
    }
    if (encoder_initialized_) {
      const vpx_codec_err_t result = vpx_codec_destroy(&encoder_);
      encoder_initialized_ = false;
      RequireVpx(&encoder_, "failed to destroy libvpx encoder", result);
    }
  }

  void DestroyEncoderNoThrow() noexcept {
    if (image_initialized_) {
      vpx_img_free(&image_);
      image_initialized_ = false;
    }
    if (encoder_initialized_) {
      vpx_codec_destroy(&encoder_);
      encoder_initialized_ = false;
    }
  }

  void Mux(const std::string& audio_path) {
    VideoPacketReader video_packets(packet_path_, frames_per_second_);
    std::unique_ptr<AudioPacketReader> audio_packets;
    if (!audio_path.empty()) {
      audio_packets = std::make_unique<AudioPacketReader>(
          audio_path, audio_sample_rate_, audio_channels_);
    }

    mkvmuxer::MkvWriter writer;
    if (!writer.Open(output_path_.c_str())) {
      throw std::runtime_error("libwebm failed to create staged output");
    }
    mkvmuxer::Segment segment;
    if (!segment.Init(&writer)) {
      writer.Close();
      throw std::runtime_error("libwebm failed to initialize a segment");
    }
    segment.set_mode(mkvmuxer::Segment::kFile);
    segment.OutputCues(true);
    mkvmuxer::SegmentInfo* info = segment.GetSegmentInfo();
    info->set_timecode_scale(kWebmTimecodeScale);
    info->set_writing_app("FlashDreams native WebM output");

    const std::uint64_t video_track_number =
        segment.AddVideoTrack(width_, height_, 0);
    if (video_track_number == 0) {
      writer.Close();
      throw std::runtime_error("libwebm failed to add the video track");
    }
    auto* video_track = static_cast<mkvmuxer::VideoTrack*>(
        segment.GetTrackByNumber(video_track_number));
    video_track->set_codec_id(codec_ == "vp9" ? "V_VP9" : "V_VP8");
    video_track->set_frame_rate(frames_per_second_);
    video_track->set_default_duration(kNanosecondsPerSecond /
                                      static_cast<std::uint64_t>(frames_per_second_));
    if (!segment.CuesTrack(video_track_number)) {
      writer.Close();
      throw std::runtime_error("libwebm failed to select the video cues track");
    }

    std::uint64_t audio_track_number = 0;
    if (audio_packets != nullptr) {
      audio_track_number =
          segment.AddAudioTrack(audio_sample_rate_, audio_channels_, 0);
      if (audio_track_number == 0) {
        writer.Close();
        throw std::runtime_error("libwebm failed to add the audio track");
      }
      auto* audio_track = static_cast<mkvmuxer::AudioTrack*>(
          segment.GetTrackByNumber(audio_track_number));
      audio_track->set_codec_id(mkvmuxer::Tracks::kOpusCodecId);
      audio_track->set_bit_depth(32);
      audio_track->set_default_duration(
          static_cast<std::uint64_t>(audio_packets->frame_samples()) *
          kNanosecondsPerSecond / static_cast<std::uint64_t>(audio_sample_rate_));
      const int lookahead = audio_packets->lookahead_samples();
      audio_track->set_codec_delay(
          static_cast<std::uint64_t>(lookahead) * kNanosecondsPerSecond /
          static_cast<std::uint64_t>(audio_sample_rate_));
      audio_track->set_seek_pre_roll(kOpusSeekPreRollNanoseconds);
      const auto opus_header =
          OpusHeader(audio_channels_, audio_sample_rate_, lookahead);
      if (!audio_track->SetCodecPrivate(opus_header.data(), opus_header.size())) {
        writer.Close();
        throw std::runtime_error("libwebm failed to set the OpusHead");
      }
    }

    std::optional<EncodedPacket> video = video_packets.Next();
    std::optional<EncodedPacket> audio =
        audio_packets == nullptr ? std::nullopt : audio_packets->Next();
    while (video.has_value() || audio.has_value()) {
      if (audio.has_value() &&
          (!video.has_value() || audio->timestamp_ns < video->timestamp_ns)) {
        const bool added = audio->discard_padding_ns == 0
                               ? segment.AddFrame(audio->data.data(),
                                                  audio->data.size(),
                                                  audio_track_number,
                                                  audio->timestamp_ns, true)
                               : segment.AddFrameWithDiscardPadding(
                                     audio->data.data(), audio->data.size(),
                                     audio->discard_padding_ns,
                                     audio_track_number, audio->timestamp_ns,
                                     true);
        if (!added) {
          writer.Close();
          throw std::runtime_error("libwebm failed to mux an Opus packet");
        }
        audio = audio_packets->Next();
      } else {
        if (!segment.AddFrame(video->data.data(), video->data.size(),
                              video_track_number, video->timestamp_ns,
                              video->key)) {
          writer.Close();
          throw std::runtime_error("libwebm failed to mux a VPx packet");
        }
        video = video_packets.Next();
      }
    }
    if (!segment.Finalize()) {
      writer.Close();
      throw std::runtime_error("libwebm failed to finalize staged output");
    }
    writer.Close();
  }

  void RequireOpen() const {
    if (closed_) {
      throw std::runtime_error("WebmWriter is already closed");
    }
    if (aborted_) {
      throw std::runtime_error("WebmWriter was aborted");
    }
    if (encoder_finished_) {
      throw std::runtime_error("WebmWriter video encoding has already finished");
    }
  }

  static void RemoveForAbort(const std::string& path, std::string* failures) {
    std::error_code error;
    std::filesystem::remove(path, error);
    if (error) {
      if (!failures->empty()) {
        *failures += "; ";
      }
      *failures += "failed to remove " + path + ": " + error.message();
    }
  }

  void AbortNoThrow() noexcept {
    try {
      Abort();
    } catch (...) {
    }
  }

  std::string output_path_;
  std::string packet_path_;
  int width_;
  int height_;
  int frames_per_second_;
  std::string codec_;
  int audio_sample_rate_;
  int audio_channels_;
  std::ofstream packet_stream_;
  vpx_codec_ctx_t encoder_{};
  vpx_image_t image_{};
  bool encoder_initialized_ = false;
  bool image_initialized_ = false;
  bool encoder_finished_ = false;
  bool closed_ = false;
  bool aborted_ = false;
  std::uint64_t frames_submitted_ = 0;
};

WebmWriter::WebmWriter(std::string output_path, int width, int height,
                       int frames_per_second, std::string codec,
                       int audio_sample_rate, int audio_channels)
    : impl_(std::make_unique<Impl>(
          std::move(output_path), width, height, frames_per_second,
          std::move(codec), audio_sample_rate, audio_channels)) {}

WebmWriter::~WebmWriter() = default;

void WebmWriter::WriteVideo(const std::uint8_t* rgb24, std::size_t length) {
  impl_->WriteVideo(rgb24, length);
}

void WebmWriter::Close(const std::string& audio_path) {
  impl_->Close(audio_path);
}

void WebmWriter::Abort() { impl_->Abort(); }

const std::string& WebmWriter::codec() const { return impl_->codec(); }

bool WebmWriter::closed() const { return impl_->closed(); }

const char* LibvpxVersion() { return vpx_codec_version_str(); }

const char* LibopusVersion() { return opus_get_version_string(); }

const char* LibwebmVersion() { return FLASHDREAMS_WEBM_LIBWEBM_VERSION; }

}  // namespace flashdreams_webm
