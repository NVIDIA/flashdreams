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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace flashdreams_webm {

class WebmWriter final {
 public:
  WebmWriter(std::string output_path, int width, int height,
             int frames_per_second, std::string codec, int audio_sample_rate,
             int audio_channels);
  ~WebmWriter();

  WebmWriter(const WebmWriter&) = delete;
  WebmWriter& operator=(const WebmWriter&) = delete;

  void WriteVideo(const std::uint8_t* rgb24, std::size_t length);
  void Close(const std::string& audio_path);
  void Abort();

  const std::string& codec() const;
  bool closed() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

const char* LibvpxVersion();
const char* LibopusVersion();
const char* LibwebmVersion();

}  // namespace flashdreams_webm

