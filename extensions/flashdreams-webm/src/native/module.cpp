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

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#include "webm_writer.h"

namespace {

using flashdreams_webm::WebmWriter;

struct PyWebmWriter {
  PyObject_HEAD
  WebmWriter* writer;
};

bool PathToString(PyObject* object, std::string* result) {
  PyObject* path = PyOS_FSPath(object);
  if (path == nullptr) {
    return false;
  }
  PyObject* bytes = path;
  if (PyUnicode_Check(path)) {
    bytes = PyUnicode_EncodeFSDefault(path);
    Py_DECREF(path);
    if (bytes == nullptr) {
      return false;
    }
  }
  if (!PyBytes_Check(bytes)) {
    Py_DECREF(bytes);
    PyErr_SetString(PyExc_TypeError, "path must resolve to str or bytes");
    return false;
  }
  result->assign(PyBytes_AS_STRING(bytes),
                 static_cast<std::size_t>(PyBytes_GET_SIZE(bytes)));
  Py_DECREF(bytes);
  return true;
}

PyObject* TranslateCurrentException() {
  try {
    throw;
  } catch (const std::invalid_argument& error) {
    PyErr_SetString(PyExc_ValueError, error.what());
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "unknown native WebM failure");
  }
  return nullptr;
}

PyObject* WriterNew(PyTypeObject* type, PyObject*, PyObject*) {
  auto* self = reinterpret_cast<PyWebmWriter*>(type->tp_alloc(type, 0));
  if (self != nullptr) {
    self->writer = nullptr;
  }
  return reinterpret_cast<PyObject*>(self);
}

int WriterInit(PyWebmWriter* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "path", "width", "height", "frames_per_second", "codec",
      "audio_sample_rate", "audio_channels", nullptr};
  PyObject* path_object = nullptr;
  int width = 0;
  int height = 0;
  int frames_per_second = 0;
  const char* codec = "vp9";
  int audio_sample_rate = 0;
  int audio_channels = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "Oiii|sii:WebmWriter",
          const_cast<char**>(keywords), &path_object, &width, &height,
          &frames_per_second, &codec, &audio_sample_rate, &audio_channels)) {
    return -1;
  }
  std::string path;
  if (!PathToString(path_object, &path)) {
    return -1;
  }
  try {
    std::unique_ptr<WebmWriter> writer = std::make_unique<WebmWriter>(
        std::move(path), width, height, frames_per_second, codec,
        audio_sample_rate, audio_channels);
    delete self->writer;
    self->writer = writer.release();
    return 0;
  } catch (...) {
    TranslateCurrentException();
    return -1;
  }
}

void WriterDealloc(PyWebmWriter* self) {
  delete self->writer;
  self->writer = nullptr;
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

WebmWriter* RequireWriter(PyWebmWriter* self) {
  if (self->writer == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "WebmWriter is not initialized");
  }
  return self->writer;
}

PyObject* WriterWriteVideo(PyWebmWriter* self, PyObject* object) {
  WebmWriter* writer = RequireWriter(self);
  if (writer == nullptr) {
    return nullptr;
  }
  Py_buffer buffer{};
  if (PyObject_GetBuffer(object, &buffer, PyBUF_CONTIG_RO) != 0) {
    return nullptr;
  }
  try {
    writer->WriteVideo(static_cast<const std::uint8_t*>(buffer.buf),
                       static_cast<std::size_t>(buffer.len));
  } catch (...) {
    PyBuffer_Release(&buffer);
    return TranslateCurrentException();
  }
  PyBuffer_Release(&buffer);
  Py_RETURN_NONE;
}

PyObject* WriterClose(PyWebmWriter* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"audio_path", nullptr};
  PyObject* audio_path_object = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:close",
                                   const_cast<char**>(keywords),
                                   &audio_path_object)) {
    return nullptr;
  }
  WebmWriter* writer = RequireWriter(self);
  if (writer == nullptr) {
    return nullptr;
  }
  std::string audio_path;
  if (audio_path_object != Py_None &&
      !PathToString(audio_path_object, &audio_path)) {
    return nullptr;
  }
  try {
    writer->Close(audio_path);
  } catch (...) {
    return TranslateCurrentException();
  }
  Py_RETURN_NONE;
}

PyObject* WriterAbort(PyWebmWriter* self, PyObject*) {
  WebmWriter* writer = RequireWriter(self);
  if (writer == nullptr) {
    return nullptr;
  }
  try {
    writer->Abort();
  } catch (...) {
    return TranslateCurrentException();
  }
  Py_RETURN_NONE;
}

PyObject* WriterCodec(PyWebmWriter* self, void*) {
  WebmWriter* writer = RequireWriter(self);
  if (writer == nullptr) {
    return nullptr;
  }
  return PyUnicode_FromString(writer->codec().c_str());
}

PyObject* WriterClosed(PyWebmWriter* self, void*) {
  WebmWriter* writer = RequireWriter(self);
  if (writer == nullptr) {
    return nullptr;
  }
  return PyBool_FromLong(writer->closed() ? 1 : 0);
}

PyMethodDef kWriterMethods[] = {
    {"write_video", reinterpret_cast<PyCFunction>(WriterWriteVideo), METH_O,
     "Encode contiguous RGB24 frames."},
    {"close", reinterpret_cast<PyCFunction>(WriterClose),
     METH_VARARGS | METH_KEYWORDS, "Finalize the staged WebM file."},
    {"abort", reinterpret_cast<PyCFunction>(WriterAbort), METH_NOARGS,
     "Discard native staging state."},
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef kWriterGetSet[] = {
    {const_cast<char*>("codec"), reinterpret_cast<getter>(WriterCodec), nullptr,
     const_cast<char*>("Selected video codec."), nullptr},
    {const_cast<char*>("closed"), reinterpret_cast<getter>(WriterClosed),
     nullptr, const_cast<char*>("Whether finalization completed."), nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyType_Slot kWriterSlots[] = {
    {Py_tp_new, reinterpret_cast<void*>(WriterNew)},
    {Py_tp_init, reinterpret_cast<void*>(WriterInit)},
    {Py_tp_dealloc, reinterpret_cast<void*>(WriterDealloc)},
    {Py_tp_methods, kWriterMethods},
    {Py_tp_getset, kWriterGetSet},
    {Py_tp_doc,
     const_cast<char*>("Native incremental VP8/VP9 and Opus WebM writer.")},
    {0, nullptr},
};

PyType_Spec kWriterSpec = {
    "flashdreams_webm._native.WebmWriter",
    sizeof(PyWebmWriter),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    kWriterSlots,
};

PyObject* Versions(PyObject*, PyObject*) {
  PyObject* versions = PyDict_New();
  if (versions == nullptr) {
    return nullptr;
  }
  const struct {
    const char* name;
    const char* version;
  } entries[] = {
      {"libvpx", flashdreams_webm::LibvpxVersion()},
      {"libopus", flashdreams_webm::LibopusVersion()},
      {"libwebm", flashdreams_webm::LibwebmVersion()},
  };
  for (const auto& entry : entries) {
    PyObject* value = PyUnicode_FromString(entry.version);
    if (value == nullptr || PyDict_SetItemString(versions, entry.name, value) != 0) {
      Py_XDECREF(value);
      Py_DECREF(versions);
      return nullptr;
    }
    Py_DECREF(value);
  }
  return versions;
}

PyMethodDef kModuleMethods[] = {
    {"versions", Versions, METH_NOARGS,
     "Return versions of the statically linked native libraries."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Native VP8/VP9, Opus, and WebM writer.",
    -1,
    kModuleMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit__native() {
  PyObject* module = PyModule_Create(&kModule);
  if (module == nullptr) {
    return nullptr;
  }
  PyObject* writer_type = PyType_FromSpec(&kWriterSpec);
  if (writer_type == nullptr ||
      PyModule_AddObject(module, "WebmWriter", writer_type) != 0) {
    Py_XDECREF(writer_type);
    Py_DECREF(module);
    return nullptr;
  }
  return module;
}
