#include "Util.hpp"
#include <md4c.h>
#include <algorithm>
#include <climits>
#include <string>
#include <vector>

namespace liblloyal_node {

struct Section {
  std::string heading;
  unsigned level = 0;
  int start_line = 0;
  int end_line = 0;
};

struct ParseState {
  const char* input;
  size_t input_size;
  std::vector<size_t> line_starts;

  int depth = 0;
  bool in_heading = false;
  unsigned heading_level = 0;
  std::string heading_text;

  // Byte offset of the first text seen in the current top-level block
  size_t block_first_offset = SIZE_MAX;

  std::vector<Section> sections;

  void build_line_table() {
    line_starts.push_back(0);
    for (size_t i = 0; i < input_size; i++) {
      if (input[i] == '\n') {
        line_starts.push_back(i + 1);
      }
    }
  }

  // Binary search: find the 1-indexed line number containing the given byte offset
  int line_at(size_t offset) const {
    auto it = std::upper_bound(line_starts.begin(), line_starts.end(), offset);
    return static_cast<int>(it - line_starts.begin());
  }

  // Line number of the last content line in the input
  int last_line() const {
    if (input_size == 0) return 0;
    size_t last = input_size - 1;
    if (input[last] == '\n' && last > 0) last--;
    return line_at(last);
  }
};

// md4c callbacks — static functions with C-compatible signatures

static int on_enter_block(MD_BLOCKTYPE type, void* detail, void* userdata) {
  auto* s = static_cast<ParseState*>(userdata);
  s->depth++;

  // depth==2 means direct child of MD_BLOCK_DOC (top-level block)
  if (s->depth == 2) {
    s->block_first_offset = SIZE_MAX;

    if (type == MD_BLOCK_H) {
      s->in_heading = true;
      s->heading_level = static_cast<MD_BLOCK_H_DETAIL*>(detail)->level;
      s->heading_text.clear();
    }
  }
  return 0;
}

static int on_leave_block(MD_BLOCKTYPE type, void* /* detail */, void* userdata) {
  auto* s = static_cast<ParseState*>(userdata);

  if (s->depth == 2 && type == MD_BLOCK_H && s->block_first_offset != SIZE_MAX) {
    int heading_line = s->line_at(s->block_first_offset);

    // Close previous section
    if (!s->sections.empty()) {
      s->sections.back().end_line = heading_line - 1;
    }

    // Start new section at this heading
    Section sec;
    sec.heading = s->heading_text;
    sec.level = s->heading_level;
    sec.start_line = heading_line;
    s->sections.push_back(sec);

    s->in_heading = false;
  }

  s->depth--;
  return 0;
}

static int on_enter_span(MD_SPANTYPE /* type */, void* /* detail */, void* /* userdata */) {
  return 0;
}

static int on_leave_span(MD_SPANTYPE /* type */, void* /* detail */, void* /* userdata */) {
  return 0;
}

static int on_text(MD_TEXTTYPE /* type */, const MD_CHAR* text, MD_SIZE size, void* userdata) {
  auto* s = static_cast<ParseState*>(userdata);

  // Track first text offset for the current top-level block
  if (s->depth >= 2 && s->block_first_offset == SIZE_MAX) {
    s->block_first_offset = static_cast<size_t>(text - s->input);
  }

  // Accumulate heading text
  if (s->in_heading) {
    s->heading_text.append(text, size);
  }

  return 0;
}

// N-API entry points

void Util::Init(Napi::Env env, Napi::Object exports) {
  exports.Set("parseMarkdown", Napi::Function::New(env, ParseMarkdown));
}

Napi::Value Util::ParseMarkdown(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "parseMarkdown expects a string argument")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  std::string input = info[0].As<Napi::String>().Utf8Value();

  // Empty input → empty result
  if (input.empty()) {
    return Napi::Array::New(env, 0);
  }

  ParseState state;
  state.input = input.c_str();
  state.input_size = input.size();
  state.build_line_table();

  // Preamble: content before the first heading
  Section preamble;
  preamble.start_line = 1;
  state.sections.push_back(preamble);

  MD_PARSER parser = {};
  parser.abi_version = 0;
  parser.flags = MD_FLAG_TABLES | MD_FLAG_STRIKETHROUGH;
  parser.enter_block = on_enter_block;
  parser.leave_block = on_leave_block;
  parser.enter_span = on_enter_span;
  parser.leave_span = on_leave_span;
  parser.text = on_text;

  md_parse(input.c_str(), static_cast<MD_SIZE>(input.size()), &parser, &state);

  // Close last section
  if (!state.sections.empty()) {
    state.sections.back().end_line = state.last_line();
  }

  // Remove empty sections (startLine > endLine)
  state.sections.erase(
      std::remove_if(state.sections.begin(), state.sections.end(),
                     [](const Section& sec) { return sec.start_line > sec.end_line; }),
      state.sections.end());

  // Build N-API result
  Napi::Array result = Napi::Array::New(env, state.sections.size());
  for (size_t i = 0; i < state.sections.size(); i++) {
    const auto& sec = state.sections[i];
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("heading", sec.heading);
    obj.Set("level", static_cast<double>(sec.level));
    obj.Set("startLine", static_cast<double>(sec.start_line));
    obj.Set("endLine", static_cast<double>(sec.end_line));
    result.Set(static_cast<uint32_t>(i), obj);
  }

  return result;
}

} // namespace liblloyal_node
