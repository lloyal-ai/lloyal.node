#pragma once

#include <napi.h>
#include <lloyal/chat_in.hpp>
#include <memory>

/**
 * The PEG parser a format built, held natively and owned by this handle.
 *
 * formatChat hands one back as the result's `parser`; parseChatOutput reads
 * the arena through it. The handle is the ownership: copying it into another
 * record keeps the arena alive, and the last copy's collection frees it. The
 * arena is a pure grammar — it holds no model and no context — so a handle
 * outlives the context that produced it and a context's disposal never
 * touches it. A parser depends on the whole conversation it was rendered for
 * (the generation prompt is a literal inside it), so handles are never
 * shared between formats: each format owns its own.
 */
class ChatParser : public Napi::ObjectWrap<ChatParser> {
public:
  static Napi::Function Init(Napi::Env env, Napi::Object exports);

  /** Wrap an arena. The only way a handle is made. */
  static Napi::Object New(Napi::Env env, std::shared_ptr<const common_peg_arena> arena);

  /** The arena behind a value, or null when the value is not a handle this addon made: the check is the
   *  native type tag the constructor set, never the prototype, which JS can reassign. */
  static std::shared_ptr<const common_peg_arena> arenaOf(Napi::Value value);

  /** Handles alive in this process — what the lifetime tests watch. */
  static size_t live();

  ChatParser(const Napi::CallbackInfo& info);
  ~ChatParser();

private:
  std::shared_ptr<const common_peg_arena> _arena;
};
