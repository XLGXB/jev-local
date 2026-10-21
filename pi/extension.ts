import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { PiBridge, chooseActiveTool } from "./bridge.mjs";

const AskParams = Type.Object({
  observation_id: Type.String({ description: "Caller-supplied ID for this explicit observation (1..128 characters)." }),
  state: Type.Unknown({ description: "Public, finite JSON state for this observation only. No history or labels." }),
  questions: Type.Record(Type.String(), Type.Unknown(), {
    description: "1..8 public typed questions: type, instructions and optional criteria only. No labels.",
  }),
});

const ChooseParams = Type.Object({
  observation_id: Type.String({ description: "Caller-supplied ID for this explicit observation (1..128 characters)." }),
  state: Type.Unknown({ description: "Public, finite JSON state for this observation only." }),
  instructions: Type.String({ description: "What currently active tool would best fit this task?" }),
  candidate_names: Type.Optional(Type.Array(Type.String(), {
    description: "Optional closed set of 2..16 currently active tool names. Unknown, duplicate or inactive names fail.",
  })),
});

export default function jevLocalPiExtension(pi: ExtensionAPI) {
  const bridge = new PiBridge();

  pi.on("session_start", async () => { await bridge.startSession(); });
  pi.on("session_shutdown", async () => { await bridge.shutdown(); });

  pi.registerTool({
    name: "jev_ask",
    label: "Jev typed advice",
    description: "Ask the explicitly configured local Jev worker for typed Noul, Choice or Score advice. "
      + "Advisory only: never grants permission and never executes an action.",
    promptSnippet: "Local typed advice only; unqualified semantics, never execution authority.",
    parameters: AskParams,
    executionMode: "sequential",
    async execute(_toolCallId, params, signal) {
      const packet = await bridge.ask(params, signal);
      const advice = {
        observation_id: packet.observation_id,
        answers: packet.reply.answers,
        runtime_fingerprint: packet.reply.runtime_fingerprint,
        calibrated: false,
        advisory_only: true,
        execution_authorized: false,
        semantic_quality: "unqualified",
      };
      return { content: [{ type: "text", text: JSON.stringify(advice) }], details: advice };
    },
  });

  pi.registerTool({
    name: "jev_choose_tool",
    label: "Jev active-tool suggestion",
    description: "Suggest one of the currently active Pi tools from current tool metadata or an explicit "
      + "2..16-name closed set. Does not call or authorize the suggested tool.",
    promptSnippet: "Advisory tool suggestion only; caller decides independently whether any tool may run.",
    parameters: ChooseParams,
    executionMode: "sequential",
    async execute(_toolCallId, params, signal) {
      const advice = await chooseActiveTool(bridge,
        () => pi.getActiveTools(), () => pi.getAllTools(), params, signal);
      return { content: [{ type: "text", text: JSON.stringify(advice) }], details: advice };
    },
  });
}
