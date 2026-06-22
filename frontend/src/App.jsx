import { useMemo, useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "https://llm-serving-inference-platform.onrender.com";
const DEFAULT_MODEL =
  import.meta.env.VITE_MODEL_NAME || "sshleifer/tiny-gpt2";

function formatLatency(startMs, endMs) {
  return `${Math.max(1, endMs - startMs)} ms`;
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Inference console is online. Ask something to test latency and batching.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [lastStats, setLastStats] = useState(null);
  const [error, setError] = useState("");
  const [streamMode, setStreamMode] = useState(false);

  const tokenSummary = useMemo(() => {
    if (!lastStats) {
      return "No requests yet";
    }
    return `${lastStats.promptTokens} prompt • ${lastStats.completionTokens} completion • ${lastStats.totalTokens} total`;
  }, [lastStats]);

  async function submitPrompt(event) {
    event.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) {
      return;
    }

    const userMessage = { role: "user", content: trimmed };
    const requestMessages = [...messages, userMessage];

    let assistantIndex = -1;
    if (streamMode) {
      assistantIndex = requestMessages.length;
      setMessages([...requestMessages, { role: "assistant", content: "" }]);
    } else {
      setMessages(requestMessages);
    }

    setInput("");
    setError("");
    setIsLoading(true);

    const startedAt = Date.now();

    try {
      const response = await fetch(`${API_BASE_URL}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: DEFAULT_MODEL,
          messages: requestMessages,
          temperature: 0.2,
          top_p: 0.9,
          max_tokens: 64,
          stream: streamMode,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Inference request failed");
      }

      if (!streamMode) {
        const data = await response.json();
        const assistantText = data?.choices?.[0]?.message?.content || "No output returned";

        setMessages((prev) => [...prev, { role: "assistant", content: assistantText }]);
        setLastStats({
          latency: formatLatency(startedAt, Date.now()),
          promptTokens: data.usage?.prompt_tokens ?? 0,
          completionTokens: data.usage?.completion_tokens ?? 0,
          totalTokens: data.usage?.total_tokens ?? 0,
        });
      } else {
        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("Streaming reader is unavailable");
        }

        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let streamUsage = null;
        let sawDone = false;

        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split("\n\n");
          buffer = events.pop() || "";

          for (const eventChunk of events) {
            const lines = eventChunk
              .split("\n")
              .filter((line) => line.startsWith("data: "))
              .map((line) => line.slice(6).trim());

            for (const line of lines) {
              if (line === "[DONE]") {
                sawDone = true;
                break;
              }

              let parsed;
              try {
                parsed = JSON.parse(line);
              } catch {
                continue;
              }
              const delta = parsed?.choices?.[0]?.delta?.content || "";
              if (parsed?.usage) {
                streamUsage = parsed.usage;
              }

              if (delta) {
                setMessages((prev) => {
                  const updated = [...prev];
                  const current = updated[assistantIndex]?.content || "";
                  updated[assistantIndex] = { role: "assistant", content: `${current}${delta}` };
                  return updated;
                });
              }
            }

            if (sawDone) {
              break;
            }
          }

          if (sawDone) {
            try {
              await reader.cancel();
            } catch {
              // Ignore cancellation errors when stream already ended server-side.
            }
            break;
          }
        }

        setLastStats({
          latency: formatLatency(startedAt, Date.now()),
          promptTokens: streamUsage?.prompt_tokens ?? 0,
          completionTokens: streamUsage?.completion_tokens ?? 0,
          totalTokens: streamUsage?.total_tokens ?? 0,
        });
      }
    } catch (requestError) {
      setError(requestError.message || "Unknown error");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "The backend could not process the request. Check API URL and model startup logs.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="page-shell">
      <div className="ambient-glow" />
      <main className="console">
        <header className="console-header">
          <div>
            <p className="eyebrow">LLM Serving Platform</p>
            <h1>Low-Latency Inference Console</h1>
            <label className="stream-toggle">
              <input
                type="checkbox"
                checked={streamMode}
                onChange={(event) => setStreamMode(event.target.checked)}
              />
              Stream tokens live
            </label>
          </div>
          <div className="meta-card">
            <p className="meta-title">Last Request</p>
            <p>{lastStats?.latency || "Waiting"}</p>
            <p className="meta-subtle">{tokenSummary}</p>
          </div>
        </header>

        <section className="chat-panel" aria-live="polite">
          {messages.map((message, index) => (
            <article key={`${message.role}-${index}`} className={`bubble ${message.role}`}>
              <p className="bubble-role">{message.role}</p>
              <p>{message.content}</p>
            </article>
          ))}
          {isLoading && (
            <article className="bubble assistant thinking">
              <p className="bubble-role">assistant</p>
              <p>Generating response...</p>
            </article>
          )}
        </section>

        <form className="composer" onSubmit={submitPrompt}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            rows={3}
            placeholder="Ask the model anything"
          />
          <div className="composer-row">
            <p className="endpoint">{API_BASE_URL}/v1/chat/completions</p>
            <button type="submit" disabled={isLoading || !input.trim()}>
              {isLoading ? "Running..." : "Run Inference"}
            </button>
          </div>
          {error ? <p className="error-text">{error}</p> : null}
        </form>
      </main>
    </div>
  );
}
