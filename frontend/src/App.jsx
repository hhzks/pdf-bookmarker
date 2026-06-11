import { useEffect, useRef, useState } from "react";
import { createJob, getJob, downloadUrl } from "./api";

const MODELS = [
  { value: "anthropic:claude-opus-4-8", label: "Claude Opus 4.8 (default)" },
  { value: "anthropic:claude-sonnet-4-6", label: "Claude Sonnet 4.6" },
  { value: "gemini:gemini-3.5-flash", label: "Gemini 3.5 Flash" },
];

const POLL_MS = 1500;

export default function App() {
  // idle | uploading | processing | done | failed
  const [phase, setPhase] = useState("idle");
  const [file, setFile] = useState(null);
  const [llmMode, setLlmMode] = useState("auto");
  const [model, setModel] = useState(MODELS[0].value);
  const [apiKey, setApiKey] = useState("");
  const [showKeyField, setShowKeyField] = useState(false);
  const [progress, setProgress] = useState(0);
  const [jobId, setJobId] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    if (phase !== "processing" || !jobId) return;
    const timer = setInterval(async () => {
      try {
        const body = await getJob(jobId);
        if (body.status === "done") {
          setResult(body);
          setPhase("done");
        } else if (body.status === "failed") {
          setError(body.error ?? "Processing failed.");
          setPhase("failed");
        }
      } catch (err) {
        setError(err.message);
        setPhase("failed");
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [phase, jobId]);

  function pickFile(f) {
    if (f && f.name.toLowerCase().endsWith(".pdf")) {
      setFile(f);
      setError(null);
    } else {
      setError("Please choose a PDF file.");
    }
  }

  async function start() {
    setPhase("uploading");
    setProgress(0);
    setError(null);
    try {
      const id = await createJob(
        file,
        { llmMode, model, apiKey: apiKey.trim() },
        setProgress
      );
      setJobId(id);
      setPhase("processing");
    } catch (err) {
      setError(err.message);
      setPhase("failed");
    }
  }

  function reset() {
    setPhase("idle");
    setFile(null);
    setJobId(null);
    setResult(null);
    setError(null);
    setProgress(0);
  }

  return (
    <main className="page">
      <h1>PDF Bookmarker</h1>
      <p className="tagline">
        Add a clickable, hierarchical outline to any text-based PDF.
      </p>

      {phase === "idle" && (
        <section className="card">
          <div
            className={`dropzone ${dragOver ? "over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              pickFile(e.dataTransfer.files[0]);
            }}
            onClick={() => inputRef.current.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept="application/pdf,.pdf"
              hidden
              onChange={(e) => pickFile(e.target.files[0])}
            />
            {file ? (
              <p className="filename">{file.name}</p>
            ) : (
              <p>Drop a PDF here, or click to choose one (max 50 MB)</p>
            )}
          </div>

          <fieldset className="options">
            <legend>LLM verification</legend>
            {[
              ["auto", "Auto — only when the detected outline looks unreliable"],
              ["always", "Always — verify every outline with the LLM"],
              ["never", "Never — heuristics only"],
            ].map(([value, label]) => (
              <label key={value} className="radio">
                <input
                  type="radio"
                  name="llm_mode"
                  value={value}
                  checked={llmMode === value}
                  onChange={() => setLlmMode(value)}
                />
                {label}
              </label>
            ))}

            {llmMode !== "never" && (
              <>
                <label className="field">
                  Model
                  <select value={model} onChange={(e) => setModel(e.target.value)}>
                    {MODELS.map((m) => (
                      <option key={m.value} value={m.value}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  className="linklike"
                  onClick={() => setShowKeyField(!showKeyField)}
                >
                  {showKeyField ? "Use the server's API key" : "Use my own API key"}
                </button>
                {showKeyField && (
                  <label className="field">
                    API key (used only for this job, never stored)
                    <input
                      type="password"
                      value={apiKey}
                      autoComplete="off"
                      placeholder="sk-… or AIza…"
                      onChange={(e) => setApiKey(e.target.value)}
                    />
                  </label>
                )}
              </>
            )}
          </fieldset>

          {error && <p className="error">{error}</p>}
          <button className="primary" disabled={!file} onClick={start}>
            Add bookmarks
          </button>
        </section>
      )}

      {phase === "uploading" && (
        <section className="card center">
          <p>Uploading {file?.name}…</p>
          <progress value={progress} max="1" />
        </section>
      )}

      {phase === "processing" && (
        <section className="card center">
          <div className="spinner" />
          <p>Detecting the outline… this can take a minute when the LLM runs.</p>
        </section>
      )}

      {phase === "done" && (
        <section className="card center">
          <p className="success">
            Done — {result.bookmark_count} bookmark
            {result.bookmark_count === 1 ? "" : "s"} added.
          </p>
          <a className="primary button" href={downloadUrl(jobId)}>
            Download PDF
          </a>
          <p className="note">Files are deleted from the server after 1 hour.</p>
          <button className="linklike" onClick={reset}>
            Process another file
          </button>
        </section>
      )}

      {phase === "failed" && (
        <section className="card center">
          <p className="error">{error}</p>
          <button className="primary" onClick={reset}>
            Try again
          </button>
        </section>
      )}
    </main>
  );
}
