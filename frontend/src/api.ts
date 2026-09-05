const BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export type LlmMode = "auto" | "always" | "never";

/** Mirrors `Job.status` in backend/app/jobs.py. */
export type JobStatus = "queued" | "processing" | "done" | "failed";

/**
 * The body of `GET /api/jobs/{id}`. The backend omits `error` and
 * `bookmark_count` while they are null, so both are optional here.
 */
export interface JobState {
  status: JobStatus;
  error?: string;
  bookmark_count?: number;
}

export interface CreateJobOptions {
  llmMode: LlmMode;
  model?: string;
  apiKey?: string;
}

// XMLHttpRequest instead of fetch: fetch has no upload-progress events.
export function createJob(
  file: File,
  { llmMode, model, apiKey }: CreateJobOptions,
  onProgress?: (fraction: number) => void
): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  form.append("llm_mode", llmMode);
  if (model) form.append("model", model);
  if (apiKey) form.append("api_key", apiKey);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/api/jobs`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      const body: { job_id: string; detail?: string } | null = xhr.response;
      if (xhr.status === 202 && body) resolve(body.job_id);
      else reject(new Error(body?.detail ?? `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Network error during upload."));
    xhr.send(form);
  });
}

export async function getJob(jobId: string): Promise<JobState> {
  const res = await fetch(`${BASE}/api/jobs/${jobId}`);
  if (res.status === 404) {
    throw new Error("This job has expired — please upload the file again.");
  }
  if (!res.ok) throw new Error(`Status check failed (${res.status}).`);
  return res.json() as Promise<JobState>;
}

export function downloadUrl(jobId: string): string {
  return `${BASE}/api/jobs/${jobId}/download`;
}
